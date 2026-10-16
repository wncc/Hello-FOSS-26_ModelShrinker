"""Finds which layers can be pruned safely, and prunes a layer together with its
BatchNorm and the next layer's inputs.

Layers whose outputs are added together (residual connections) must keep the same channels,
so they are pruned as one group: the same channel indices are removed from every layer of the
group, from their BatchNorms and from the layers that read from them. Anything unsafe (concat,
model output, depthwise conv, an add with the model input, ...) is skipped with a reason."""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch import nn

from .utils import (
    Indices,
    get_module,
    replace_module,
    slice_bn,
    slice_conv_in,
    slice_conv_out,
    slice_linear_in,
    slice_linear_out,
)

logger = logging.getLogger(__name__)

_ELEMENTWISE_MODULES = (
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh,
    nn.Hardswish, nn.Hardsigmoid, nn.Mish, nn.PReLU,
    nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.AlphaDropout, nn.Identity,
)
_POOL_MODULES = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)
_ELEMENTWISE_FUNCS = {
    F.relu, F.relu6, F.leaky_relu, F.elu, F.gelu, F.silu, F.hardswish, F.mish,
    F.dropout, F.dropout1d, F.dropout2d, torch.relu, torch.sigmoid, torch.tanh,
    F.sigmoid, F.tanh,
}
_POOL_FUNCS = {F.max_pool2d, F.avg_pool2d, F.adaptive_avg_pool2d, F.adaptive_max_pool2d}
_ELEMENTWISE_METHODS = {"relu", "relu_", "sigmoid", "tanh", "contiguous"}
_IGNORED_METHODS = {"size", "dim"}


@dataclass
class ChannelGroup:
    """Layers whose output channels must be pruned together, plus everything that depends on them.

    ``producer`` is the first layer of the group in execution order. ``producers`` lists every
    layer of the group (it includes ``producer``): more than one when their outputs are added
    together by a residual connection.
    """

    producer: str
    batchnorms: List[str] = field(default_factory=list)
    # spatial = H*W inputs per channel at a Conv -> Flatten -> Linear boundary, else 1
    consumers: List[Tuple[str, int]] = field(default_factory=list)
    producers: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.producer not in self.producers:
            self.producers.insert(0, self.producer)


class _Skip(Exception):
    pass


def analyze(model: nn.Module) -> Tuple[Dict[str, ChannelGroup], Dict[str, str]]:
    """Return ``(prunable, skipped)``: {name: ChannelGroup} and {name: reason}.

    Every prunable layer is a key. Layers tied together by a residual add share the same
    ``ChannelGroup`` object, so use ``group.producer`` to tell groups apart.
    """
    weight_layers = [n for n, m in model.named_modules() if type(m) in (nn.Conv2d, nn.Linear)]
    try:
        graph = fx.symbolic_trace(model).graph
    except Exception as e:  # noqa: BLE001
        reason = f"model could not be traced with torch.fx ({type(e).__name__}: {e})"
        return {}, {n: reason for n in weight_layers}

    modules = dict(model.named_modules())
    call_count: Dict[str, int] = {}
    for node in graph.nodes:
        if node.op == "call_module":
            call_count[node.target] = call_count.get(node.target, 0) + 1

    prunable: Dict[str, ChannelGroup] = {}
    skipped: Dict[str, str] = {}
    for node in graph.nodes:
        if node.op != "call_module" or type(modules[node.target]) not in (nn.Conv2d, nn.Linear):
            continue
        name = node.target
        if name in prunable or name in skipped:
            continue
        try:
            group = _group_for(node, modules, call_count)
        except _Skip as e:
            skipped[name] = str(e)
            continue
        for member in group.producers:
            prunable[member] = group

    for n in weight_layers:
        if n not in prunable and n not in skipped:
            skipped[n] = "not used in forward()"
    return prunable, skipped


def _out_size(layer: nn.Module) -> int:
    return layer.out_channels if type(layer) is nn.Conv2d else layer.out_features


def _producer_problem(name: str, modules, call_count) -> Optional[str]:
    """Why layer ``name`` can't have its output channels pruned, or None if it can."""
    if call_count[name] != 1:
        return "module is called more than once in forward()"
    layer = modules[name]
    if type(layer) is nn.Conv2d and layer.groups != 1:
        return "grouped/depthwise convolution"
    return None


def _group_for(node: fx.Node, modules, call_count) -> ChannelGroup:
    name = node.target
    problem = _producer_problem(name, modules, call_count)
    if problem:
        raise _Skip(problem)
    is_conv = type(modules[name]) is nn.Conv2d
    channels = _out_size(modules[name])
    group = ChannelGroup(producer=name)

    # Layers whose outputs are added together join the group, so we keep following the
    # outputs of every layer that joins until no new layer is found.
    pending = [node]
    visited = set()
    while pending:
        producer = pending.pop()
        stack = [(u, False) for u in producer.users]
        while stack:
            cur, flat = stack.pop()
            if (cur, flat) in visited:
                continue
            visited.add((cur, flat))
            kind, extra = _classify(cur, modules)

            if kind == "ignore":
                continue
            if kind == "output":
                raise _Skip("its output is (part of) the model output")
            if kind == "unsupported":
                raise _Skip(f"its output feeds `{extra}`, which pruning can't pass through")

            mod = modules.get(cur.target) if cur.op == "call_module" else None
            if kind in ("conv", "linear", "batchnorm") and call_count[cur.target] != 1:
                raise _Skip(f"`{cur.target}` is called more than once in forward()")

            if kind == "conv":
                if flat or not is_conv:
                    raise _Skip(f"`{cur.target}` gets a reshaped input")
                if mod.groups != 1:
                    raise _Skip(f"it feeds grouped/depthwise conv `{cur.target}`")
                if mod.in_channels != channels:
                    raise _Skip(f"channel mismatch with `{cur.target}`")
                _add_consumer(group, cur.target, 1)
            elif kind == "linear":
                if is_conv and not flat:
                    raise _Skip(f"`{cur.target}` is applied to a 4D conv output without flattening")
                if mod.in_features % channels != 0:
                    raise _Skip(f"`{cur.target}` in_features ({mod.in_features}) is not a multiple of {channels}")
                spatial = mod.in_features // channels
                if spatial != 1 and not (is_conv and flat):
                    raise _Skip(f"channel mismatch with `{cur.target}`")
                _add_consumer(group, cur.target, spatial)
            elif kind == "batchnorm":
                expect = nn.BatchNorm2d if is_conv else nn.BatchNorm1d
                if flat or type(mod) is not expect or mod.num_features != channels:
                    raise _Skip(f"unexpected BatchNorm `{cur.target}`")
                if cur.target not in group.batchnorms:
                    group.batchnorms.append(cur.target)
                stack.extend((u, flat) for u in cur.users)
            elif kind == "elementwise":
                stack.extend((u, flat) for u in cur.users)
            elif kind == "pool":
                if flat or not is_conv:
                    raise _Skip(f"pooling after flatten at `{cur}`")
                stack.extend((u, flat) for u in cur.users)
            elif kind == "flatten":
                if not is_conv:
                    raise _Skip("a Linear output is reshaped")
                if extra is not None:
                    raise _Skip(extra)
                stack.extend((u, True) for u in cur.users)
            elif kind == "add":
                if flat:
                    raise _Skip("its output is added to another tensor after flattening (residual connection)")
                pending.extend(_join_residual(cur, modules, call_count, is_conv, channels, group))
                stack.extend((u, flat) for u in cur.users)

    if not group.consumers:
        raise _Skip("no following layer found")
    return group


def _join_residual(add_node: fx.Node, modules, call_count, is_conv: bool, channels: int, group: ChannelGroup):
    """Put every layer that feeds ``add_node`` into ``group``. Returns the nodes of the new layers."""
    joined = []
    for operand in _add_operands(add_node):
        for producer in _producers_behind(operand, modules):
            pname = producer.target
            if pname in group.producers:
                continue
            problem = _producer_problem(pname, modules, call_count)
            if problem:
                raise _Skip(f"it is added (residual connection) to `{pname}`, which can't be pruned: {problem}")
            other = modules[pname]
            if (type(other) is nn.Conv2d) != is_conv or _out_size(other) != channels:
                raise _Skip(f"channel mismatch with `{pname}` in a residual connection")
            group.producers.append(pname)
            joined.append(producer)
    return joined


def _add_operands(add_node: fx.Node) -> List[fx.Node]:
    operands = list(add_node.args[:2])
    operands += [v for k, v in add_node.kwargs.items() if k in ("input", "other")]
    if len(operands) != 2 or not all(isinstance(o, fx.Node) for o in operands):
        raise _Skip("it is added to a constant (residual connection), which pruning can't handle")
    return operands


def _producers_behind(node: fx.Node, modules) -> List[fx.Node]:
    """The Conv2d/Linear nodes whose output channels reach ``node`` unchanged (through
    BatchNorm, activations, pooling and other adds)."""
    found, seen, stack = [], set(), [node]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur.op == "placeholder":
            raise _Skip("it is added to the model input (residual connection), which can't be pruned")
        kind, extra = _classify(cur, modules)
        if kind in ("conv", "linear"):
            found.append(cur)
        elif kind in ("batchnorm", "elementwise", "pool"):
            source = cur.args[0] if cur.args else None
            if not isinstance(source, fx.Node):
                raise _Skip(f"it is added (residual connection) to `{cur}`, whose input can't be traced")
            stack.append(source)
        elif kind == "add":
            stack.extend(_add_operands(cur))
        else:
            what = extra if kind == "unsupported" and extra else str(cur)
            raise _Skip(f"it is added (residual connection) to `{what}`, which can't be traced back to a Conv2d/Linear layer")
    return found


def _add_consumer(group: ChannelGroup, name: str, spatial: int) -> None:
    if all(c[0] != name for c in group.consumers):
        group.consumers.append((name, spatial))


def _classify(node: fx.Node, modules) -> Tuple[str, object]:
    if node.op == "output":
        return "output", None
    if node.op == "call_module":
        m = modules[node.target]
        t = type(m)
        if t is nn.Conv2d:
            return "conv", None
        if t is nn.Linear:
            return "linear", None
        if t in (nn.BatchNorm1d, nn.BatchNorm2d):
            return "batchnorm", None
        if isinstance(m, _ELEMENTWISE_MODULES):
            if isinstance(m, nn.PReLU) and m.num_parameters != 1:
                return "unsupported", node.target
            return "elementwise", None
        if isinstance(m, _POOL_MODULES):
            return "pool", None
        if t is nn.Flatten:
            ok = m.start_dim == 1 and m.end_dim == -1
            return "flatten", None if ok else "nn.Flatten must use start_dim=1, end_dim=-1"
        return "unsupported", f"{node.target} ({t.__name__})"
    if node.op == "call_function":
        if node.target in _ELEMENTWISE_FUNCS:
            return "elementwise", None
        if node.target in _POOL_FUNCS:
            return "pool", None
        if node.target is torch.flatten:
            start = node.kwargs.get("start_dim", node.args[1] if len(node.args) > 1 else 0)
            end = node.kwargs.get("end_dim", node.args[2] if len(node.args) > 2 else -1)
            ok = start == 1 and end == -1
            return "flatten", None if ok else "torch.flatten must use start_dim=1"
        if node.target is getattr and len(node.args) > 1 and node.args[1] in ("shape", "ndim", "device", "dtype"):
            return "ignore", None
        if node.target in (torch.reshape,):
            return _classify_reshape(node.args[1:] if len(node.args) > 1 else [node.kwargs.get("shape")])
        if node.target in (operator.add, operator.iadd, torch.add):
            return "add", None
        return "unsupported", getattr(node.target, "__name__", str(node.target))
    if node.op == "call_method":
        if node.target in _ELEMENTWISE_METHODS:
            return "elementwise", None
        if node.target in _IGNORED_METHODS:
            return "ignore", None
        if node.target == "flatten":
            start = node.kwargs.get("start_dim", node.args[1] if len(node.args) > 1 else 0)
            end = node.kwargs.get("end_dim", node.args[2] if len(node.args) > 2 else -1)
            ok = start == 1 and end == -1
            return "flatten", None if ok else "x.flatten must use start_dim=1"
        if node.target in ("view", "reshape"):
            return _classify_reshape(node.args[1:])
        return "unsupported", f".{node.target}()"
    return "unsupported", str(node)


def _classify_reshape(shape_args) -> Tuple[str, object]:
    if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
        shape_args = shape_args[0]
    if len(shape_args) == 2 and shape_args[1] == -1:
        return "flatten", None
    return "flatten", "reshape with a hard-coded size (use x.flatten(1) or x.view(x.size(0), -1))"


def prunable_layers(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yield ``(name, layer)`` once per prunable group, in execution order.

    Layers tied together by a residual add are one group and are yielded as their first layer;
    ``prune_channels`` on that layer prunes the whole group, so every group is pruned only once.
    """
    prunable, skipped = analyze(model)
    for name, reason in skipped.items():
        logger.info("Skipping layer %s: %s", name, reason)
    for name, group in prunable.items():
        if name == group.producer:
            yield name, get_module(model, name)


def skipped_layers(model: nn.Module) -> Dict[str, str]:
    """``{layer name: reason}`` for every Conv2d/Linear that can't be pruned."""
    return analyze(model)[1]


def prune_channels(model: nn.Module, name: str, keep: Indices) -> nn.Module:
    """Keep only output channels ``keep`` of layer ``name``; its BatchNorm and next layers are sliced too.

    If ``name`` is tied to other layers by a residual add, the same channels are kept in all of them.
    """
    prunable, skipped = analyze(model)
    if name not in prunable:
        reason = skipped.get(name, "not a Conv2d/Linear layer in this model")
        raise ValueError(f"Layer {name!r} can't be pruned: {reason}")
    group = prunable[name]
    n = _out_size(get_module(model, name))
    keep = torch.unique(torch.as_tensor(keep, dtype=torch.long).flatten())
    if keep.numel() == 0 or keep[0] < 0 or keep[-1] >= n:
        raise ValueError(f"keep must be non-empty indices in [0, {n}) for layer {name!r}")

    for member in group.producers:
        layer = get_module(model, member)
        if type(layer) is nn.Conv2d:
            replace_module(model, member, slice_conv_out(layer, keep))
        else:
            replace_module(model, member, slice_linear_out(layer, keep))
    for bn in group.batchnorms:
        replace_module(model, bn, slice_bn(get_module(model, bn), keep))
    for cname, spatial in group.consumers:
        consumer = get_module(model, cname)
        if type(consumer) is nn.Conv2d:
            replace_module(model, cname, slice_conv_in(consumer, keep))
        else:
            # channel c owns flattened inputs [c*spatial, (c+1)*spatial)
            offsets = torch.arange(spatial, device=keep.device)
            idx = (keep[:, None] * spatial + offsets[None, :]).flatten()
            replace_module(model, cname, slice_linear_in(consumer, idx))
    return model
