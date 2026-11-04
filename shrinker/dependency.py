"""Finds which layers can be pruned safely, and prunes a layer together with its
BatchNorm and the next layer's inputs. Anything unsafe (residual add, concat, model
output, depthwise conv, ...) is skipped."""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

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
    producer: str
    batchnorms: List[str] = field(default_factory=list)
    # spatial = H*W inputs per channel at a Conv -> Flatten -> Linear boundary, else 1
    consumers: List[Tuple[str, int]] = field(default_factory=list)


class _Skip(Exception):
    pass


def analyze(model: nn.Module) -> Tuple[Dict[str, ChannelGroup], Dict[str, str]]:
    """Return ``(prunable, skipped)``: {name: ChannelGroup} and {name: reason}."""
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
    seen = set()
    for node in graph.nodes:
        if node.op != "call_module" or type(modules[node.target]) not in (nn.Conv2d, nn.Linear):
            continue
        name = node.target
        if name in seen:
            continue
        seen.add(name)
        try:
            prunable[name] = _group_for(node, modules, call_count)
        except _Skip as e:
            skipped[name] = str(e)

    for n in weight_layers:
        if n not in prunable and n not in skipped:
            skipped[n] = "not used in forward()"
    return prunable, skipped


def _group_for(node: fx.Node, modules, call_count) -> ChannelGroup:
    name = node.target
    layer = modules[name]
    if call_count[name] != 1:
        raise _Skip("module is called more than once in forward()")
    is_conv = type(layer) is nn.Conv2d
    if is_conv and layer.groups != 1:
        raise _Skip("grouped/depthwise convolution")
    channels = layer.out_channels if is_conv else layer.out_features
    group = ChannelGroup(producer=name)

    stack = [(u, False) for u in node.users]
    visited = set()
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

    if not group.consumers:
        raise _Skip("no following layer found")
    return group


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
        name = getattr(node.target, "__name__", str(node.target))
        if node.target in (operator.add, operator.iadd, torch.add):
            name = "+ (residual connection)"
        return "unsupported", name
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
    """Yield ``(name, layer)`` for every prunable layer, in execution order."""
    prunable, skipped = analyze(model)
    for name, reason in skipped.items():
        logger.info("Skipping layer %s: %s", name, reason)
    for name in prunable:
        yield name, get_module(model, name)


def skipped_layers(model: nn.Module) -> Dict[str, str]:
    """``{layer name: reason}`` for every Conv2d/Linear that can't be pruned."""
    return analyze(model)[1]


def prune_channels(model: nn.Module, name: str, keep: Indices) -> nn.Module:
    """Keep only output channels ``keep`` of layer ``name``; its BatchNorm and next layers are sliced too."""
    prunable, skipped = analyze(model)
    if name not in prunable:
        reason = skipped.get(name, "not a Conv2d/Linear layer in this model")
        raise ValueError(f"Layer {name!r} can't be pruned: {reason}")
    group = prunable[name]
    layer = get_module(model, name)
    n = layer.out_channels if type(layer) is nn.Conv2d else layer.out_features
    keep = torch.unique(torch.as_tensor(keep, dtype=torch.long).flatten())
    if keep.numel() == 0 or keep[0] < 0 or keep[-1] >= n:
        raise ValueError(f"keep must be non-empty indices in [0, {n}) for layer {name!r}")

    if type(layer) is nn.Conv2d:
        replace_module(model, name, slice_conv_out(layer, keep))
    else:
        replace_module(model, name, slice_linear_out(layer, keep))
    for bn in group.batchnorms:
        replace_module(model, bn, slice_bn(get_module(model, bn), keep))
    for cname, spatial in group.consumers:
        consumer = get_module(model, cname)
        if type(consumer) is nn.Conv2d:
            replace_module(model, cname, slice_conv_in(consumer, keep))
        else:
            # channel c owns flattened inputs [c*spatial, (c+1)*spatial)
            idx = (keep[:, None] * spatial + torch.arange(spatial, device=keep.device)[None, :]).flatten()
            replace_module(model, cname, slice_linear_in(consumer, idx))
    return model
