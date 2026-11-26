"""Exact FLOPs/parameter count of a model after channel pruning, without pruning it.

Pruning a group keeps ``max(1, round(n * (1 - ratio)))`` of its ``n`` channels (see
``keep_indices``). A Conv2d/Linear loses output channels when its own group is pruned and input
channels when the group feeding it is pruned, so its cost scales with
``kept_out / out * kept_in / in``. Everything else stays the same.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from torch import nn

from . import benchmark
from .dependency import analyze
from .utils import unfused_transformer_layers


def kept_channels(n: int, ratio: float) -> int:
    """How many of ``n`` channels ``keep_indices`` keeps at ``ratio``."""
    return max(1, int(round(n * (1.0 - ratio))))


class PruningCost:
    """``cost(ratios)`` gives the FLOPs (``metric="flops"``) or parameter count
    (``metric="params"``) that ``count_flops``/``count_parameters`` would report after
    ``compress(model, "l1_pruning", ratio=ratios)``.

    ``groups`` lists the prunable layers in execution order (the keys ``l1_pruning`` accepts),
    and ``full`` is the cost of the unpruned model.
    """

    def __init__(self, model: nn.Module, example_input: Any = None, metric: str = "flops"):
        if metric not in ("flops", "params"):
            raise ValueError(f"metric must be 'flops' or 'params', got {metric!r}")
        if metric == "flops" and example_input is None:
            raise ValueError("example_input is needed to count FLOPs")

        prunable, _ = analyze(model)
        groups = {g.producer: g for g in prunable.values()}
        self.groups: List[str] = list(groups)
        self.sizes: Dict[str, int] = {}
        out_of: Dict[str, str] = {}
        in_of: Dict[str, str] = {}
        for producer, g in groups.items():
            layer = model.get_submodule(producer)
            self.sizes[producer] = layer.out_channels if type(layer) is nn.Conv2d else layer.out_features
            for member in g.producers:
                out_of[member] = producer
            for consumer, _ in g.consumers:
                in_of[consumer] = producer

        # (cost, group of its outputs or None, group of its inputs or None)
        self._terms: List[Tuple[int, Any, Any]] = []
        if metric == "flops":
            self.full = benchmark.count_flops(model, example_input) or 0
            layer_flops = _conv_linear_flops(model, example_input)
            fixed = self.full - sum(layer_flops.values())
            for name, flops in layer_flops.items():
                self._terms.append((flops, out_of.get(name), in_of.get(name)))
        else:
            self.full = benchmark.count_parameters(model)
            fixed = self.full
            bn_of = {bn: producer for producer, g in groups.items() for bn in g.batchnorms}
            for name, m in model.named_modules():
                if type(m) in (nn.Conv2d, nn.Linear) and (name in out_of or name in in_of):
                    out_g, in_g = out_of.get(name), in_of.get(name)
                    self._terms.append((m.weight.numel(), out_g, in_g))
                    fixed -= m.weight.numel()
                    if m.bias is not None:
                        self._terms.append((m.bias.numel(), out_g, None))
                        fixed -= m.bias.numel()
                elif name in bn_of:
                    for p in m.parameters(recurse=False):
                        self._terms.append((p.numel(), bn_of[name], None))
                        fixed -= p.numel()
        self._fixed = fixed

    def __call__(self, ratios: Dict[str, float]) -> int:
        kept = {g: kept_channels(n, ratios.get(g, 0.0)) for g, n in self.sizes.items()}
        total = self._fixed
        for cost, out_g, in_g in self._terms:
            num, den = 1, 1
            for g in (out_g, in_g):
                if g is not None:
                    num *= kept[g]
                    den *= self.sizes[g]
            total += cost * num // den
        return total

    def fraction(self, ratios: Dict[str, float]) -> float:
        """Share of the full cost that is left after pruning with ``ratios``."""
        return self(ratios) / self.full if self.full else 1.0


def _conv_linear_flops(model: nn.Module, example_input: Any) -> Dict[str, int]:
    """FLOPs of each float Conv2d/Linear, counted the same way as ``count_flops``."""
    flops: Dict[str, int] = {}
    handles = []

    def hook(name):
        def fn(m, inp, out):
            per_out = m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1] if type(m) is nn.Conv2d else m.in_features
            flops[name] = flops.get(name, 0) + 2 * out.numel() * per_out
        return fn

    for name, m in model.named_modules():
        if type(m) in (nn.Conv2d, nn.Linear):
            handles.append(m.register_forward_hook(hook(name)))
    was_training = model.training
    model.eval()
    try:
        with unfused_transformer_layers(example_input) as x:
            model(*x) if isinstance(x, tuple) else model(x)
    finally:
        model.train(was_training)
        for h in handles:
            h.remove()
    return flops
