"""Removes the filters/neurons whose weights have the smallest L1 norm (sum of absolute values)."""

from typing import Dict, Union

from torch import nn

from shrinker.registry import register
from shrinker.utils import keep_indices, prunable_layers, prune_channels, resolve_ratios


@register
def apply(model: nn.Module, ratio: Union[float, Dict[str, float]] = 0.3) -> nn.Module:
    """Remove the filters/neurons with the smallest L1 norm from every prunable layer.

    ratio: one fraction used for every layer, or {layer name: fraction}. Layers missing
    from the dict are left unpruned.
    """
    ratios = resolve_ratios(model, ratio)
    for name, layer in prunable_layers(model):
        if name not in ratios:
            continue
        scores = layer.weight.detach().abs().flatten(1).sum(dim=1)
        keep = keep_indices(scores, ratios[name])
        prune_channels(model, name, keep)
    return model