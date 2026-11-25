"""Removes the filters closest to the layer's geometric median, i.e. the most redundant ones.

Paper: He et al. 2019, Filter Pruning via Geometric Median, https://arxiv.org/abs/1811.00250

A filter near the geometric median of its layer can be represented by the other filters, so
removing it loses little. Following Eq. 4-9 of the paper, the median itself is never computed:
the filters with the smallest total distance to the other filters are the ones to remove.
"""

from typing import Dict, Union

import torch
from torch import nn

from shrinker.registry import register
from shrinker.utils import keep_indices, prunable_layers, prune_channels, resolve_ratios


@register
def apply(model: nn.Module, ratio: Union[float, Dict[str, float]] = 0.3) -> nn.Module:
    """Remove the most redundant filters/neurons: those closest to the geometric median.

    ratio: one fraction used for every layer, or {layer name: fraction}. Layers missing
    from the dict are left unpruned.
    """
    ratios = resolve_ratios(model, ratio)
    for name, layer in prunable_layers(model):
        if name not in ratios:
            continue
        filters = layer.weight.detach().flatten(1).float()
        scores = torch.cdist(filters, filters, p=2).sum(dim=1)
        keep = keep_indices(scores, ratios[name])
        prune_channels(model, name, keep)
    return model