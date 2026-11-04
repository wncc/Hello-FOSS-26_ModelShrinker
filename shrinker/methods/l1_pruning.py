"""Removes the filters/neurons whose weights have the smallest L1 norm (sum of absolute values)."""

from torch import nn

from shrinker.registry import register
from shrinker.utils import keep_indices, prunable_layers, prune_channels


@register(kind="pruning", paper="https://arxiv.org/abs/1608.08710")
def apply(model: nn.Module, ratio: float = 0.3) -> nn.Module:
    """Remove the filters/neurons with the smallest L1 norm from every prunable layer."""
    for name, layer in prunable_layers(model):
        scores = layer.weight.detach().abs().flatten(1).sum(dim=1)
        keep = keep_indices(scores, ratio)
        prune_channels(model, name, keep)
    return model
