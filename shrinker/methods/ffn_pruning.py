"""Removes feed-forward neurons from nn.TransformerEncoderLayer / nn.TransformerDecoderLayer.

Paper: Kwon et al. 2022, A Fast Post-Training Pruning Framework for Transformers, https://arxiv.org/abs/2204.09656

Neuron j of a layer's feed-forward block is row j of linear1 and column j of linear2, so removing it
changes nothing outside the layer. Neurons are ranked by the L1 norm of both. Hand-written blocks
built from plain nn.Linear layers are already pruned by l1_pruning.
"""

from __future__ import annotations

import logging
from typing import Dict, Union

from torch import nn

from shrinker.registry import register
from shrinker.utils import is_shared, keep_indices, resolve_module_ratios, slice_linear_in, slice_linear_out

logger = logging.getLogger(__name__)

_LAYERS = (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)


@register(kind="pruning", paper="https://arxiv.org/abs/2204.09656")
def apply(model: nn.Module, ratio: Union[float, Dict[str, float]] = 0.3) -> nn.Module:
    """Remove the feed-forward neurons with the smallest L1 norm from every Transformer layer.

    ratio: one fraction used for every layer, or {layer name: fraction} (e.g. "encoder.layers.0").
    Layers missing from the dict are left unpruned.
    """
    layers = {name: m for name, m in model.named_modules() if type(m) in _LAYERS}
    if not layers:
        return model
    ratios = resolve_module_ratios(layers, ratio, "Transformer layers")

    for name, layer in layers.items():
        if name not in ratios:
            continue
        if is_shared(model, layer) or type(layer.linear1) is not nn.Linear or type(layer.linear2) is not nn.Linear:
            logger.info("Skipping %s: shared layer or non-standard feed-forward block", name)
            continue
        scores = layer.linear1.weight.detach().abs().sum(1) + layer.linear2.weight.detach().abs().sum(0)
        if layer.linear1.bias is not None:
            scores = scores + layer.linear1.bias.detach().abs()
        keep = keep_indices(scores, ratios[name])
        layer.linear1 = slice_linear_out(layer.linear1, keep)
        layer.linear2 = slice_linear_in(layer.linear2, keep)
    return model
