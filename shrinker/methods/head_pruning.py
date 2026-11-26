"""Removes whole attention heads from nn.TransformerEncoderLayer / nn.TransformerDecoderLayer.

Paper: Michel et al. 2019, Are Sixteen Heads Really Better than One?, https://arxiv.org/abs/1905.10650

A head owns slices of the query, key and value projections and the matching columns of out_proj.
Heads are ranked by the L1 norm of those weights, and self_attn is replaced by a smaller
PrunedMultiheadAttention. Only self-attention is pruned: the paper found encoder-decoder attention
much more sensitive to losing heads, so the decoder's cross-attention is left alone.
"""

from __future__ import annotations

import logging
from typing import Dict, Union

from torch import nn

from shrinker.attention import PrunedMultiheadAttention
from shrinker.registry import register
from shrinker.utils import is_shared, keep_indices, resolve_module_ratios

logger = logging.getLogger(__name__)

_LAYERS = (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)


@register(kind="pruning", paper="https://arxiv.org/abs/1905.10650")
def apply(model: nn.Module, ratio: Union[float, Dict[str, float]] = 0.25) -> nn.Module:
    """Remove the attention heads with the smallest L1 norm from every Transformer layer.

    ratio: one fraction used for every layer, or {layer name: fraction}. Layers missing from the
    dict are left unpruned. At least one head is always kept; with few heads the steps are coarse
    (4 heads and ratio=0.1 removes none).
    """
    layers = {name: m for name, m in model.named_modules() if type(m) in _LAYERS}
    if not layers:
        return model
    ratios = resolve_module_ratios(layers, ratio, "Transformer layers")

    pruned = False
    for name, layer in layers.items():
        if name not in ratios:
            continue
        attn = layer.self_attn
        if is_shared(model, layer) or not _supported(attn):
            logger.info("Skipping %s: shared layer or unsupported attention", name)
            continue
        keep = keep_indices(_head_scores(attn), ratios[name])
        if len(keep) == attn.num_heads:
            continue
        layer.self_attn = PrunedMultiheadAttention(attn, keep.tolist())
        pruned = True

    if pruned:
        # With a padding mask, nn.TransformerEncoder may convert its input to a nested tensor,
        # which reads self_attn.in_proj_weight of the first layer. The pruned module has none.
        for m in model.modules():
            if isinstance(m, nn.TransformerEncoder):
                m.use_nested_tensor = False
    return model


def _supported(attn: nn.Module) -> bool:
    return (
        type(attn) is nn.MultiheadAttention
        and attn._qkv_same_embed_dim
        and attn.bias_k is None
        and not attn.add_zero_attn
        and attn.num_heads > 1
    )


def _head_scores(attn: nn.MultiheadAttention):
    E, H = attn.embed_dim, attn.num_heads
    d = E // H
    w_q, w_k, w_v = attn.in_proj_weight.detach().abs().view(3, H, d, E)
    w_o = attn.out_proj.weight.detach().abs().view(E, H, d)
    return w_q.sum((1, 2)) + w_k.sum((1, 2)) + w_v.sum((1, 2)) + w_o.sum((0, 2))
