"""Multi-head attention with some heads removed, used by the head_pruning method.

nn.MultiheadAttention always uses ``head_dim = embed_dim // num_heads``, so it can't hold fewer
heads of the same size. ``PrunedMultiheadAttention`` takes its place as the ``self_attn`` of an
nn.TransformerEncoderLayer / nn.TransformerDecoderLayer and is called the same way.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class PrunedMultiheadAttention(nn.Module):
    """The heads of ``mha`` listed in ``keep_heads``, each with its original size."""

    def __init__(self, mha: nn.MultiheadAttention, keep_heads: Sequence[int]):
        super().__init__()
        E, H = mha.embed_dim, mha.num_heads
        self.embed_dim, self.head_dim, self.orig_heads = E, E // H, H
        self.num_heads = len(keep_heads)
        self.batch_first = mha.batch_first
        self.dropout = mha.dropout
        # The Transformer layers check these before using their fused kernel, which reads the
        # weights of nn.MultiheadAttention directly. False sends them down the normal path.
        self._qkv_same_embed_dim = False
        self.in_proj_bias = None
        self.register_buffer("kept", torch.as_tensor(list(keep_heads)), persistent=False)

        d, device, dtype = self.head_dim, mha.in_proj_weight.device, mha.in_proj_weight.dtype
        idx = torch.cat([torch.arange(h * d, (h + 1) * d) for h in keep_heads]).to(device)
        bias = mha.in_proj_bias is not None
        self.q, self.k, self.v = (nn.Linear(E, len(idx), bias=bias, device=device, dtype=dtype) for _ in range(3))
        self.out_proj = nn.Linear(len(idx), E, bias=mha.out_proj.bias is not None, device=device, dtype=dtype)
        with torch.no_grad():
            for i, proj in enumerate((self.q, self.k, self.v)):
                proj.weight.copy_(mha.in_proj_weight[i * E + idx])
                if bias:
                    proj.bias.copy_(mha.in_proj_bias[i * E + idx])
            self.out_proj.weight.copy_(mha.out_proj.weight[:, idx])
            if mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(mha.out_proj.bias)
        for proj in (self.q, self.k, self.v):
            proj.requires_grad_(mha.in_proj_weight.requires_grad)
        self.out_proj.requires_grad_(mha.out_proj.weight.requires_grad)
        self.train(mha.training)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        if not self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        B, L, _ = query.shape
        S = key.shape[1]
        h, d = self.num_heads, self.head_dim
        q = self.q(query).view(B, L, h, d).transpose(1, 2)
        k = self.k(key).view(B, S, h, d).transpose(1, 2)
        v = self.v(value).view(B, S, h, d).transpose(1, 2)
        mask = self._mask(attn_mask, key_padding_mask, B, L, S, q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
            is_causal=bool(is_causal) and mask is None,
        )
        out = self.out_proj(out.transpose(1, 2).reshape(B, L, h * d))
        return (out if self.batch_first else out.transpose(0, 1)), None

    def _mask(self, attn_mask, key_padding_mask, B: int, L: int, S: int, dtype) -> Optional[torch.Tensor]:
        """One additive float mask. In nn.MultiheadAttention a True entry means "don't attend"."""
        mask = None
        if attn_mask is not None:
            mask = _additive(attn_mask, dtype)
            if mask.dim() == 3:  # one mask per original head: (B * heads, L, S)
                mask = mask.view(B, self.orig_heads, L, S)[:, self.kept]
        if key_padding_mask is not None:
            padding = _additive(key_padding_mask, dtype).view(B, 1, 1, S)
            mask = padding if mask is None else mask + padding
        return mask


def _additive(mask: torch.Tensor, dtype) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(mask, float("-inf"))
    return mask.to(dtype)
