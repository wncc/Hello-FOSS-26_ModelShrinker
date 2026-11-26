import copy

import pytest
import torch
from torch import nn

from examples.test_models import SmallTransformer
from shrinker import compress

X = torch.randn(3, 6, 16)


class PreNormEncoder(SmallTransformer):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)


class DecoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 32)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(32, 4, 64, dropout=0.0, batch_first=True), 2)
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        h = self.proj(x)
        mask = nn.Transformer.generate_square_subsequent_mask(h.shape[1])
        return self.head(self.decoder(h, h, tgt_mask=mask)[:, -1])


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 32)
        self.transformer = nn.Transformer(32, 4, 2, 2, 64, dropout=0.0)
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        src = self.proj(x).transpose(0, 1)
        tgt = src[-3:]
        mask = nn.Transformer.generate_square_subsequent_mask(3)
        return self.head(self.transformer(src, tgt, tgt_mask=mask)[-1])


class PaddedEncoder(nn.Module):
    """Default enable_nested_tensor=True, so eval runs PyTorch's fused fast path."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 32)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, batch_first=True), 2)
        self.head = nn.Linear(32, 10)

    def forward(self, x, padding):
        return self.head(self.encoder(self.proj(x), src_key_padding_mask=padding)[:, 0])


def layers(model):
    return [m for m in model.modules() if isinstance(m, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer))]


def zero_removed_neurons(original, pruned):
    ref = copy.deepcopy(original)
    for layer, small in zip(layers(ref), layers(pruned)):
        w1 = layer.linear1.weight
        kept = {j for j in range(w1.shape[0]) if any(torch.equal(w1[j], row) for row in small.linear1.weight)}
        with torch.no_grad():
            for j in set(range(w1.shape[0])) - kept:
                layer.linear2.weight[:, j] = 0
    return ref


def test_feed_forward_width_shrinks():
    pruned = compress(SmallTransformer(), "ffn_pruning", ratio=0.5)
    for layer in pruned.encoder.layers:
        assert layer.linear1.out_features == 32 and layer.linear2.in_features == 32
        assert layer.linear2.out_features == 32


@pytest.mark.parametrize("cls", [SmallTransformer, PreNormEncoder, DecoderOnly, Seq2Seq])
def test_pruned_model_equals_original_with_removed_neurons_zeroed(cls):
    original = cls().eval()
    pruned = compress(original, "ffn_pruning", ratio=0.4).eval()
    with torch.no_grad():
        assert torch.allclose(pruned(X), zero_removed_neurons(original, pruned)(X), atol=1e-5)


def test_fast_path_with_padding_mask():
    padding = torch.zeros(3, 6, dtype=torch.bool)
    padding[:, 4:] = True
    original = PaddedEncoder().eval()
    pruned = compress(original, "ffn_pruning", ratio=0.4).eval()
    with torch.no_grad():
        expected = zero_removed_neurons(original, pruned)(X, padding)
        assert torch.allclose(pruned(X, padding), expected, atol=1e-5)


def test_ratio_dict_and_unknown_name():
    pruned = compress(SmallTransformer(), "ffn_pruning", ratio={"encoder.layers.0": 0.5})
    assert [layer.linear1.out_features for layer in pruned.encoder.layers] == [32, 64]
    with pytest.raises(ValueError, match="not one of the"):
        compress(SmallTransformer(), "ffn_pruning", ratio={"encoder.layers.5": 0.5})


def test_frozen_weights_stay_frozen():
    pruned = compress(SmallTransformer().requires_grad_(False), "ffn_pruning", ratio=0.5)
    assert not any(p.requires_grad for p in pruned.parameters())
