import copy

import pytest
import torch
from torch import nn

from examples.test_models import SmallTransformer
from shrinker import compress
from shrinker.attention import PrunedMultiheadAttention
from shrinker.benchmark import count_flops
from tests.helpers import make_loader

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


class MaskedEncoder(nn.Module):
    """Default enable_nested_tensor=True; takes the masks as arguments."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 32)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, batch_first=True), 2)
        self.head = nn.Linear(32, 10)

    def forward(self, x, mask=None, padding=None):
        return self.head(self.encoder(self.proj(x), mask=mask, src_key_padding_mask=padding)[:, 0])


def layers(model):
    return [m for m in model.modules() if isinstance(m, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer))]


def zero_removed_heads(original, pruned):
    ref = copy.deepcopy(original)
    for layer, small in zip(layers(ref), layers(pruned)):
        if isinstance(small.self_attn, PrunedMultiheadAttention):
            attn, d = layer.self_attn, layer.self_attn.head_dim
            with torch.no_grad():
                for h in set(range(attn.num_heads)) - set(small.self_attn.kept.tolist()):
                    attn.out_proj.weight[:, h * d:(h + 1) * d] = 0
    return ref


def test_heads_are_removed():
    pruned = compress(SmallTransformer(), "head_pruning", ratio=0.5)
    for layer in pruned.encoder.layers:
        assert layer.self_attn.num_heads == 2
        assert layer.self_attn.out_proj.in_features == 16 and layer.self_attn.out_proj.out_features == 32


@pytest.mark.parametrize("cls", [SmallTransformer, PreNormEncoder, DecoderOnly, Seq2Seq])
def test_pruned_model_equals_original_with_removed_heads_zeroed(cls):
    original = cls().eval()
    pruned = compress(original, "head_pruning", ratio=0.5).eval()
    with torch.no_grad():
        assert torch.allclose(pruned(X), zero_removed_heads(original, pruned)(X), atol=1e-5)


def test_cross_attention_is_left_alone():
    pruned = compress(DecoderOnly(), "head_pruning", ratio=0.5)
    for layer in pruned.decoder.layers:
        assert isinstance(layer.self_attn, PrunedMultiheadAttention)
        assert type(layer.multihead_attn) is nn.MultiheadAttention


def _masks():
    padding = torch.zeros(3, 6, dtype=torch.bool)
    padding[:, 4:] = True
    per_head = torch.rand(3 * 4, 6, 6) > 0.7
    per_head[..., 0] = False  # every query can see at least one key
    return {
        "float causal": {"mask": nn.Transformer.generate_square_subsequent_mask(6)},
        "bool causal": {"mask": torch.triu(torch.ones(6, 6, dtype=torch.bool), 1)},
        "per head": {"mask": per_head},
        "padding": {"padding": padding},
    }


@pytest.mark.parametrize("name", list(_masks()))
def test_masks(name):
    kwargs = _masks()[name]
    original = MaskedEncoder().eval()
    pruned = compress(original, "head_pruning", ratio=0.5).eval()
    reference = zero_removed_heads(original, pruned)
    reference.encoder.use_nested_tensor = False  # the nested path writes zeros on padded rows
    with torch.no_grad():
        assert torch.allclose(pruned(X, **kwargs), reference(X, **kwargs), atol=1e-5)


def test_ratio_dict_and_unknown_name():
    pruned = compress(SmallTransformer(), "head_pruning", ratio={"encoder.layers.1": 0.5})
    assert type(pruned.encoder.layers[0].self_attn) is nn.MultiheadAttention
    assert pruned.encoder.layers[1].self_attn.num_heads == 2
    with pytest.raises(ValueError, match="not one of the"):
        compress(SmallTransformer(), "head_pruning", ratio={"encoder": 0.5})


def test_frozen_weights_stay_frozen():
    pruned = compress(SmallTransformer().requires_grad_(False), "head_pruning", ratio=0.5)
    assert not any(p.requires_grad for p in pruned.parameters())


def test_counted_flops_go_down():
    model = SmallTransformer()
    assert count_flops(compress(model, "head_pruning", ratio=0.5), X) < count_flops(model, X)


def test_refit_brings_outputs_closer_to_the_original():
    model = SmallTransformer().eval()
    plain = compress(model, "head_pruning", ratio=0.5).eval()
    refit = compress(model, "head_pruning", ratio=0.5, loader=make_loader((6, 16), n=512, batch_size=64)).eval()
    x = torch.randn(256, *(6, 16))
    with torch.no_grad():
        target = model(x)
        assert (refit(x) - target).pow(2).mean() < (plain(x) - target).pow(2).mean()


def test_refit_is_skipped_with_too_little_data():
    model = SmallTransformer()
    plain = compress(model, "head_pruning", ratio=0.5)
    tiny = compress(model, "head_pruning", ratio=0.5, loader=make_loader((6, 16), n=4, batch_size=2))
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), tiny.state_dict().values()))
