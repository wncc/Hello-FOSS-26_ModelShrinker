import copy

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from examples.test_models import SmallGRU, SmallLSTM
from shrinker import compress
from tests.helpers import make_loader

X = torch.randn(3, 6, 16)


class SeqFirstLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(16, 20, num_layers=3, bidirectional=True)
        self.fc = nn.Linear(40, 10)

    def forward(self, x):
        out, _ = self.rnn(x.transpose(0, 1))
        return self.fc(out[-1])


class LastHiddenLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(16, 24, num_layers=2, batch_first=True)
        self.fc = nn.Linear(24, 10)

    def forward(self, x):
        _, (h, c) = self.rnn(x)
        return self.fc(h[-1])


class PackedGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(16, 24, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(48, 10)

    def forward(self, x, lengths):
        out, _ = self.rnn(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(out[:, 0])


def zero_removed_units(original, pruned):
    """The original model with the units that ``pruned`` dropped disconnected."""
    ref = copy.deepcopy(original)
    rnn, small = ref.rnn, pruned.rnn
    gates, hidden = (4 if isinstance(rnn, nn.LSTM) else 3), rnn.hidden_size
    kept = [j for j in range(hidden) if any(torch.equal(rnn.weight_ih_l0[j], row) for row in small.weight_ih_l0)]
    dropped = torch.tensor(sorted(set(range(hidden)) - set(kept)))
    with torch.no_grad():
        for p in rnn.parameters():
            p[torch.cat([dropped + g * hidden for g in range(gates)])] = 0
    return ref


def test_hidden_size_and_head_shrink():
    lstm = compress(SmallLSTM(), "rnn_hidden_pruning", ratio=0.5)
    assert lstm.rnn.hidden_size == 16 and lstm.fc.in_features == 16
    gru = compress(SmallGRU(), "rnn_hidden_pruning", ratio=0.5)
    assert gru.rnn.hidden_size == 12 and gru.fc.in_features == 24


@pytest.mark.parametrize("cls", [SmallLSTM, SmallGRU, SeqFirstLSTM, LastHiddenLSTM])
def test_pruned_model_equals_original_with_removed_units_zeroed(cls):
    original = cls().eval()
    pruned = compress(original, "rnn_hidden_pruning", ratio=0.4).eval()
    assert pruned.rnn.hidden_size < original.rnn.hidden_size
    with torch.no_grad():
        assert torch.allclose(pruned(X), zero_removed_units(original, pruned)(X), atol=1e-6)


def test_packed_sequences():
    original, lengths = PackedGRU().eval(), torch.tensor([6, 4, 2])
    pruned = compress(original, "rnn_hidden_pruning", example_input=(X, lengths), ratio=0.4).eval()
    assert pruned.rnn.hidden_size < original.rnn.hidden_size
    with torch.no_grad():
        assert torch.allclose(pruned(X, lengths), zero_removed_units(original, pruned)(X, lengths), atol=1e-6)


def test_keeps_units_with_largest_weights():
    model = SmallLSTM()
    with torch.no_grad():
        for name, p in model.rnn.named_parameters():
            if name.startswith("weight"):
                p.view(4, 32, -1)[:, :16] *= 1e-3  # units 0-15 become the weakest
    pruned = compress(model, "rnn_hidden_pruning", ratio=0.5)
    assert torch.equal(pruned.rnn.weight_hh_l0[:16], model.rnn.weight_hh_l0[16:32, 16:])


class _UsesCellState(SmallLSTM):
    def forward(self, x):
        _, (h, c) = self.rnn(x)
        return self.fc(c[-1])


class _GivenH0(SmallLSTM):
    def forward(self, x):
        h0 = torch.zeros(2, x.shape[0], 32)
        out, _ = self.rnn(x, (h0, h0))
        return self.fc(out[:, -1])


class _ConcatDirections(SmallGRU):
    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))


class _FeatureMean(SmallLSTM):
    """Averages over features; with 32 time steps the head still gets 32 inputs."""

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out.mean(dim=2))


class _Projected(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(16, 32, batch_first=True, proj_size=8)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


@pytest.mark.parametrize("cls", [_UsesCellState, _GivenH0, _ConcatDirections, _FeatureMean, _Projected])
def test_unsupported_uses_leave_the_rnn_unchanged(cls):
    x = torch.randn(3, 32, 16)
    pruned = compress(cls(), "rnn_hidden_pruning", example_input=x)
    assert pruned.rnn.hidden_size == cls().rnn.hidden_size


def test_ratio_dict_and_unknown_name():
    pruned = compress(SmallLSTM(), "rnn_hidden_pruning", ratio={"rnn": 0.25})
    assert pruned.rnn.hidden_size == 24
    with pytest.raises(ValueError, match="not one of the"):
        compress(SmallLSTM(), "rnn_hidden_pruning", ratio={"fc": 0.5})


def test_ratio_zero_changes_nothing():
    model = SmallGRU().eval()
    pruned = compress(model, "rnn_hidden_pruning", ratio=0.0).eval()
    with torch.no_grad():
        assert torch.equal(pruned(X), model(X))


def test_frozen_weights_stay_frozen():
    model = SmallLSTM().requires_grad_(False)
    pruned = compress(model, "rnn_hidden_pruning", ratio=0.5)
    assert not any(p.requires_grad for p in pruned.parameters())


def test_refit_brings_outputs_closer_to_the_original():
    model = SmallLSTM().eval()
    plain = compress(model, "rnn_hidden_pruning", ratio=0.5).eval()
    refit = compress(model, "rnn_hidden_pruning", ratio=0.5, loader=make_loader((7, 16), n=512, batch_size=64)).eval()
    x = torch.randn(256, *(7, 16))
    with torch.no_grad():
        target = model(x)
        assert (refit(x) - target).pow(2).mean() < (plain(x) - target).pow(2).mean()


def test_refit_is_skipped_with_too_little_data():
    model = SmallLSTM()
    plain = compress(model, "rnn_hidden_pruning", ratio=0.5)
    tiny = compress(model, "rnn_hidden_pruning", ratio=0.5, loader=make_loader((7, 16), n=4, batch_size=2))
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), tiny.state_dict().values()))
