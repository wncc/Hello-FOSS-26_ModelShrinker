import logging

import pytest
import torch
from torch import nn

from shrinker import compress
from shrinker.methods._quantization_common import _select_backend


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.net(x)


class _Branchy(nn.Module):
    """Data-dependent control flow, which torch.fx cannot trace."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        if x.sum() > 0:
            return self.linear(x)
        return torch.zeros(x.shape[0], 2)


class _DictLoader:
    def __init__(self):
        self.called = False

    def __iter__(self):
        self.called = True
        return iter([{"x": torch.randn(2, 20)}])


def make_loader():
    x = torch.randn(16, 20)
    y = torch.randint(0, 4, (16,))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)


def test_ptq_converts_supported_model_with_observed_ranges():
    loader = make_loader()
    small = compress(_Mlp().eval(), "ptq_quantization", loader=loader, example_input=torch.randn(2, 20))
    assert any("quantized" in type(m).__module__ for m in small.modules())
    with torch.no_grad():
        out = small(torch.randn(2, 20))
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_ptq_calibration_accepts_unlabeled_batches():
    loader = torch.utils.data.DataLoader(torch.randn(16, 20), batch_size=8)
    small = compress(_Mlp().eval(), "ptq_quantization", loader=loader, example_input=torch.randn(2, 20))
    assert any("quantized" in type(m).__module__ for m in small.modules())


def test_ptq_max_batches_must_be_positive():
    loader = make_loader()
    with pytest.raises(ValueError, match="max_batches"):
        compress(_Mlp(), "ptq_quantization", loader=loader, max_batches=0)


def test_ptq_requires_loader():
    with pytest.raises(ValueError, match="loader"):
        compress(_Mlp(), "ptq_quantization", loader=None)


def test_ptq_returns_model_unchanged_when_tracing_fails(caplog):
    caplog.set_level(logging.INFO)
    small = compress(_Branchy().eval(), "ptq_quantization", loader=make_loader(), example_input=torch.randn(2, 4))
    assert type(small) is _Branchy
    assert type(small.linear) is nn.Linear
    assert "trace and prepare" in caplog.text


def test_ptq_returns_model_unchanged_for_unusable_batches(caplog):
    caplog.set_level(logging.INFO)
    loader = _DictLoader()
    small = compress(_Mlp().eval(), "ptq_quantization", loader=loader, example_input=torch.randn(2, 20))
    assert type(small.net[1]) is nn.Linear
    assert loader.called
    assert "loader must yield" in caplog.text


def test_ptq_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend must be"):
        compress(_Mlp(), "ptq_quantization", loader=make_loader(), backend="nope")


def test_select_backend_rejects_unknown_names():
    with pytest.raises(ValueError, match="backend must be"):
        _select_backend("nope")
