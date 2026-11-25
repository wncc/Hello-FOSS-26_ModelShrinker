import warnings

import pytest
import torch
from torch import nn

from examples.test_models import SmallVGG
from shrinker import compare, compress
from shrinker.benchmark import (
    count_flops,
    count_nonzero_parameters,
    count_parameters,
    measure_latency,
    model_size_mb,
)
from tests.helpers import make_loader

X = torch.randn(1, 3, 32, 32)


def test_compare_report():
    model = SmallVGG()
    small = compress(model, "l1_pruning", ratio=0.5)
    report = compare(model, small, X, val_loader=make_loader((3, 32, 32)), runs=5, warmup=1, verbose=False)
    a, b = report["original"], report["compressed"]
    assert set(a) == {"params", "nonzero_params", "size_mb", "latency_ms", "flops", "accuracy"}
    assert b["params"] < a["params"]
    assert b["size_mb"] < a["size_mb"]
    assert b["flops"] < a["flops"]
    assert a["latency_ms"] > 0 and b["latency_ms"] > 0
    assert 0 <= a["accuracy"] <= 1


def test_compare_prints_table(capsys):
    model = SmallVGG()
    compare(model, compress(model, "l1_pruning"), X, runs=3, warmup=1)
    out = capsys.readouterr().out
    for word in ("params", "size (MB)", "latency (ms)", "MFLOPs"):
        assert word in out
    assert "accuracy" not in out


def test_measurement_restores_training_mode():
    model = SmallVGG().train()
    measure_latency(model, X, warmup=1, runs=2)
    count_flops(model, X)
    assert model.training


def test_flops_hand_computed():
    lin = torch.nn.Linear(10, 5)
    assert count_flops(lin, torch.randn(1, 10)) == 2 * 10 * 5


def test_size_matches_params():
    model = torch.nn.Linear(1000, 1000)
    assert abs(model_size_mb(model) - count_parameters(model) * 4 / 1e6) < 0.01


def test_nonzero_row_only_when_weights_are_zeroed(capsys):
    model = SmallVGG()
    compare(model, compress(model, "l1_pruning"), X, runs=2, warmup=1)
    assert "non-zero" not in capsys.readouterr().out

    sparse = compress(model, "l1_pruning")
    with torch.no_grad():
        sparse.classifier[1].weight[:, ::2] = 0
    report = compare(model, sparse, X, runs=2, warmup=1)
    assert "non-zero params" in capsys.readouterr().out
    assert report["compressed"]["nonzero_params"] < report["compressed"]["params"]


@pytest.fixture
def quantized_engine():
    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not engines:
        pytest.skip("no quantized engine available")
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = engines[-1]
    yield engines[-1]
    if previous in engines:  # 'none' (the default on Apple Silicon) can't be set back, see #14
        torch.backends.quantized.engine = previous


def _quantized_pair(engine):
    """A float Conv/Linear model and its FX-converted INT8 version."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 8, 3, groups=8, bias=False),  # the backend packs a zero bias for this one
        nn.Flatten(),
        nn.Linear(8 * 6 * 6, 16), nn.ReLU(),
        nn.Linear(16, 4),
    ).eval()
    x = torch.randn(2, 3, 8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_fx(model, get_default_qconfig_mapping(engine), (x,))
        prepared(x)
        quantized = convert_fx(prepared)
    assert not list(quantized.parameters()), "expected packed weights, not nn.Parameters"
    return model, quantized, x


def test_quantized_model_reports_its_packed_parameters(quantized_engine):
    model, quantized, _ = _quantized_pair(quantized_engine)
    assert count_parameters(quantized) == count_parameters(model)
    # rounding to INT8 can turn a few tiny weights into exact zeros, but never most of them
    assert 0.9 * count_parameters(model) < count_nonzero_parameters(quantized) <= count_parameters(model)


def test_quantized_model_zeroed_weights_are_not_counted_as_nonzero(quantized_engine):
    model, quantized, _ = _quantized_pair(quantized_engine)
    linear = getattr(quantized, "6")
    weight, bias = linear.weight(), linear.bias()
    zeroed = torch.quantize_per_tensor(
        torch.zeros(weight.shape), scale=1.0, zero_point=0, dtype=torch.qint8
    )
    before = count_nonzero_parameters(quantized)
    linear.set_weight_bias(zeroed, bias)
    assert count_parameters(quantized) == count_parameters(model)
    assert count_nonzero_parameters(quantized) == before - int(torch.count_nonzero(weight.dequantize()))


def test_quantized_model_flops_match_the_float_model(quantized_engine):
    model, quantized, x = _quantized_pair(quantized_engine)
    assert count_flops(quantized, x) == count_flops(model, x) > 0


def test_compare_does_not_report_a_quantized_model_as_empty(quantized_engine, capsys):
    model, quantized, x = _quantized_pair(quantized_engine)
    report = compare(model, quantized, x, runs=2, warmup=1)
    out = capsys.readouterr().out
    assert report["compressed"]["params"] == report["original"]["params"]
    assert report["compressed"]["flops"] == report["original"]["flops"]
    assert "-100.0%" not in out


def test_flops_are_unknown_when_no_conv_or_linear_layer_ran(capsys):
    model = nn.Sequential(nn.ReLU())
    x = torch.randn(1, 4)
    assert count_flops(model, x) is None
    compare(model, model, x, runs=2, warmup=1)
    assert "MFLOPs" not in capsys.readouterr().out
