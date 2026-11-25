import copy
import inspect
import io
import itertools
import warnings

import pytest
import torch

from examples.test_models import TEST_MODELS
from shrinker import compress
from shrinker.benchmark import count_nonzero_parameters, count_parameters, model_size_mb
from shrinker.registry import KINDS, LOAD_ERRORS, REGISTRY
from tests.helpers import make_loader

METHODS = sorted(REGISTRY)


def run_method(name, model, input_shape):
    x = torch.randn(*input_shape)
    kwargs = {}
    if "loader" in inspect.signature(REGISTRY[name].fn).parameters:
        kwargs["loader"] = make_loader(input_shape[1:], num_classes=_num_outputs(model, x))
    return compress(model, name, example_input=x, **kwargs), x


def _num_outputs(model, x):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        n = model(x).shape[1]
    model.train(was_training)
    return n


def _is_quantized(model):
    return any("quantized" in type(m).__module__ for m in model.modules())


def test_all_method_files_load():
    assert not LOAD_ERRORS, f"These method files failed to import: {LOAD_ERRORS}"


def test_at_least_one_method_registered():
    assert METHODS


@pytest.mark.parametrize("name", METHODS)
def test_signature(name):
    params = list(inspect.signature(REGISTRY[name].fn).parameters.values())
    assert params and params[0].name == "model", f"{name}: first argument must be `model`"
    for p in params[1:]:
        assert p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_KEYWORD), (
            f"{name}: `{p.name}` must be a keyword argument"
        )
        if p.name != "loader" and p.kind is not p.VAR_KEYWORD:
            assert p.default is not p.empty, f"{name}: option `{p.name}` needs a default value"


@pytest.mark.parametrize("model_name", list(TEST_MODELS))
@pytest.mark.parametrize("name", METHODS)
def test_method_contract(name, model_name):
    cls, shape = TEST_MODELS[model_name]
    original = cls().eval()
    before = copy.deepcopy(original.state_dict())
    structure = str(original)

    small, x = run_method(name, original, shape)

    assert small is not original
    with torch.no_grad():
        out_small, out_orig = small.eval()(x), original(x)
    assert out_small.shape == out_orig.shape
    assert torch.isfinite(out_small).all()

    assert str(original) == structure
    after = original.state_dict()
    assert before.keys() == after.keys()
    for k in before:
        assert torch.equal(before[k], after[k]), f"original model changed: {k}"

    # DFQ's bias absorption/correction adds a bias to a few bias-free convs, hence the small tolerance.
    assert count_parameters(small) <= count_parameters(original) * 1.01
    assert model_size_mb(small) <= model_size_mb(original) * 1.01

    # Some PyTorch model types (e.g. FX-quantized) can't be pickled; that's not the method's fault.
    try:
        buf = io.BytesIO()
        torch.save(small, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False).eval()
    except Exception as e:
        warnings.warn(f"{name} on {model_name}: result can't be saved with torch.save ({type(e).__name__}: {e})")
        return
    with torch.no_grad():
        assert torch.allclose(loaded(x), out_small)


@pytest.mark.parametrize("name", METHODS)
def test_shrinks_at_least_one_model(name):
    shrunk = []
    for model_name, (cls, shape) in TEST_MODELS.items():
        original = cls()
        try:
            small, _ = run_method(name, original, shape)
        except Exception:
            continue  # crashes are reported by test_method_contract
        if (
            count_parameters(small) < count_parameters(original)
            or model_size_mb(small) < model_size_mb(original)
            or count_nonzero_parameters(small) < count_nonzero_parameters(original)
        ):
            shrunk.append(model_name)
    assert shrunk, f"{name} with default options didn't make any test model smaller"


def _kind_order(name):
    return KINDS.index(REGISTRY[name].kind)


PAIRS = [(a, b) for a, b in itertools.product(METHODS, repeat=2) if _kind_order(a) <= _kind_order(b)]


@pytest.mark.parametrize("model_name", ["SmallMLP", "SmallVGG"])
@pytest.mark.parametrize("first,second", PAIRS)
def test_methods_combine(first, second, model_name):
    cls, shape = TEST_MODELS[model_name]
    once, x = run_method(first, cls(), shape)
    if _is_quantized(once):
        pytest.skip(f"{first} returns a quantized model; nothing is applied after quantization")
    twice, _ = run_method(second, once, shape)
    with torch.no_grad():
        assert twice.eval()(x).shape == cls().eval()(x).shape
