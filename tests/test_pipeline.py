import copy

import pytest
import torch
from torch import nn

from examples.test_models import SmallMLP, SmallVGG
from shrinker import CompressionError, Pipeline, compress, register
from shrinker.registry import REGISTRY, UnknownMethodError
from shrinker.utils import replace_module, slice_linear_out
from tests.helpers import make_loader

X_VGG = torch.randn(2, 3, 32, 32)


def _first_conv_channels(model):
    return next(m for m in model.modules() if isinstance(m, nn.Conv2d)).out_channels


def _is_quantized(model):
    return any("quantized" in type(m).__module__ for m in model.modules())


# ---- running ---------------------------------------------------------------------------------


def test_steps_run_in_the_given_order(restore_registry):
    calls = []

    @register("record_a")
    def apply(model):
        calls.append("a")
        return model

    @register("record_b")
    def apply_b(model):
        calls.append("b")
        return model

    Pipeline(["record_b", "record_a", "record_b"])(SmallMLP())
    assert calls == ["b", "a", "b"]


def test_each_step_gets_its_own_options():
    small = Pipeline([("l1_pruning", {"ratio": 0.5}), ("l1_pruning", {"ratio": 0.5})])(SmallVGG())
    assert _first_conv_channels(small) == 8  # 32 -> 16 -> 8


def test_same_result_as_calling_compress_twice():
    model = SmallVGG().eval()
    by_hand = compress(compress(model, "l1_pruning", ratio=0.3), "fpgm", ratio=0.2).eval()
    piped = Pipeline([("l1_pruning", {"ratio": 0.3}), ("fpgm", {"ratio": 0.2})])(model).eval()
    with torch.no_grad():
        assert torch.allclose(piped(X_VGG), by_hand(X_VGG))


def test_single_step_matches_compress():
    model = SmallVGG().eval()
    with torch.no_grad():
        expected = compress(model, "l1_pruning", ratio=0.4).eval()(X_VGG)
        got = Pipeline([("l1_pruning", {"ratio": 0.4})])(model).eval()(X_VGG)
    assert torch.allclose(got, expected)


def test_original_model_is_never_changed():
    model = SmallVGG().eval()
    before = copy.deepcopy(model.state_dict())
    structure = str(model)
    small = Pipeline([("l1_pruning", {"ratio": 0.5}), "fpgm"])(model, example_input=X_VGG)
    assert small is not model
    assert str(model) == structure
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_prune_then_quantize():
    model = SmallVGG().eval()
    small = Pipeline([("l1_pruning", {"ratio": 0.3}), "dfq_quantization"])(model, example_input=X_VGG)
    assert _is_quantized(small)
    with torch.no_grad():
        assert small(X_VGG).shape == model(X_VGG).shape


def test_step_that_needs_a_loader():
    loader = make_loader((3, 32, 32))
    small = Pipeline([("gradient_based", {"loader": loader, "ratio": 0.3}), "l1_pruning"])(SmallVGG())
    assert _first_conv_channels(small) < 32


def test_pipeline_can_be_reused_on_several_models():
    pipe = Pipeline([("l1_pruning", {"ratio": 0.5})])
    assert _first_conv_channels(pipe(SmallVGG())) == 16
    assert _first_conv_channels(pipe(SmallVGG())) == 16


def test_training_mode_preserved():
    pipe = Pipeline(["l1_pruning", "fpgm"])
    assert pipe(SmallVGG().train()).training
    assert not pipe(SmallVGG().eval()).training


def test_output_shape_checked_after_each_step(restore_registry):
    @register("drop_classes")
    def apply(model):
        replace_module(model, "classifier.4", slice_linear_out(model.classifier[4], [0, 1]))
        return model

    with pytest.raises(CompressionError, match=r"step 2/2 \('drop_classes'\).*changed the output shape"):
        Pipeline(["l1_pruning", "drop_classes"])(SmallVGG(), example_input=X_VGG)


def test_errors_inside_a_step_name_the_step():
    pipe = Pipeline(["fpgm", ("l1_pruning", {"ratio": 1.5})])
    with pytest.raises(ValueError, match=r"step 2/2 \('l1_pruning'\).*ratio"):
        pipe(SmallMLP())


def test_quantized_output_mid_pipeline_is_caught(restore_registry):
    # A quantization method registered with the wrong kind slips past the build-time check.
    dfq = REGISTRY["dfq_quantization"].fn

    @register("sneaky_quant", kind="other")
    def apply(model, example_input=None):
        return dfq(model, example_input=example_input)

    with pytest.raises(CompressionError, match=r"Step 1/2 \('sneaky_quant'\) returned a quantized model"):
        Pipeline(["sneaky_quant", "l1_pruning"])(SmallVGG().eval(), example_input=X_VGG)


# ---- building --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "steps",
    [
        ["dfq_quantization", "l1_pruning"],
        [("l1_pruning", {"ratio": 0.3}), "dfq_quantization", "fpgm"],
        ["dfq_quantization", "dfq_quantization"],
    ],
)
def test_nothing_may_follow_quantization(steps):
    with pytest.raises(ValueError, match="after step .* a quantization method"):
        Pipeline(steps)


def test_unknown_method():
    with pytest.raises(UnknownMethodError, match="Step 2: Unknown method 'nope'"):
        Pipeline(["l1_pruning", "nope"])


def test_unknown_option():
    with pytest.raises(TypeError, match="unknown option.*ratoi.*accepts: ratio"):
        Pipeline([("l1_pruning", {"ratoi": 0.3})])


def test_missing_required_option():
    with pytest.raises(TypeError, match="needs option.*loader"):
        Pipeline(["gradient_based"])


def test_example_input_is_not_a_step_option():
    with pytest.raises(TypeError, match="pass example_input when calling the pipeline"):
        Pipeline([("dfq_quantization", {"example_input": X_VGG})])


def test_methods_with_var_kwargs_accept_any_option(restore_registry):
    seen = {}

    @register("flexible")
    def apply(model, **kwargs):
        seen.update(kwargs)
        return model

    Pipeline([("flexible", {"anything": 1})])(SmallMLP())
    assert seen == {"anything": 1}


@pytest.mark.parametrize(
    "steps, error",
    [
        ("l1_pruning", TypeError),  # a bare string, not a list
        ([], ValueError),
        ([42], TypeError),
        ([("l1_pruning",)], TypeError),
        ([("l1_pruning", 0.3)], TypeError),  # options must be a dict
        (None, TypeError),
    ],
)
def test_bad_steps(steps, error):
    with pytest.raises(error):
        Pipeline(steps)


def test_rejects_non_module():
    with pytest.raises(TypeError, match="nn.Module"):
        Pipeline(["l1_pruning"])("not a model")


def test_options_are_copied():
    options = {"ratio": 0.5}
    pipe = Pipeline([("l1_pruning", options)])
    options["ratio"] = 0.9  # changing the caller's dict afterwards...
    pipe.steps[0][1]["ratio"] = 0.9  # ...or the returned copy...
    assert pipe.steps == (("l1_pruning", {"ratio": 0.5}),)  # ...doesn't change the pipeline
    assert _first_conv_channels(pipe(SmallVGG())) == 16


def test_repr_and_len():
    pipe = Pipeline([("l1_pruning", {"ratio": 0.3}), "dfq_quantization"])
    assert repr(pipe) == "Pipeline([('l1_pruning', {'ratio': 0.3}), 'dfq_quantization'])"
    assert len(pipe) == 2
    assert [name for name, _ in pipe] == ["l1_pruning", "dfq_quantization"]


def test_compress_points_to_pipeline_when_given_a_list():
    with pytest.raises(TypeError, match="use shrinker.Pipeline"):
        compress(SmallMLP(), ["l1_pruning", "fpgm"])
