"""Tests specific to the knowledge_distillation method."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from examples.test_models import SmallMLP, SmallVGG, TEST_MODELS
from shrinker import CompressionError, compress
from shrinker.benchmark import count_flops, count_nonzero_parameters, count_parameters, model_size_mb
from shrinker.methods.knowledge_distillation import _soft_target_loss, _zero_smallest_weights, apply
from tests.helpers import make_loader


def _make_loader_for(model_cls, input_shape, n=32, batch_size=8):
    """Build a (x, y) DataLoader compatible with a given model class."""
    x = torch.randn(n, *input_shape)
    model = model_cls().eval()
    with torch.no_grad():
        num_classes = model(x[:1]).shape[1]
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _kl_to_teacher(student, teacher, x, temperature=4.0):
    with torch.no_grad():
        return _soft_target_loss(student.eval()(x), teacher.eval()(x), temperature).item()


class _NotTraceable(nn.Module):
    """forward() branches on the data, so torch.fx cannot trace it and nothing can be pruned."""

    def __init__(self, hidden_layers=1):
        super().__init__()
        self.hidden = nn.ModuleList(nn.Linear(4, 4) for _ in range(hidden_layers))
        self.out = nn.Linear(4, 3)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        if x.sum() > 0:
            x = x * 2
        return self.out(x)


# --- the student is really a smaller network -------------------------------------------------

@pytest.mark.parametrize("model_name", list(TEST_MODELS))
def test_student_is_smaller_than_teacher(model_name):
    cls, shape = TEST_MODELS[model_name]
    teacher = cls().eval()
    x = torch.randn(*shape)
    with torch.no_grad():
        teacher_out = teacher(x)
    loader = make_loader(shape[1:], num_classes=teacher_out.shape[1])

    student = compress(teacher, "knowledge_distillation", example_input=x, loader=loader, epochs=1)

    with torch.no_grad():
        assert student.eval()(x).shape == teacher_out.shape
    assert count_parameters(student) < count_parameters(teacher)
    assert model_size_mb(student) < model_size_mb(teacher)
    assert count_flops(student, x) < count_flops(teacher, x)


def test_default_student_removes_half_of_each_prunable_layer():
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    student = compress(SmallMLP(), "knowledge_distillation", loader=loader, epochs=1)
    linears = [m for m in student.net if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(784, 128), (128, 64), (64, 10)]


def test_ratio_option_sets_the_student_size():
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    student = compress(SmallMLP(), "knowledge_distillation", loader=loader, epochs=1, ratio=0.25)
    linears = [m for m in student.net if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(784, 192), (192, 96), (96, 10)]


@pytest.mark.parametrize("ratio", [-0.1, 0.0, 1.0, 1.5])
def test_invalid_ratio(ratio):
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    with pytest.raises(ValueError, match="ratio"):
        compress(SmallMLP(), "knowledge_distillation", loader=loader, ratio=ratio)


# --- the student really learns from the teacher ----------------------------------------------

def test_soft_targets_move_student_toward_teacher():
    """With alpha=1 (only soft loss) the student starts different from the teacher and gets closer."""
    teacher = SmallMLP().eval()
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    x = torch.cat([xb for xb, _ in loader])
    options = dict(loader=loader, alpha=1.0, sparsity=0.0)

    untrained = compress(teacher, "knowledge_distillation", epochs=0, **options)
    trained = compress(teacher, "knowledge_distillation", epochs=5, **options)

    before = _kl_to_teacher(untrained, teacher, x)
    after = _kl_to_teacher(trained, teacher, x)
    assert before > 1e-4, "the student must start different from the teacher (that was the bug)"
    assert after < 0.5 * before, "training should pull the student toward the teacher"


# --- passing your own student ----------------------------------------------------------------

def test_custom_student_is_trained_and_returned():
    teacher = SmallMLP()
    my_student = SmallMLP(hidden=64)
    before = {k: v.clone() for k, v in my_student.state_dict().items()}
    loader = _make_loader_for(SmallMLP, (1, 28, 28))

    result = compress(teacher, "knowledge_distillation", loader=loader, student=my_student, epochs=1)

    assert count_parameters(result) == count_parameters(my_student)  # same architecture as the student
    assert result is not my_student  # we train a copy ...
    for k, v in my_student.state_dict().items():  # ... so the caller's student is untouched
        assert torch.equal(before[k], v), f"the caller's student changed: {k}"
    assert any(not torch.equal(before[k], v) for k, v in result.state_dict().items()), "it was not trained"


def test_custom_student_with_wrong_number_of_outputs():
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    with pytest.raises(ValueError, match="output shape"):
        compress(SmallMLP(), "knowledge_distillation", loader=loader, student=SmallMLP(num_classes=5), epochs=1)


@pytest.mark.parametrize("hidden", [256, 512])  # as big as the SmallMLP teacher, and bigger
def test_custom_student_must_be_smaller(hidden):
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    with pytest.raises(ValueError, match="student must be smaller"):
        compress(SmallMLP(), "knowledge_distillation", loader=loader, student=SmallMLP(hidden=hidden), epochs=1)


def test_student_must_be_a_module():
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    with pytest.raises(TypeError, match="student must be an nn.Module"):
        compress(SmallMLP(), "knowledge_distillation", loader=loader, student="small_model.pt")


# --- when no smaller student can be built ------------------------------------------------------

def test_nothing_prunable_gives_clear_error():
    model = nn.Sequential(nn.Linear(4, 3))  # its only layer is the output layer
    loader = make_loader((4,), num_classes=3)
    with pytest.raises(CompressionError, match="knowledge_distillation failed.*no layer.*pruned safely.*model output"):
        compress(model, "knowledge_distillation", loader=loader, epochs=1)


def test_untraceable_model_gives_clear_error():
    loader = make_loader((4,), num_classes=3)
    with pytest.raises(CompressionError, match="could not be traced"):
        compress(_NotTraceable(), "knowledge_distillation", loader=loader, epochs=1)


def test_error_message_stays_short_for_big_models():
    loader = make_loader((4,), num_classes=3)
    with pytest.raises(CompressionError, match="and 5 more") as error:  # 10 layers, only 5 are listed
        compress(_NotTraceable(hidden_layers=9), "knowledge_distillation", loader=loader, epochs=1)
    assert "hidden.9" not in str(error.value) and "hidden.8" not in str(error.value)


def test_untraceable_model_works_with_own_student():
    loader = make_loader((4,), num_classes=3)
    small = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 3))
    result = compress(_NotTraceable(), "knowledge_distillation", loader=loader, student=small, epochs=1)
    assert count_parameters(result) < count_parameters(_NotTraceable())


def test_pruning_that_removes_nothing_gives_clear_error():
    model = nn.Sequential(nn.Linear(4, 1), nn.ReLU(), nn.Linear(1, 3))  # the hidden layer has 1 neuron
    loader = make_loader((4,), num_classes=3)
    with pytest.raises(CompressionError, match="did not make the student smaller"):
        compress(model, "knowledge_distillation", loader=loader, epochs=1)


# --- sparsity (optional extra step) ------------------------------------------------------------

def test_zero_smallest_weights_zeroes_the_requested_fraction():
    model = SmallMLP()
    _zero_smallest_weights(model, 0.3)
    weights = torch.cat([p.detach().flatten() for p in model.parameters() if p.dim() >= 2])
    assert 0.29 <= (weights == 0).float().mean().item() <= 0.31


def test_sparsity_zeros_weights_of_the_student():
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    small = compress(SmallMLP(), "knowledge_distillation", loader=loader, sparsity=0.3, epochs=1)
    assert count_nonzero_parameters(small) < count_parameters(small), "sparsity should zero out some weights"


def test_zero_sparsity_zeroes_nothing():
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    small = compress(SmallMLP(), "knowledge_distillation", loader=loader, sparsity=0.0, epochs=1)
    assert count_nonzero_parameters(small) == count_parameters(small)


def test_no_zeroed_weights_by_default():
    loader = _make_loader_for(SmallMLP, (1, 28, 28))
    small = compress(SmallMLP(), "knowledge_distillation", loader=loader, epochs=1)
    assert count_nonzero_parameters(small) == count_parameters(small)


# --- nothing else is affected ------------------------------------------------------------------

def test_works_on_cnn():
    """Distillation should work on CNN models too."""
    model = SmallVGG()
    loader = _make_loader_for(SmallVGG, (3, 32, 32))
    x = torch.randn(2, 3, 32, 32)

    small = compress(
        model,
        "knowledge_distillation",
        example_input=x,
        loader=loader,
        epochs=1,
        sparsity=0.1,
    )

    with torch.no_grad():
        out = small.eval()(x)
    assert out.shape == model.eval()(x).shape
    assert torch.isfinite(out).all()


def test_original_model_unchanged():
    """The original model should never be modified."""
    model = SmallMLP()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = _make_loader_for(SmallMLP, (1, 28, 28))

    compress(model, "knowledge_distillation", loader=loader, epochs=1)

    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"Original model changed: {k}"


def test_apply_leaves_the_teacher_exactly_as_it_found_it():
    """Even when apply() is called directly (without compress() making a copy first)."""
    teacher = SmallMLP().train()  # train mode on purpose: it must still be in train mode afterwards
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    flags = [p.requires_grad for p in teacher.parameters()]

    student = apply(teacher, _make_loader_for(SmallMLP, (1, 28, 28)), epochs=1)

    assert student is not teacher
    assert teacher.training
    assert [p.requires_grad for p in teacher.parameters()] == flags
    for k, v in teacher.state_dict().items():
        assert torch.equal(before[k], v), f"teacher changed: {k}"

def test_custom_student_with_wrong_output_shape_is_rejected_without_training():
    loader = _make_loader_for(SmallMLP, (1, 28, 28))

    with pytest.raises(ValueError, match="output shape"):
        compress(
            SmallMLP(),
            "knowledge_distillation",
            loader=loader,
            student=SmallMLP(num_classes=5),
            epochs=0,
        )