import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import shrinker
from examples.test_models import TEST_MODELS
from shrinker.dependency import analyze
from shrinker.summary import format_summary, summary


class Unused(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 2)
        self.spare = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Untraceable(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        if x.sum() > 0:
            x = self.fc1(x)
        return self.fc2(x)


def by_name(rows):
    return {r["name"]: r for r in rows}


def test_summary_is_exported_from_package():
    assert shrinker.summary is summary
    assert "summary" in shrinker.__all__


@pytest.mark.parametrize("model_name", list(TEST_MODELS))
def test_one_row_per_conv_and_linear_layer_in_module_order(model_name):
    model = TEST_MODELS[model_name][0]()
    expected = [n for n, m in model.named_modules() if type(m) in (nn.Conv2d, nn.Linear)]
    assert [r["name"] for r in summary(model)] == expected


def test_row_reports_type_weight_shape_and_parameter_count():
    rows = by_name(summary(Unused()))
    assert rows["fc1"]["type"] == "Linear"
    assert rows["fc1"]["shape"] == (6, 4)
    assert rows["fc1"]["params"] == 6 * 4 + 6

    conv = by_name(summary(TEST_MODELS["SmallVGG"][0]()))["features.0"]
    assert conv["type"] == "Conv2d"
    assert conv["shape"] == (32, 3, 3, 3)
    assert conv["params"] == 32 * 3 * 3 * 3  # bias=False


@pytest.mark.parametrize("model_name", list(TEST_MODELS))
def test_prunability_and_reasons_match_analyze(model_name):
    model = TEST_MODELS[model_name][0]()
    prunable, skipped = analyze(model)
    for row in summary(model):
        assert row["prunable"] == (row["name"] in prunable)
        assert row["reason"] == skipped.get(row["name"])


def test_prunable_layer_has_no_reason_and_output_layer_explains_itself():
    rows = by_name(summary(Unused()))
    assert rows["fc1"]["prunable"] is True
    assert rows["fc1"]["reason"] is None
    assert rows["fc2"]["prunable"] is False
    assert "model output" in rows["fc2"]["reason"]


def test_layer_not_used_in_forward_is_shown_that_way():
    row = by_name(summary(Unused()))["spare"]
    assert row["prunable"] is False
    assert row["reason"] == "not used in forward()"


def test_untraceable_model_marks_every_layer_with_the_trace_error():
    rows = summary(Untraceable())
    assert [r["name"] for r in rows] == ["fc1", "fc2"]
    assert all(not r["prunable"] and "could not be traced" in r["reason"] for r in rows)


def test_output_shape_is_none_without_example_input():
    assert all(r["output_shape"] is None for r in summary(Unused()))


def test_output_shape_is_recorded_from_example_input():
    cls, shape = TEST_MODELS["SmallVGG"]
    rows = by_name(summary(cls(), example_input=torch.randn(*shape)))
    assert rows["features.0"]["output_shape"] == (shape[0], 32, shape[2], shape[3])
    assert rows["classifier.4"]["output_shape"] == (shape[0], 10)


def test_layer_not_used_in_forward_has_no_output_shape():
    rows = by_name(summary(Unused(), example_input=torch.randn(3, 4)))
    assert rows["fc1"]["output_shape"] == (3, 6)
    assert rows["spare"]["output_shape"] is None


def test_summary_leaves_the_model_untouched():
    model = TEST_MODELS["SmallVGG"][0]()  # has BatchNorm and Dropout
    model.train()
    before = copy.deepcopy(model.state_dict())
    summary(model, example_input=torch.randn(*TEST_MODELS["SmallVGG"][1]))
    assert model.training
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(len(m._forward_hooks) == 0 for m in model.modules())


def test_prints_a_table_with_every_layer_reason_and_totals(capsys):
    model = Unused()
    rows = summary(model)
    out = capsys.readouterr().out
    assert out.rstrip("\n") == format_summary(rows)
    for name in ("fc1", "fc2", "spare"):
        assert name in out
    assert "not used in forward()" in out
    total = sum(r["params"] for r in rows)
    assert f"{total:,}" in out
    assert "1 of 3 layers prunable" in out


def test_table_has_an_output_column_only_when_shapes_were_recorded():
    model = Unused()
    assert "output" not in format_summary(summary(model)).splitlines()[0]
    with_shapes = format_summary(summary(model, example_input=torch.randn(3, 4)))
    assert "output" in with_shapes.splitlines()[0]
    assert "(3, 6)" in with_shapes


def test_model_without_conv_or_linear_layers_gives_no_rows(capsys):
    assert summary(nn.Sequential(nn.ReLU())) == []
    assert "no Conv2d/Linear layers" in capsys.readouterr().out
