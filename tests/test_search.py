import copy

import pytest
import torch

from examples.test_models import TEST_MODELS
from shrinker import compress, search_ratios
from shrinker import search as search_module
from shrinker.benchmark import count_flops, count_parameters
from shrinker.cost import PruningCost
from shrinker.dependency import prunable_layers
from tests.helpers import make_loader


def _setup(name="SmallVGG"):
    cls, shape = TEST_MODELS[name]
    model = cls().eval()
    return model, make_loader(shape[1:]), torch.randn(*shape)


@pytest.mark.parametrize("target", [0.3, 0.5, 0.7])
def test_flops_budget_is_met(target):
    model, loader, x = _setup()
    ratios = search_ratios(model, loader, target=target, episodes=4)
    removed = 1 - count_flops(compress(model, "l1_pruning", ratio=ratios), x) / count_flops(model, x)
    assert target <= removed < target + 0.05


def test_params_budget_is_met():
    model, loader, _ = _setup()
    ratios = search_ratios(model, loader, target=0.5, metric="params", episodes=4)
    removed = 1 - count_parameters(compress(model, "l1_pruning", ratio=ratios)) / count_parameters(model)
    assert 0.5 <= removed < 0.55


@pytest.mark.parametrize("name", ["SmallVGG", "SmallResNet", "SmallFunctionalCNN", "SmallTabularMLP"])
def test_result_works_with_compress(name):
    model, loader, x = _setup(name)
    ratios = search_ratios(model, loader, target=0.4, episodes=4)
    assert list(ratios) == [n for n, _ in prunable_layers(model)]
    assert all(0.0 <= r <= 0.8 for r in ratios.values())
    compress(model, "l1_pruning", example_input=x, ratio=ratios)


def test_same_seed_same_plan_and_model_untouched():
    model, loader, _ = _setup()
    model.train()
    before = copy.deepcopy(model.state_dict())
    a = search_ratios(model, loader, target=0.5, episodes=8, seed=3)
    b = search_ratios(model, loader, target=0.5, episodes=8, seed=3)
    assert a == b
    assert model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_report_never_worse_than_uniform():
    model, loader, _ = _setup()
    ratios, report = search_ratios(model, loader, target=0.5, episodes=8, return_report=True)
    assert report["chosen"] in ("uniform", "greedy", "reinforce")
    assert report["accuracy"] >= report["uniform_accuracy"]
    assert report["accuracy"] >= report["greedy_accuracy"]
    assert len(report["rewards"]) == 8
    assert report["removed"] >= 0.5


def test_uniform_and_greedy_agents_skip_rl(monkeypatch):
    model, loader, x = _setup()

    def no_rl(*args, **kwargs):
        raise AssertionError("the RL agent should not run")

    monkeypatch.setattr(search_module, "_Walker", no_rl)
    uniform = search_ratios(model, loader, target=0.5, agent="uniform")
    assert max(uniform.values()) - min(uniform.values()) <= 0.01
    greedy = search_ratios(model, loader, target=0.5, agent="greedy", episodes=0)
    cost = PruningCost(model, x)
    assert 1 - cost.fraction(greedy) >= 0.5


def test_no_episodes_still_returns_a_plan():
    model, loader, x = _setup()
    ratios = search_ratios(model, loader, target=0.5, episodes=0)
    assert 1 - PruningCost(model, x).fraction(ratios) >= 0.5


def test_unreachable_target_says_what_is_possible():
    model, loader, _ = _setup("SmallMLP")
    with pytest.raises(ValueError, match="removes only"):
        search_ratios(model, loader, target=0.95, max_ratio=0.5)


def test_other_pruning_methods():
    model, loader, x = _setup()
    ratios = search_ratios(model, loader, target=0.5, episodes=4, method="fpgm")
    removed = 1 - count_flops(compress(model, "fpgm", ratio=ratios), x) / count_flops(model, x)
    assert removed >= 0.5


def test_bad_arguments():
    model, loader, _ = _setup()
    with pytest.raises(ValueError, match="agent"):
        search_ratios(model, loader, agent="ddpg")
    with pytest.raises(ValueError, match="target"):
        search_ratios(model, loader, target=1.0)
    with pytest.raises(ValueError, match="ratio"):
        search_ratios(model, loader, method="ptq_quantization")
    with pytest.raises(ValueError, match="inputs, labels"):
        search_ratios(model, [torch.randn(2, 3, 32, 32)])
    with pytest.raises(ValueError, match="empty"):
        search_ratios(model, [])
    lstm, lstm_loader, _ = _setup("SmallLSTM")
    with pytest.raises(ValueError, match="can be pruned"):
        search_ratios(lstm, lstm_loader)
