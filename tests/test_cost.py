import random

import pytest
import torch

from examples.test_models import TEST_MODELS
from shrinker import compress
from shrinker.benchmark import count_flops, count_parameters
from shrinker.cost import PruningCost, kept_channels
from shrinker.utils import keep_indices

PRUNABLE = [name for name in TEST_MODELS if PruningCost(TEST_MODELS[name][0](), metric="params").groups]


@pytest.mark.parametrize("name", PRUNABLE)
def test_cost_matches_the_pruned_model(name):
    cls, shape = TEST_MODELS[name]
    model = cls().eval()
    x = torch.randn(*shape)
    flops, params = PruningCost(model, x, "flops"), PruningCost(model, metric="params")
    assert flops.full == count_flops(model, x)
    assert params.full == count_parameters(model)

    rng = random.Random(0)
    for _ in range(5):
        ratios = {g: rng.choice([0.0, 0.1, 0.25, 0.5, 0.77, 0.9]) for g in flops.groups}
        pruned = compress(model, "l1_pruning", ratio=ratios)
        assert flops(ratios) == count_flops(pruned, x)
        assert params(ratios) == count_parameters(pruned)


def test_residual_groups_are_one_entry():
    cls, shape = TEST_MODELS["SmallResNet"]
    cost = PruningCost(cls(), torch.randn(*shape))
    assert "layer1.conv1" in cost.groups
    assert "layer1.conv2" not in cost.groups  # pruned together with stem.0


@pytest.mark.parametrize("n", [1, 3, 7, 64])
@pytest.mark.parametrize("ratio", [0.0, 0.05, 0.3, 0.5, 0.95])
def test_kept_channels_matches_keep_indices(n, ratio):
    assert kept_channels(n, ratio) == keep_indices(torch.rand(n), ratio).numel()


def test_bad_arguments():
    cls, _ = TEST_MODELS["SmallMLP"]
    with pytest.raises(ValueError, match="metric"):
        PruningCost(cls(), metric="latency")
    with pytest.raises(ValueError, match="example_input"):
        PruningCost(cls(), metric="flops")
