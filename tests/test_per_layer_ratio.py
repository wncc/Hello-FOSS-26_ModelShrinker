import pytest
import torch
from torch import nn

from examples.test_models import SmallVGG
from shrinker import compress
from shrinker.utils import prunable_layers

METHODS = ["l1_pruning", "fpgm", "gradient_based"]


def _kwargs(method):
    """gradient_based needs data to score filters; the other methods do not."""
    if method == "gradient_based":
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        return {"loader": [(x, y)]}
    return {}


def _channels(model):
    return {n: m.out_channels for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}


@pytest.mark.parametrize("method", METHODS)
def test_dict_prunes_only_named_layers(method):
    before = _channels(SmallVGG())
    small = compress(SmallVGG(), method, ratio={"features.0": 0.5}, **_kwargs(method))
    after = _channels(small)
    assert after["features.0"] == before["features.0"] // 2
    for name in before:
        if name != "features.0":
            assert after[name] == before[name]


@pytest.mark.parametrize("method", METHODS)
def test_each_layer_gets_its_own_ratio(method):
    before = _channels(SmallVGG())
    ratios = {"features.0": 0.5, "features.7": 0.25}
    after = _channels(compress(SmallVGG(), method, ratio=ratios, **_kwargs(method)))
    assert after["features.0"] == before["features.0"] // 2
    assert after["features.7"] == before["features.7"] * 3 // 4


@pytest.mark.parametrize("method", METHODS)
def test_float_matches_dict_with_same_value_everywhere(method):
    names = [name for name, _ in prunable_layers(SmallVGG())]
    from_float = _channels(compress(SmallVGG(), method, ratio=0.5, **_kwargs(method)))
    from_dict = _channels(compress(SmallVGG(), method, ratio={n: 0.5 for n in names}, **_kwargs(method)))
    assert from_float == from_dict


@pytest.mark.parametrize("method", METHODS)
def test_unknown_layer_name_raises(method):
    with pytest.raises(ValueError, match="oops"):
        compress(SmallVGG(), method, ratio={"oops": 0.5}, **_kwargs(method))


@pytest.mark.parametrize("method", METHODS)
def test_unprunable_layer_name_raises(method):
    # classifier.4 produces the model output, so it cannot be pruned
    with pytest.raises(ValueError, match="can't be pruned"):
        compress(SmallVGG(), method, ratio={"classifier.4": 0.5}, **_kwargs(method))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_out_of_range_ratio_in_dict_raises(method, bad):
    with pytest.raises(ValueError, match="ratio"):
        compress(SmallVGG(), method, ratio={"features.0": bad}, **_kwargs(method))


@pytest.mark.parametrize("method", METHODS)
def test_empty_dict_changes_nothing(method):
    before = _channels(SmallVGG())
    after = _channels(compress(SmallVGG(), method, ratio={}, **_kwargs(method)))
    assert after == before