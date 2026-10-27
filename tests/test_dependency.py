import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from examples.test_models import TEST_MODELS, BasicBlock
from shrinker import compress
from shrinker.dependency import analyze
from shrinker.utils import get_module, prunable_layers, prune_channels, skipped_layers
from tests.helpers import make_loader




class FunctionalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)
        self.fc1 = nn.Linear(12 * 4 * 4, 20)
        self.bn_fc = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        return self.fc2(x)


class FanOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.a = nn.Conv2d(8, 6, 3, padding=1)
        self.b = nn.Conv2d(8, 6, 1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, 4))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.head(self.a(x) + self.b(x))


class HardcodedView(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, 3)

    def forward(self, x):
        return self.fc(self.conv(x).view(-1, 4 * 8 * 8))


class Reused(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.out = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.out(self.conv(self.conv(x)))


class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 4, 3, padding=1)
        self.b = nn.Conv2d(3, 4, 3, padding=1)
        self.out = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.out(torch.cat([self.a(x), self.b(x)], dim=1))


class Depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.out = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.out(self.dw(self.conv(x)))


class Untraceable(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        if x.sum() > 0:
            x = self.fc1(x)
        return self.fc2(x)


class Unused(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 2)
        self.spare = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ResidualStack(nn.Module):
    """Two identity-shortcut blocks in a row: stem, b1.conv2 and b2.conv2 all share channels."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.b1 = BasicBlock(8, 8)
        self.b2 = BasicBlock(8, 8)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 5))

    def forward(self, x):
        return self.head(self.b2(self.b1(self.stem(x))))


class ResidualMLP(nn.Module):
    """Linear residual block with BatchNorm1d: fc_in and fc2 are added together."""

    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(10, 16)
        self.fc1 = nn.Linear(16, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        h = self.fc2(torch.relu(self.bn(self.fc1(x))))
        return self.out(torch.relu(x + h))


class ResidualOnInput(nn.Module):
    """A conv output is added to the model input, whose channels can't be pruned."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.out = nn.Conv2d(3, 2, 1)

    def forward(self, x):
        return self.out(self.conv(x) + x)


class ResidualDepthwise(nn.Module):
    """A conv output is added to a depthwise conv output, which can't be pruned."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.out = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        h = self.conv(x)
        return self.out(self.dw(h) + h)


class ResidualToOutput(nn.Module):
    """The added result is the model output."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 4, 3, padding=1)
        self.b = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return self.a(x) + self.b(x)


class ResidualWithCat(nn.Module):
    """A conv output is added to a concatenation, which pruning can't trace back."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.a = nn.Conv2d(3, 4, 3, padding=1)
        self.b = nn.Conv2d(3, 4, 3, padding=1)
        self.out = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.out(self.conv(x) + torch.cat([self.a(x), self.b(x)], dim=1))

class ResidualAfterFlatten(nn.Module):
    """A flattened conv output is added to a Linear output."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(3 * 4 * 4, 4 * 4 * 4)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        return self.out(torch.flatten(self.conv(x), 1) + self.fc(torch.flatten(x, 1)))


class ResidualWithConstant(nn.Module):
    """A conv output is added to a Python constant."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.out = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.out(self.conv(x) + 1.0)


MODELS = {
    **{name: (cls, shape) for name, (cls, shape) in TEST_MODELS.items()},
    "FunctionalCNN": (FunctionalCNN, (3, 3, 16, 16)),
    "FanOut": (FanOut, (2, 3, 8, 8)),
    "ResidualStack": (ResidualStack, (2, 3, 8, 8)),
    "ResidualMLP": (ResidualMLP, (4, 10)),
}


def randomize_bn(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            with torch.no_grad():
                m.running_mean.uniform_(-1, 1)
                m.running_var.uniform_(0.5, 2)
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.5, 0.5)
    return model


def disconnect(model, group, dropped):
    for cname, spatial in group.consumers:
        consumer = get_module(model, cname)
        with torch.no_grad():
            if isinstance(consumer, nn.Conv2d):
                consumer.weight[:, dropped] = 0
            else:
                idx = (dropped[:, None] * spatial + torch.arange(spatial)[None, :]).flatten()
                consumer.weight[:, idx] = 0


def num_out(layer):
    return layer.out_channels if isinstance(layer, nn.Conv2d) else layer.out_features




@pytest.mark.parametrize("model_name", list(MODELS))
def test_each_layer_pruned_alone_matches_reference(model_name):
    cls, shape = MODELS[model_name]
    base = randomize_bn(cls()).eval()
    x = torch.randn(*shape)
    prunable, _ = analyze(base)
    assert prunable, "expected at least one prunable layer"
    for name, group in prunable.items():
        n = num_out(get_module(base, name))
        perm = torch.randperm(n)
        keep, dropped = perm[: max(1, n // 2)].sort().values, perm[max(1, n // 2):]

        pruned = prune_channels(copy.deepcopy(base), name, keep).eval()
        ref = copy.deepcopy(base)
        disconnect(ref, group, dropped)
        with torch.no_grad():
            assert torch.allclose(pruned(x), ref.eval()(x), atol=1e-5), f"{model_name}: {name} misaligned"
        assert num_out(get_module(pruned, name)) == len(keep)


@pytest.mark.parametrize("model_name", list(MODELS))
def test_pruning_every_layer_matches_reference(model_name):
    cls, shape = MODELS[model_name]
    base = randomize_bn(cls()).eval()
    x = torch.randn(*shape)
    groups, _ = analyze(base)
    unique = {id(g): g for g in groups.values()}.values()  # layers tied by a residual add share one group

    pruned, ref = copy.deepcopy(base), copy.deepcopy(base)
    for group in unique:
        name = group.producer
        n = num_out(get_module(base, name))
        perm = torch.randperm(n)
        keep, dropped = perm[: max(1, n // 3)].sort().values, perm[max(1, n // 3):]
        prune_channels(pruned, name, keep)
        disconnect(ref, group, dropped)
    with torch.no_grad():
        assert torch.allclose(pruned.eval()(x), ref.eval()(x), atol=1e-5)
    assert sum(p.numel() for p in pruned.parameters()) < sum(p.numel() for p in base.parameters())


def test_expected_groups_for_test_models():
    vgg, _ = analyze(TEST_MODELS["SmallVGG"][0]())
    assert vgg["features.0"].batchnorms == ["features.1"]
    assert vgg["features.14"].consumers == [("classifier.1", 16)]
    assert "classifier.4" not in vgg

    mlp, _ = analyze(TEST_MODELS["SmallMLP"][0]())
    assert list(mlp) == ["net.1", "net.3"]

    res, skipped = analyze(TEST_MODELS["SmallResNet"][0]())
    assert set(res) == {
        "stem.0", "layer1.conv1", "layer1.conv2", "layer2.conv1", "layer2.conv2",
        "layer2.shortcut.0", "layer3.conv1", "layer3.conv2", "layer3.shortcut.0",
    }
    assert set(skipped) == {"fc"}
    assert "model output" in skipped["fc"]

    # layers whose outputs are added share one group
    g1 = res["stem.0"]
    assert res["layer1.conv2"] is g1
    assert g1.producers == ["stem.0", "layer1.conv2"]
    assert set(g1.batchnorms) == {"stem.1", "layer1.bn2"}
    assert {c for c, _ in g1.consumers} == {"layer1.conv1", "layer2.conv1", "layer2.shortcut.0"}

    g2 = res["layer2.conv2"]
    assert res["layer2.shortcut.0"] is g2
    assert set(g2.producers) == {"layer2.conv2", "layer2.shortcut.0"}
    assert {c for c, _ in g2.consumers} == {"layer3.conv1", "layer3.shortcut.0"}

    g3 = res["layer3.conv2"]
    assert res["layer3.shortcut.0"] is g3
    assert g3.consumers == [("fc", 1)]

    # the first conv of each block is not tied to anything
    assert res["layer1.conv1"].producers == ["layer1.conv1"]

    stack, _ = analyze(ResidualStack())
    assert stack["stem.0"] is stack["b1.conv2"] is stack["b2.conv2"]
    assert {c for c, _ in stack["stem.0"].consumers} == {"b1.conv1", "b2.conv1", "head.2"}

    fan, skipped = analyze(FanOut())
    assert fan["conv1"].consumers == [("b", 1), ("a", 1)] or fan["conv1"].consumers == [("a", 1), ("b", 1)]
    assert fan["a"] is fan["b"]
    assert fan["a"].consumers == [("head.2", 1)]
    assert set(skipped) == {"head.2"}


@pytest.mark.parametrize(
    "cls, layer, reason",
    [
        (HardcodedView, "conv", "hard-coded"),
        (Reused, "conv", "more than once"),
        (Concat, "a", "cat"),
        (Depthwise, "conv", "depthwise"),
        (Depthwise, "dw", "depthwise"),
        (Untraceable, "fc1", "torch.fx"),
        (Unused, "spare", "not used"),
        (ResidualOnInput, "conv", "model input"),
        (ResidualDepthwise, "conv", "depthwise"),
        (ResidualDepthwise, "dw", "depthwise"),
        (ResidualToOutput, "a", "model output"),
        (ResidualToOutput, "b", "model output"),
        (ResidualWithCat, "conv", "cat"),
        (ResidualAfterFlatten, "conv", "flatten"),
        (ResidualWithConstant, "conv", "constant"),
    ],
)
def test_unsafe_layers_are_skipped_with_reason(cls, layer, reason):
    skipped = skipped_layers(cls())
    assert layer in skipped
    assert reason in skipped[layer]
    with pytest.raises(ValueError, match="can't be pruned"):
        prune_channels(cls(), layer, [0])


def test_residual_layers_are_pruned_together():
    model = TEST_MODELS["SmallResNet"][0]()
    prune_channels(model, "layer1.conv2", [0, 1, 2, 3])  # any layer of the group prunes the whole group
    for name in ("stem.0", "layer1.conv2"):
        assert get_module(model, name).out_channels == 4
    for name in ("stem.1", "layer1.bn2"):
        assert get_module(model, name).num_features == 4
    for name in ("layer1.conv1", "layer2.conv1", "layer2.shortcut.0"):
        assert get_module(model, name).in_channels == 4
    assert get_module(model, "layer1.conv1").out_channels == 16  # not part of the group
    with torch.no_grad():
        assert model.eval()(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_prunable_layers_yields_one_entry_per_group():
    names = [n for n, _ in prunable_layers(TEST_MODELS["SmallResNet"][0]())]
    assert len(names) == len(set(names)) == 6  # 3 residual groups + 3 inner convs
    assert "stem.0" in names and "layer1.conv2" not in names


@pytest.mark.parametrize("method, options", [("l1_pruning", {}), ("fpgm", {}), ("gradient_based", {"loader": True})])
def test_methods_prune_each_group_once(method, options):
    model = TEST_MODELS["SmallResNet"][0]()
    kwargs = {"ratio": 0.5}
    if options.get("loader"):
        kwargs["loader"] = make_loader((3, 32, 32))
    small = compress(model, method, example_input=torch.randn(2, 3, 32, 32), **kwargs)
    # ratio 0.5 halves each group exactly once, and the layers of a group stay aligned
    assert small.stem[0].out_channels == small.layer1.conv2.out_channels == 8
    assert small.layer2.conv2.out_channels == small.layer2.shortcut[0].out_channels == 16
    assert small.layer3.conv2.out_channels == small.layer3.shortcut[0].out_channels == 32


def test_unused_layer_does_not_block_others():
    assert [n for n, _ in prunable_layers(Unused())] == ["fc1"]


def test_prunable_layers_yields_current_module():
    model = TEST_MODELS["SmallMLP"][0]()
    seen = []
    for name, layer in prunable_layers(model):
        seen.append(layer.in_features)
        prune_channels(model, name, list(range(num_out(layer) // 2)))
    assert seen == [784, 128]


def test_invalid_keep():
    model = TEST_MODELS["SmallMLP"][0]()
    with pytest.raises(ValueError):
        prune_channels(model, "net.1", [])
    with pytest.raises(ValueError):
        prune_channels(model, "net.1", [256])
    with pytest.raises(ValueError):
        prune_channels(model, "net.1", [-1])
    with pytest.raises(ValueError, match="model output"):
        prune_channels(model, "net.5", [0])


def test_skipped_layers_are_logged(caplog):
    caplog.set_level("INFO", logger="shrinker")
    list(prunable_layers(TEST_MODELS["SmallMLP"][0]()))
    assert "Skipping layer net.5" in caplog.text


def test_residual_stateless_batchnorm():
    base = FanOut().eval()
    base.bn = nn.BatchNorm2d(
        6, affine=False, track_running_stats=False
    )
    # Put stateless BatchNorm after the residual addition.
    base.head = nn.Sequential(base.bn, base.head)
    del base.bn

    x = torch.randn(2, 3, 8, 8)
    keep = torch.tensor([0, 2, 4])
    dropped = torch.tensor([1, 3, 5])

    groups, _ = analyze(base)
    reference = copy.deepcopy(base)
    disconnect(reference, groups["a"], dropped)

    pruned = prune_channels(copy.deepcopy(base), "a", keep).eval()

    assert pruned.a.out_channels == pruned.b.out_channels == 3
    assert pruned.head[0].num_features == 3
    assert not pruned.head[0].affine
    assert not pruned.head[0].track_running_stats

    with torch.no_grad():
        torch.testing.assert_close(
            pruned(x), reference(x), atol=1e-5, rtol=1e-5
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
def test_residual_cuda_keep_indices():
    base = FanOut().cuda().eval()
    x = torch.randn(2, 3, 8, 8, device="cuda")
    keep = torch.tensor([0, 2, 4], device="cuda")
    dropped = torch.tensor([1, 3, 5], device="cuda")

    reference = copy.deepcopy(base)
    with torch.no_grad():
        reference.head[2].weight[:, dropped] = 0

    pruned = prune_channels(copy.deepcopy(base), "a", keep).eval()

    assert pruned.head[2].in_features == 3
    assert pruned.head[2].weight.is_cuda

    with torch.no_grad():
        torch.testing.assert_close(
            pruned(x), reference(x), atol=1e-5, rtol=1e-5
        )