import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from examples.test_models import TEST_MODELS
from shrinker.dependency import analyze
from shrinker.utils import get_module, prunable_layers, prune_channels, skipped_layers




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


MODELS = {
    **{name: (cls, shape) for name, (cls, shape) in TEST_MODELS.items()},
    "FunctionalCNN": (FunctionalCNN, (3, 3, 16, 16)),
    "FanOut": (FanOut, (2, 3, 8, 8)),
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

    pruned, ref = copy.deepcopy(base), copy.deepcopy(base)
    for name, group in groups.items():
        n = num_out(get_module(base, name))
        perm = torch.randperm(n)
        keep, dropped = perm[: max(1, n // 3)].sort().values, perm[max(1, n // 3):]
        prune_channels(pruned, name, keep)
        disconnect(ref, group, dropped)
    with torch.no_grad():
        assert torch.allclose(pruned.eval()(x), ref.eval()(x), atol=1e-5)
    assert sum(p.numel() for p in pruned.parameters()) < sum(p.numel() for p in base.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("model_name", list(MODELS))
def test_pruning_on_cuda_matches_cpu(model_name):
    cls, shape = MODELS[model_name]
    base = randomize_bn(cls()).eval()
    x = torch.randn(*shape)
    groups, _ = analyze(base)

    on_cpu, on_gpu = copy.deepcopy(base), copy.deepcopy(base).cuda()
    for name in groups:
        n = num_out(get_module(base, name))
        keep = torch.randperm(n)[: max(1, n // 2)].sort().values
        prune_channels(on_cpu, name, keep)
        prune_channels(on_gpu, name, keep.cuda())  # keep lives on the model's device, as in the methods
    with torch.no_grad():
        assert torch.allclose(on_gpu(x.cuda()).cpu(), on_cpu(x), atol=1e-3, rtol=1e-3)


def test_expected_groups_for_test_models():
    vgg, _ = analyze(TEST_MODELS["SmallVGG"][0]())
    assert vgg["features.0"].batchnorms == ["features.1"]
    assert vgg["features.14"].consumers == [("classifier.1", 16)]
    assert "classifier.4" not in vgg

    mlp, _ = analyze(TEST_MODELS["SmallMLP"][0]())
    assert list(mlp) == ["net.1", "net.3"]

    res, skipped = analyze(TEST_MODELS["SmallResNet"][0]())
    assert set(res) == {"layer1.conv1", "layer2.conv1", "layer3.conv1"}
    assert "residual" in skipped["layer1.conv2"]
    assert "model output" in skipped["fc"]

    fan, skipped = analyze(FanOut())
    assert fan["conv1"].consumers == [("b", 1), ("a", 1)] or fan["conv1"].consumers == [("a", 1), ("b", 1)]
    assert "a" in skipped and "b" in skipped


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
    ],
)
def test_unsafe_layers_are_skipped_with_reason(cls, layer, reason):
    skipped = skipped_layers(cls())
    assert layer in skipped
    assert reason in skipped[layer]
    with pytest.raises(ValueError, match="can't be pruned"):
        prune_channels(cls(), layer, [0])


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
