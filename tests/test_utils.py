import pytest
import torch
from torch import nn

from shrinker.utils import (
    keep_indices,
    replace_module,
    slice_bn,
    slice_conv_in,
    slice_conv_out,
    slice_linear_in,
    slice_linear_out,
)

KEEP = [0, 2, 5]


def test_slice_conv_out():
    conv = nn.Conv2d(3, 8, 3, padding=1, stride=2, dilation=1)
    x = torch.randn(2, 3, 9, 9)
    new = slice_conv_out(conv, KEEP)
    assert new.out_channels == 3 and new.weight.shape == (3, 3, 3, 3)
    assert (new.stride, new.padding) == (conv.stride, conv.padding)
    assert torch.allclose(new(x), conv(x)[:, KEEP])


def test_slice_conv_in():
    conv = nn.Conv2d(8, 4, 3, padding=1)
    x = torch.randn(2, 8, 6, 6)
    new = slice_conv_in(conv, KEEP)
    mask = torch.zeros(8)
    mask[KEEP] = 1
    assert new.in_channels == 3
    assert torch.allclose(new(x[:, KEEP]), conv(x * mask[None, :, None, None]), atol=1e-6)


def test_slice_linear():
    lin = nn.Linear(8, 6)
    x = torch.randn(4, 8)
    assert torch.allclose(slice_linear_out(lin, KEEP)(x), lin(x)[:, KEEP])
    mask = torch.zeros(8)
    mask[KEEP] = 1
    assert torch.allclose(slice_linear_in(lin, KEEP)(x[:, KEEP]), lin(x * mask), atol=1e-6)


@pytest.mark.parametrize("affine", [True, False])
def test_slice_bn(affine):
    bn = nn.BatchNorm2d(8, affine=affine)
    bn.running_mean.uniform_(-1, 1)
    bn.running_var.uniform_(0.5, 2)
    bn.eval()
    x = torch.randn(2, 8, 4, 4)
    new = slice_bn(bn, KEEP).eval()
    assert new.num_features == 3
    assert torch.allclose(new(x[:, KEEP]), bn(x)[:, KEEP], atol=1e-6)


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("bn_type", [nn.BatchNorm1d, nn.BatchNorm2d])
def test_slice_bn_without_running_stats(bn_type, affine):
    # affine=False with track_running_stats=False leaves the layer with no tensors at all
    bn = bn_type(8, affine=affine, track_running_stats=False)
    if affine:
        with torch.no_grad():
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.uniform_(-0.5, 0.5)
    x = torch.randn(4, 8) if bn_type is nn.BatchNorm1d else torch.randn(4, 8, 4, 4)
    new = slice_bn(bn, KEEP)
    assert type(new) is bn_type and new.num_features == 3
    assert (new.affine, new.track_running_stats) == (affine, False)
    assert torch.allclose(new(x[:, KEEP]), bn(x)[:, KEEP], atol=1e-6)


def test_pruning_a_layer_followed_by_a_batchnorm_without_tensors():
    from shrinker import compress

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8, affine=False, track_running_stats=False),
        nn.ReLU(),
        nn.Conv2d(8, 4, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    small = compress(model, "l1_pruning", ratio=0.5, example_input=torch.randn(2, 3, 8, 8))
    assert small[0].out_channels == 4 and small[1].num_features == 4


def test_slicing_makes_new_modules_and_keeps_mode():
    lin = nn.Linear(8, 6).eval()
    new = slice_linear_out(lin, KEEP)
    assert new is not lin and not new.training
    assert lin.out_features == 6


def test_bad_indices():
    lin = nn.Linear(8, 6)
    with pytest.raises(ValueError):
        slice_linear_out(lin, [])
    with pytest.raises(ValueError):
        slice_linear_out(lin, [6])
    with pytest.raises(ValueError, match="depthwise"):
        slice_conv_out(nn.Conv2d(4, 4, 3, groups=4), [0])


def test_unsorted_duplicate_indices_are_normalized():
    lin = nn.Linear(8, 6)
    new = slice_linear_out(lin, [5, 0, 2, 2])
    assert torch.equal(new.weight, lin.weight[[0, 2, 5]])


def test_replace_module_nested():
    model = nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.ReLU(), nn.Linear(2, 3)))
    new = nn.Linear(2, 3)
    replace_module(model, "1.1", new)
    assert model[1][1] is new
    with pytest.raises(AttributeError):
        replace_module(model, "1.7", new)


def test_keep_indices():
    scores = torch.tensor([0.1, 5.0, 0.3, 4.0, 2.0])
    assert keep_indices(scores, 0.4).tolist() == [1, 3, 4]
    assert keep_indices(scores, 0.0).tolist() == [0, 1, 2, 3, 4]
    assert keep_indices(scores, 0.99).tolist() == [1]
    with pytest.raises(ValueError):
        keep_indices(scores, 1.0)
