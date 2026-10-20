from __future__ import annotations

from typing import Dict, Sequence, Union

import torch
from torch import nn

Indices = Union[Sequence[int], torch.Tensor]


def get_module(model: nn.Module, name: str) -> nn.Module:
    return model.get_submodule(name)


def replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    if not name:
        raise ValueError("Cannot replace the root module")
    get_module(model, name)
    parent_name, _, attr = name.rpartition(".")
    setattr(get_module(model, parent_name), attr, new_module)


def keep_indices(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    """Sorted indices of the top (1 - ratio) scores. Always keeps at least one."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    scores = torch.as_tensor(scores).detach().flatten()
    n_keep = max(1, int(round(scores.numel() * (1.0 - ratio))))
    return torch.topk(scores, n_keep).indices.sort().values


def _index(keep: Indices, size: int, what: str, device) -> torch.Tensor:
    idx = torch.as_tensor(keep, dtype=torch.long).flatten().to(device)
    if idx.numel() == 0:
        raise ValueError(f"{what}: keep must contain at least one index")
    idx = torch.unique(idx)
    if idx[0] < 0 or idx[-1] >= size:
        raise ValueError(f"{what}: keep indices must be in [0, {size}), got min={idx[0]}, max={idx[-1]}")
    return idx


def _copy_param(dst: nn.Parameter, src: torch.Tensor) -> None:
    with torch.no_grad():
        dst.copy_(src)
    dst.requires_grad_(src.requires_grad)


def _check_conv(conv: nn.Module) -> None:
    if type(conv) is not nn.Conv2d:
        raise TypeError(f"Expected nn.Conv2d, got {type(conv).__name__}")
    if conv.groups != 1:
        raise ValueError("Grouped/depthwise convolutions (groups != 1) are not supported")


def _new_conv(conv: nn.Conv2d, in_channels: int, out_channels: int) -> nn.Conv2d:
    new = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    return new.train(conv.training)


def slice_conv_out(conv: nn.Conv2d, keep: Indices) -> nn.Conv2d:
    """New Conv2d containing only the output filters in ``keep``."""
    _check_conv(conv)
    idx = _index(keep, conv.out_channels, "slice_conv_out", conv.weight.device)
    new = _new_conv(conv, conv.in_channels, len(idx))
    _copy_param(new.weight, conv.weight[idx])
    if conv.bias is not None:
        _copy_param(new.bias, conv.bias[idx])
    return new


def slice_conv_in(conv: nn.Conv2d, keep: Indices) -> nn.Conv2d:
    """New Conv2d containing only the input channels in ``keep``."""
    _check_conv(conv)
    idx = _index(keep, conv.in_channels, "slice_conv_in", conv.weight.device)
    new = _new_conv(conv, len(idx), conv.out_channels)
    _copy_param(new.weight, conv.weight[:, idx])
    if conv.bias is not None:
        _copy_param(new.bias, conv.bias)
    return new


def _new_linear(linear: nn.Linear, in_features: int, out_features: int) -> nn.Linear:
    new = nn.Linear(
        in_features,
        out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    return new.train(linear.training)


def _check_linear(linear: nn.Module) -> None:
    if type(linear) is not nn.Linear:
        raise TypeError(f"Expected nn.Linear, got {type(linear).__name__}")


def slice_linear_out(linear: nn.Linear, keep: Indices) -> nn.Linear:
    """New Linear containing only the output neurons in ``keep``."""
    _check_linear(linear)
    idx = _index(keep, linear.out_features, "slice_linear_out", linear.weight.device)
    new = _new_linear(linear, linear.in_features, len(idx))
    _copy_param(new.weight, linear.weight[idx])
    if linear.bias is not None:
        _copy_param(new.bias, linear.bias[idx])
    return new


def slice_linear_in(linear: nn.Linear, keep: Indices) -> nn.Linear:
    """New Linear containing only the input features in ``keep``."""
    _check_linear(linear)
    idx = _index(keep, linear.in_features, "slice_linear_in", linear.weight.device)
    new = _new_linear(linear, len(idx), linear.out_features)
    _copy_param(new.weight, linear.weight[:, idx])
    if linear.bias is not None:
        _copy_param(new.bias, linear.bias)
    return new


def slice_bn(bn: nn.modules.batchnorm._BatchNorm, keep: Indices) -> nn.modules.batchnorm._BatchNorm:
    """New BatchNorm1d/2d containing only the channels in ``keep``."""
    if type(bn) not in (nn.BatchNorm1d, nn.BatchNorm2d):
        raise TypeError(f"Expected nn.BatchNorm1d or nn.BatchNorm2d, got {type(bn).__name__}")
    device = (bn.weight if bn.weight is not None else bn.running_mean).device
    idx = _index(keep, bn.num_features, "slice_bn", device)
    ref = bn.weight if bn.weight is not None else bn.running_mean
    new = type(bn)(
        len(idx),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
        device=device,
        dtype=None if ref is None else ref.dtype,
    )
    if bn.affine:
        _copy_param(new.weight, bn.weight[idx])
        _copy_param(new.bias, bn.bias[idx])
    if bn.track_running_stats:
        new.running_mean.copy_(bn.running_mean[idx])
        new.running_var.copy_(bn.running_var[idx])
        new.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new.train(bn.training)

def resolve_ratios(model: nn.Module, ratio: Union[float, Dict[str, float]]) -> Dict[str, float]:
    """Turn ``ratio`` (a float, or {layer name: float}) into {layer name: float} for prunable layers."""
    prunable, skipped = analyze(model)
    if isinstance(ratio, dict):
        for name, value in ratio.items():
            if name not in prunable:
                reason = skipped.get(name, "no Conv2d/Linear layer has this name")
                raise ValueError(f"Layer {name!r} can't be pruned: {reason}")
            keep_indices(torch.ones(1), value)
        return dict(ratio)
    keep_indices(torch.ones(1), ratio)
    return {name: ratio for name in prunable}


from .dependency import analyze, prunable_layers, prune_channels, skipped_layers  # noqa: E402,F401