"""Least-squares refit shared by rnn_hidden_pruning, ffn_pruning and head_pruning.

After pruning, the layer that reads the pruned units gets fewer inputs. Given a few batches of
inputs (labels are not used), its weights are solved again so it reproduces what the original
model produced on them, as in the layer-wise reconstruction step of Kwon et al. 2022
(https://arxiv.org/abs/2204.09656). The ``_`` prefix keeps ``registry.discover()`` from treating
this file as a method.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from shrinker.utils import unfused_transformer_layers

# Below this many samples per input feature the solve is too noisy, so the sliced weights are kept.
MIN_ROWS_PER_FEATURE = 4


def calibration_inputs(loader, max_batches: int) -> List:
    """Model inputs from the first ``max_batches`` batches of ``loader``."""
    if loader is None:
        return []
    if not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError(f"max_batches must be a positive integer, got {max_batches!r}")
    inputs = []
    for batch in itertools.islice(loader, max_batches):
        x = batch[0] if isinstance(batch, (list, tuple)) and len(batch) == 2 else batch
        inputs.append(tuple(x) if isinstance(x, list) else x)
    return inputs


def record(model: nn.Module, modules: List[nn.Module], inputs: List, *, outputs: bool = False,
           with_kwargs: bool = False) -> Dict[nn.Module, list]:
    """Run ``inputs`` through ``model`` in eval mode and collect what each module receives.

    One entry per call: the positional args, (args, kwargs) with ``with_kwargs``, or
    (args, output) with ``outputs``.
    """
    seen: Dict[nn.Module, list] = {m: [] for m in modules}

    def keep(m, *values):
        seen[m].append(_detach(values if len(values) > 1 else values[0]))

    hooks = []
    for m in modules:
        if outputs:
            hooks.append(m.register_forward_hook(lambda mod, a, o, m=m: keep(m, a, o)))
        else:
            hooks.append(m.register_forward_pre_hook(lambda mod, *a, m=m: keep(m, *a), with_kwargs=with_kwargs))
    was_training = model.training
    model.eval()
    try:
        for x in inputs:
            with unfused_transformer_layers(x) as x:
                if isinstance(x, tuple):
                    model(*x)
                else:
                    model(x)
    finally:
        model.train(was_training)
        for h in hooks:
            h.remove()
    return seen


def ridge(X: torch.Tensor, Y: torch.Tensor, lam: float = 1e-3) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Weight and bias minimizing ``|X W^T + b - Y|^2 + lam * scale * |W|^2``, or None if there is too little data."""
    X, Y = X.detach().double(), Y.detach().double()
    if X.shape[0] < MIN_ROWS_PER_FEATURE * (X.shape[1] + 1):
        return None
    X1 = torch.cat([X, X.new_ones(len(X), 1)], dim=1)
    A = X1.T @ X1
    A += lam * A.diagonal().mean().clamp_min(1e-12) * torch.eye(len(A), dtype=A.dtype, device=A.device)
    try:
        solution = torch.linalg.solve(A, X1.T @ Y)
    except RuntimeError:
        return None
    if not torch.isfinite(solution).all():
        return None
    return solution[:-1].T, solution[-1]


def set_linear(linear: nn.Linear, fit) -> bool:
    """Copy a ``ridge`` result into ``linear``; does nothing and returns False if there is none."""
    if fit is None or linear.bias is None:
        return False
    with torch.no_grad():
        linear.weight.copy_(fit[0].to(linear.weight))
        linear.bias.copy_(fit[1].to(linear.bias))
    return True


def _detach(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, (tuple, list)):
        return type(obj)(_detach(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _detach(v) for k, v in obj.items()}
    return obj
