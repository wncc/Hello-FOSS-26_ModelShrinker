"""Removes the filters/neurons whose removal would change the loss the least (Taylor importance).
 

The importance of a filter is estimated with a first-order Taylor expansion of the loss: removing
a parameter w with loss gradient g changes the loss by about g * w. A filter's score combines
this term over all of its parameters (its weights and bias, plus the scale and shift of the
BatchNorm that follows it). Scores are averaged over a few mini-batches, and in every prunable
layer the filters with the lowest scores are removed.
 
Unlike weight-only criteria (l1_pruning, fpgm) this method needs data and a loss, so it takes a
``loader`` that yields ``(inputs, labels)``.
 
Two ways of combining a filter's per-parameter terms are available through ``reduce``:
    "sum_of_squares" (default)  score = sum over the filter's parameters of (g * w) ** 2
    "square_of_sum"             score = (sum over the filter's parameters of g * w) ** 2
 
Gradients come from mini-batches (a cheaper stand-in for per-sample gradients), with the model
in eval mode so BatchNorm statistics and dropout are not disturbed.
"""
 
from __future__ import annotations
 
import itertools
import logging
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
 
import torch
import torch.nn.functional as F
from torch import nn
 
from shrinker.dependency import analyze
from shrinker.registry import register
from shrinker.utils import get_module, keep_indices, prune_channels, resolve_ratios
 
logger = logging.getLogger(__name__)
 
_REDUCTIONS = ("sum_of_squares", "square_of_sum")
 
 
@register(kind="pruning", paper="https://arxiv.org/abs/1906.10771")
def apply(
    model: nn.Module,
    *,
    loader: Iterable,
    ratio: Union[float, Dict[str, float]]= 0.3,
    num_batches: int = 10,
    reduce: str = "sum_of_squares",
    loss_fn: Optional[Callable] = None,
) -> nn.Module:
    """Remove the filters/neurons with the lowest first-order Taylor importance.
 
    loader:      yields (inputs, labels) batches used to estimate importance.
    ratio:       ratio:fraction to remove from every prunable layer, or {layer name: fraction} (layers missing from the dict are left unpruned).
    num_batches: number of batches to average the importance over.
    reduce:      "sum_of_squares" or "square_of_sum" (see the module docstring).
    loss_fn:     loss(outputs, labels); defaults to cross-entropy.
    """
    ratios = resolve_ratios(model, ratio)
    if num_batches < 1:
        raise ValueError(f"num_batches must be at least 1, got {num_batches}")
    if reduce not in _REDUCTIONS:
        raise ValueError(f"reduce must be one of {_REDUCTIONS}, got {reduce!r}")
 
    prunable, skipped = analyze(model)
    for name, reason in skipped.items():
        logger.info("Skipping %s: %s", name, reason)
    prunable = {name: group for name, group in prunable.items() if name in ratios}
    if not prunable:
        logger.info("No prunable layers found; returning the model unchanged")
        return model
 
    scores = _taylor_scores(model, prunable, loader, num_batches, reduce, loss_fn or F.cross_entropy)
    for name, layer_scores in scores.items():
        prune_channels(model, name, keep_indices(layer_scores, ratios[name]))
    return model
 
 
def _filter_parameters(model: nn.Module, name: str, batchnorms: List[str]) -> List[nn.Parameter]:
    """Every parameter that belongs to the filters of layer ``name`` (one row/entry per filter)."""
    layer = get_module(model, name)
    params = [layer.weight]
    if layer.bias is not None:
        params.append(layer.bias)
    for bn_name in batchnorms:
        bn = get_module(model, bn_name)
        if bn.affine:
            params += [bn.weight, bn.bias]
    return params
 
 
def _taylor_scores(
    model: nn.Module,
    prunable: Dict[str, object],
    loader: Iterable,
    num_batches: int,
    reduce: str,
    loss_fn: Callable,
) -> Dict[str, torch.Tensor]:
    """Importance of every filter of every prunable layer, as {layer name: 1-D tensor}."""
    owned = {name: _filter_parameters(model, name, group.batchnorms) for name, group in prunable.items()}
    all_params = [p for params in owned.values() for p in params]
    device = all_params[0].device
 
    was_training = model.training
    frozen = [p for p in all_params if not p.requires_grad]
    for p in frozen:
        p.requires_grad_(True)  # a frozen model still needs gradients for scoring
    model.eval()
 
    totals = {name: torch.zeros(params[0].shape[0], device=device) for name, params in owned.items()}
    seen = 0
    try:
        with torch.enable_grad():
            for batch in itertools.islice(loader, num_batches):
                inputs, labels = _unpack(batch)
                loss = loss_fn(model(inputs.to(device)), labels.to(device))
                grads = torch.autograd.grad(loss, all_params, allow_unused=True)
                grad_of = {id(p): g for p, g in zip(all_params, grads)}
                for name, params in owned.items():
                    totals[name] += _score(params, grad_of, reduce)
                seen += 1
    finally:
        for p in frozen:
            p.requires_grad_(False)
        model.train(was_training)
 
    if seen == 0:
        raise ValueError("loader yielded no batches, so importance cannot be estimated")
    return {name: total / seen for name, total in totals.items()}
 
 
def _unpack(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("loader must yield (inputs, labels) pairs")
    return batch[0], batch[1]
 
 
def _score(params: List[nn.Parameter], grad_of: Dict[int, Optional[torch.Tensor]], reduce: str) -> torch.Tensor:
    """Per-filter score from one batch: combine g * w over the filter's parameters."""
    terms = []
    for p in params:
        g = grad_of[id(p)]
        if g is None:  # parameter did not affect the loss
            g = torch.zeros_like(p)
        terms.append((g * p).detach().flatten(1) if p.dim() > 1 else (g * p).detach().unsqueeze(1))
    taylor = torch.cat(terms, dim=1).float()  # [filters, all terms of that filter]
    if reduce == "sum_of_squares":
        return taylor.pow(2).sum(dim=1)
    return taylor.sum(dim=1).pow(2)