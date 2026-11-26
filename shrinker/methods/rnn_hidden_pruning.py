"""Shrinks the hidden size of nn.LSTM / nn.GRU layers.

Paper: Wen et al. 2018, Learning Intrinsic Sparse Structures within LSTM, https://arxiv.org/abs/1709.05027

A hidden unit owns one row in each gate of every layer and direction, a column of the recurrent
weights, and the matching inputs of the next layer and of the nn.Linear that reads the RNN. All of
these are removed together, so the shapes stay consistent. Units are ranked by the L1 norm of their
weights. nn.LSTM has a single hidden_size, so the same units are kept in every layer.

An RNN is left unchanged when its output is used in a way this method can't follow, for example
c_n, an explicit h0, proj_size > 0, torch.cat of the h_n directions, or the RNN being called twice.
"""

from __future__ import annotations

import logging
import operator
from typing import Dict, Optional, Union

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from shrinker.registry import register
from shrinker.utils import (
    get_module,
    is_shared,
    keep_indices,
    replace_module,
    resolve_module_ratios,
    slice_linear_in,
)

logger = logging.getLogger(__name__)

_GATES = {nn.LSTM: 4, nn.GRU: 3}
_ELEMENTWISE_MODULES = (nn.ReLU, nn.Tanh, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.Dropout, nn.Identity)
_ELEMENTWISE_FUNCS = {F.relu, torch.relu, torch.tanh, torch.sigmoid, F.gelu, F.dropout}


@register(kind="pruning", paper="https://arxiv.org/abs/1709.05027")
def apply(model: nn.Module, ratio: Union[float, Dict[str, float]] = 0.3) -> nn.Module:
    """Remove the hidden units with the smallest L1 norm from every nn.LSTM / nn.GRU.

    ratio: one fraction used for every RNN, or {RNN module name: fraction}. RNNs missing
    from the dict are left unpruned.
    """
    rnns = {name: m for name, m in model.named_modules() if type(m) in _GATES}
    if not rnns:
        return model
    ratios = resolve_module_ratios(rnns, ratio, "nn.LSTM/nn.GRU modules")
    try:
        graph = fx.Tracer(autowrap_functions=(pack_padded_sequence, pad_packed_sequence)).trace(model)
    except Exception as e:  # noqa: BLE001  an untraceable model is left unchanged
        logger.info("rnn_hidden_pruning: model could not be traced with torch.fx (%s); unchanged", e)
        return model

    for name, rnn in rnns.items():
        if name not in ratios:
            continue
        if getattr(rnn, "proj_size", 0) or is_shared(model, rnn):
            logger.info("Skipping %s: proj_size and shared RNNs are not supported", name)
            continue
        consumers = _consumers(graph, dict(model.named_modules()), name)
        if consumers is None:
            logger.info("Skipping %s: its output is used in a way that can't be pruned", name)
            continue
        keep = keep_indices(_unit_scores(rnn), ratios[name])
        if len(keep) == rnn.hidden_size:
            continue

        directions = 2 if rnn.bidirectional else 1
        output_cols = torch.cat([keep + d * rnn.hidden_size for d in range(directions)])
        for consumer, reads in consumers.items():
            cols = output_cols if reads == "output" else keep
            replace_module(model, consumer, slice_linear_in(get_module(model, consumer), cols))
        replace_module(model, name, _rebuild(rnn, keep))
    return model


def _suffixes(rnn: nn.Module):
    directions = 2 if rnn.bidirectional else 1
    return [f"_l{k}" + ("_reverse" if d else "") for k in range(rnn.num_layers) for d in range(directions)]


def _unit_scores(rnn: nn.Module) -> torch.Tensor:
    G, H = _GATES[type(rnn)], rnn.hidden_size
    scores = torch.zeros(H, device=rnn.weight_hh_l0.device)
    for s in _suffixes(rnn):
        w_ih = getattr(rnn, "weight_ih" + s).detach().abs()
        w_hh = getattr(rnn, "weight_hh" + s).detach().abs()
        scores += (w_ih.view(G, H, -1).sum((0, 2)) + w_hh.view(G, H, -1).sum((0, 2)) + w_hh.sum(0)).float()
    return scores


def _rebuild(rnn: nn.Module, keep: torch.Tensor) -> nn.Module:
    """A copy of ``rnn`` holding only the hidden units in ``keep``."""
    G, H = _GATES[type(rnn)], rnn.hidden_size
    directions = 2 if rnn.bidirectional else 1
    rows = torch.cat([keep + g * H for g in range(G)])
    cols = torch.cat([keep + d * H for d in range(directions)])
    new = type(rnn)(
        rnn.input_size, len(keep), rnn.num_layers, bias=rnn.bias, batch_first=rnn.batch_first,
        dropout=rnn.dropout, bidirectional=rnn.bidirectional,
        device=rnn.weight_hh_l0.device, dtype=rnn.weight_hh_l0.dtype,
    )
    with torch.no_grad():
        for s in _suffixes(rnn):
            w_ih = getattr(rnn, "weight_ih" + s)[rows]
            getattr(new, "weight_ih" + s).copy_(w_ih if s.startswith("_l0") else w_ih[:, cols])
            getattr(new, "weight_hh" + s).copy_(getattr(rnn, "weight_hh" + s)[rows][:, keep])
            if rnn.bias:
                getattr(new, "bias_ih" + s).copy_(getattr(rnn, "bias_ih" + s)[rows])
                getattr(new, "bias_hh" + s).copy_(getattr(rnn, "bias_hh" + s)[rows])
    for name, param in new.named_parameters():
        param.requires_grad_(getattr(rnn, name).requires_grad)
    new.train(rnn.training)
    if new.weight_hh_l0.is_cuda:
        new.flatten_parameters()
    return new


def _consumers(graph: fx.Graph, modules: Dict[str, nn.Module], name: str) -> Optional[Dict[str, str]]:
    """{nn.Linear name: "output" or "h_n"} for every layer reading the RNN, or None if any use is unsupported.

    The walk keeps track of each tensor's rank, so it can tell a reduction over time from one
    over the feature dimension.
    """
    calls = [n for n in graph.nodes if n.op == "call_module" and n.target == name]
    if len(calls) != 1 or len(calls[0].args) != 1 or calls[0].kwargs:
        return None  # called twice, or given h0, whose size depends on hidden_size
    node = calls[0]
    rnn = modules[name]
    widths = {"output": rnn.hidden_size * (2 if rnn.bidirectional else 1), "h_n": rnn.hidden_size}
    packed = _is_call(node.args[0], pack_padded_sequence)

    stack = []
    for user in node.users:
        if not _is_getitem(user) or user.args[1] not in (0, 1):
            return None
        if user.args[1] == 0:
            stack += [(u, "packed" if packed else "output", 3) for u in user.users]
        else:
            stack += [(u, "h_c" if type(rnn) is nn.LSTM else "h_n", 3) for u in user.users]

    found: Dict[str, str] = {}
    while stack:
        cur, kind, rank = stack.pop()
        if _is_getitem(cur) and not cur.users:
            continue  # unpacked but never used, like c in `_, (h, c) = rnn(x)`
        if kind == "packed":
            if not _is_call(cur, pad_packed_sequence):
                return None
            stack += [(u, "padded", rank) for u in cur.users]
        elif kind == "padded":  # pad_packed_sequence returns (output, lengths)
            if not _is_getitem(cur):
                return None
            if cur.args[1] == 0:
                stack += [(u, "output", rank) for u in cur.users]
        elif kind == "h_c":  # (h_n, c_n): only h_n is supported
            if not _is_getitem(cur) or cur.args[1] != 0:
                return None
            stack += [(u, "h_n", rank) for u in cur.users]
        elif _is_getitem(cur):
            new_rank = _rank_after_index(cur.args[1], rank)
            if new_rank is None:
                return None
            stack += [(u, kind, new_rank) for u in cur.users]
        elif cur.op == "call_method" and cur.target in ("mean", "sum"):
            new_rank = _rank_after_reduce(cur, rank)
            if new_rank is None:
                return None
            stack += [(u, kind, new_rank) for u in cur.users]
        elif _is_elementwise(cur, modules):
            stack += [(u, kind, rank) for u in cur.users]
        elif cur.op == "call_module" and type(modules[cur.target]) is nn.Linear:
            if modules[cur.target].in_features != widths[kind] or found.get(cur.target, kind) != kind:
                return None
            found[cur.target] = kind
        else:
            return None
    return found or None


def _is_getitem(node) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function" and node.target is operator.getitem


def _is_call(node, fn) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function" and node.target is fn


def _is_elementwise(node: fx.Node, modules) -> bool:
    if node.op == "call_module":
        return isinstance(modules[node.target], _ELEMENTWISE_MODULES)
    if node.op == "call_function":
        return node.target in _ELEMENTWISE_FUNCS
    return node.op == "call_method" and node.target in ("contiguous", "relu", "tanh")


def _rank_after_index(index, rank: int) -> Optional[int]:
    """Rank after ``x[index]``, or None if the index could touch the feature (last) dimension."""
    index = index if isinstance(index, tuple) else (index,)
    if len(index) > rank or not all(isinstance(i, (int, slice)) for i in index):
        return None
    if len(index) == rank and index[-1] != slice(None):
        return None
    return rank - sum(isinstance(i, int) for i in index)


def _rank_after_reduce(node: fx.Node, rank: int) -> Optional[int]:
    """Rank after ``x.mean(dim)`` / ``x.sum(dim)``, or None if it reduces the feature dimension."""
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    if not dims or not all(isinstance(d, int) for d in dims):
        return None
    if any(d % rank == rank - 1 for d in dims):
        return None
    keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
    return rank if keepdim else rank - len(dims)
