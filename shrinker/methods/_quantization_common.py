"""Helpers shared by the CPU INT8 quantization methods.

``qat_quantization``, ``ptq_quantization`` and ``dfq_quantization`` all lower
models onto PyTorch's CPU quantized kernels.  This module keeps the pieces
they have in common — backend selection, engine switching, argument
normalization and post-conversion graph fixes — in one place instead of one
copy per method file.  The ``_`` prefix keeps ``registry.discover()`` from
importing this file as a method.
"""

from __future__ import annotations

import platform
from contextlib import contextmanager
from typing import Any, Iterator, Tuple

import torch
from torch import nn


def _as_args(inputs: Any) -> Tuple[Any, ...]:
    return inputs if isinstance(inputs, tuple) else (inputs,)


def _select_backend(backend: str) -> str:
    """Return a supported CPU quantization backend for ``backend``."""
    available = tuple(torch.backends.quantized.supported_engines)
    if backend == "auto":
        machine = platform.machine().lower()
        candidates = ("qnnpack",) if machine.startswith(("arm", "aarch64")) else ("x86", "onednn", "fbgemm")
        for candidate in candidates:
            if candidate in available:
                return candidate
        raise RuntimeError(f"No supported CPU quantization backend is available; PyTorch reports {available}")
    if backend not in ("x86", "fbgemm", "onednn", "qnnpack"):
        raise ValueError("backend must be 'auto', 'x86', 'fbgemm', 'onednn', or 'qnnpack'")
    if backend not in available:
        raise RuntimeError(f"Quantization backend {backend!r} is unavailable; PyTorch reports {available}")
    return backend


@contextmanager
def _quantized_engine(backend: str) -> Iterator[None]:
    """Temporarily select a quantized CPU backend without leaking global state."""
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = backend
    try:
        yield
    finally:
        torch.backends.quantized.engine = previous


def _make_views_layout_safe(module: nn.Module) -> None:
    """Use reshape for traced ``view`` calls after quantization changes tensor layout.

    ``reshape`` has the same value and shape semantics as ``view`` but creates a
    contiguous copy when a quantized operator returns a non-contiguous tensor.
    """
    graph = getattr(module, "graph", None)
    if graph is None:
        return
    changed = False
    for node in graph.nodes:
        if node.op == "call_method" and node.target == "view":
            node.target = "reshape"
            changed = True
    if changed:
        graph.lint()
        module.recompile()
