from __future__ import annotations

import copy
import inspect
from typing import Any, Optional

import torch
from torch import nn

from .registry import get_method


class CompressionError(RuntimeError):
    pass


def compress(
    model: nn.Module,
    method: str,
    example_input: Optional[Any] = None,
    **kwargs: Any,
) -> nn.Module:
    """Return a compressed copy of ``model``. The original is never modified.

    If ``example_input`` is given, the result must give the same output shape on it.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"compress() expects an nn.Module, got {type(model).__name__}")
    if not isinstance(method, str):
        hint = " To apply several methods, use shrinker.Pipeline([...])." if isinstance(method, (list, tuple)) else ""
        raise TypeError(f"compress() expects a method name (str), got {type(method).__name__}.{hint}")
    info = get_method(method)

    try:
        work = copy.deepcopy(model)
    except Exception as e:
        raise TypeError(f"Model could not be deep-copied, so it cannot be compressed safely: {e}") from e

    expected = _output_shapes(work, example_input) if example_input is not None else None

    params = inspect.signature(info.fn).parameters
    accepts_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
    if example_input is not None and ("example_input" in params or accepts_any):
        kwargs["example_input"] = example_input

    try:
        inspect.signature(info.fn).bind(work, **kwargs)
    except TypeError as e:
        options = [n for n in list(params)[1:] if params[n].kind is not params[n].VAR_KEYWORD]
        raise TypeError(
            f"Bad arguments for method {method!r}: {e}. It accepts: {', '.join(options) or '(nothing)'}"
        ) from None

    result = info.fn(work, **kwargs)

    if not isinstance(result, nn.Module):
        raise CompressionError(
            f"Method {method!r} ({info.module}) must return an nn.Module, got {type(result).__name__}. "
            "Did you forget `return model`?"
        )
    result.train(model.training)

    if expected is not None:
        try:
            got = _output_shapes(result, example_input)
        except Exception as e:
            raise CompressionError(
                f"Model compressed with {method!r} ({info.module}) crashes on example_input: "
                f"{type(e).__name__}: {e}"
            ) from e
        if got != expected:
            raise CompressionError(
                f"Method {method!r} ({info.module}) changed the output shape from {expected} to {got}."
            )
    return result


def _output_shapes(model: nn.Module, example_input: Any):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(*example_input) if isinstance(example_input, tuple) else model(example_input)
    finally:
        model.train(was_training)
    return _shapes(out)


def _shapes(obj):
    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        return tuple(_shapes(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _shapes(v) for k, v in obj.items()}
    return type(obj).__name__
