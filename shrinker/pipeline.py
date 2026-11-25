"""Run several compression methods one after another (e.g. prune, then quantize)."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from torch import nn

from .core import CompressionError, compress
from .registry import UnknownMethodError, get_method

# A step is a method name, or a (name, options) pair.
Step = Union[str, Tuple[str, Mapping[str, Any]]]


class Pipeline:
    """Run compression methods in a row.

    Example::

        pipe = shrinker.Pipeline([
            ("l1_pruning", {"ratio": 0.3}),   # step with its own options
            "dfq_quantization",                # step with default options
        ])
        small = pipe(model, example_input=x)

    - Steps run in the order you give.
    - Mistakes are caught when the pipeline is built, before any work is done.
    - Quantization must be the last step.
    - Your original model is never changed.
    """

    def __init__(self, steps: Sequence[Step]):
        # 1. steps must be a non-empty list (a plain string is a common mistake)
        if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
            raise TypeError(
                "Pipeline expects a list of steps, e.g. Pipeline(['l1_pruning', 'dfq_quantization']); "
                f"got {type(steps).__name__}"
            )
        if len(steps) == 0:
            raise ValueError("Pipeline needs at least one step")

        # 2. check each step, and make sure nothing comes after quantization
        parsed = []
        quantized_at: Optional[Tuple[int, str]] = None  # where we saw a quantization step
        for i, step in enumerate(steps, 1):
            name, options = _parse_step(step, i)
            if quantized_at is not None:
                q_i, q_name = quantized_at
                raise ValueError(
                    f"Step {i} ({name!r}) comes after step {q_i} ({q_name!r}), a quantization method. "
                    "A quantized model can't be compressed further: put quantization last."
                )
            if get_method(name).kind == "quantization":
                quantized_at = (i, name)
            parsed.append((name, options))

        # 3. save the steps (a tuple, so they can't be changed later)
        self._steps: Tuple[Tuple[str, Dict[str, Any]], ...] = tuple(parsed)

    @property
    def steps(self) -> Tuple[Tuple[str, Dict[str, Any]], ...]:
        """The steps as (name, options) pairs. These are copies, so editing them is safe."""
        return tuple((name, dict(options)) for name, options in self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        return iter(self.steps)

    def __repr__(self) -> str:
        # e.g. Pipeline([('l1_pruning', {'ratio': 0.3}), 'dfq_quantization'])
        parts = [repr(name) if not options else repr((name, options)) for name, options in self._steps]
        return f"Pipeline([{', '.join(parts)}])"

    def __call__(self, model: nn.Module, example_input: Optional[Any] = None) -> nn.Module:
        """Run every step and return the compressed copy."""
        if not isinstance(model, nn.Module):
            raise TypeError(f"Pipeline expects an nn.Module, got {type(model).__name__}")

        total = len(self._steps)
        current = model
        for i, (name, options) in enumerate(self._steps, 1):
            # compress() copies the model, runs the method and checks the output shape
            try:
                current = compress(current, name, example_input=example_input, **options)
            except Exception as e:  # noqa: BLE001
                # same error, but the message now says which step failed
                wrapped = _with_step(e, i, total, name)
                if wrapped is e:
                    raise
                raise wrapped from e

            # safety net: a method gave back a quantized model but more steps are left
            if i < total and _is_quantized(current):
                raise CompressionError(
                    f"Step {i}/{total} ({name!r}) returned a quantized model, so the remaining steps can't run. "
                    "Put quantization last (or register the method with kind='quantization')."
                )
        return current


def _parse_step(step: Any, i: int) -> Tuple[str, Dict[str, Any]]:
    """Turn a step into (name, options) and check it."""
    # 1. read the step: "name" or ("name", {options})
    if isinstance(step, str):
        name, options = step, {}
    elif isinstance(step, (tuple, list)) and len(step) == 2 and isinstance(step[0], str):
        name, options = step
        if not isinstance(options, Mapping):
            raise TypeError(f"Step {i} ({name!r}): options must be a dict, got {type(options).__name__}")
        options = dict(options)  # our own copy, so later edits to your dict don't affect us
    else:
        raise TypeError(f"Step {i} must be a method name or a (name, options) pair, got {step!r}")

    # 2. the method must exist
    try:
        info = get_method(name)
    except UnknownMethodError as e:
        raise UnknownMethodError(f"Step {i}: {e}") from None

    # 3. example_input is given once when calling the pipeline, not per step
    if "example_input" in options:
        raise TypeError(
            f"Step {i} ({name!r}): pass example_input when calling the pipeline, not as a step option"
        )

    # 4. read the method's parameters (skip the first one, which is the model)
    params = inspect.signature(info.fn).parameters

    # 5. required options (no default value, e.g. loader) must be given
    missing = [
        n for n, p in list(params.items())[1:]
        if p.default is p.empty and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        and n != "example_input" and n not in options
    ]
    if missing:
        raise TypeError(
            f"Step {i} ({name!r}) needs option(s) {', '.join(missing)}, "
            f"e.g. ({name!r}, {{{missing[0]!r}: ...}})"
        )

    # 6. no unknown options (typos like "ratoi"); skipped if the method takes **kwargs
    if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        accepted = [n for n in list(params)[1:] if n != "example_input"]
        unknown = sorted(set(options) - set(accepted))
        if unknown:
            raise TypeError(
                f"Step {i} ({name!r}): unknown option(s) {', '.join(unknown)}. "
                f"It accepts: {', '.join(accepted) or '(nothing)'}"
            )
    return name, options


def _with_step(error: Exception, i: int, total: int, name: str) -> Exception:
    """Rebuild the error with the step number added to the message."""
    try:
        return type(error)(f"Pipeline step {i}/{total} ({name!r}) failed: {error}")
    except Exception:  # noqa: BLE001
        return error  # some error types can't be rebuilt; keep the original


def _is_quantized(model: nn.Module) -> bool:
    """True if any layer of the model is a quantized layer."""
    return any("quantized" in type(m).__module__ for m in model.modules())
