"""Per-layer table of a model's Conv2d/Linear layers: shape, parameters, and whether
each one can be pruned (with the reason when it can't)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .dependency import analyze


def summary(model: nn.Module, example_input: Any = None) -> List[Dict[str, Any]]:
    """Print a table with one row per ``nn.Conv2d`` / ``nn.Linear`` layer and return the rows.

    Each row is a dict with ``name``, ``type``, ``shape`` (of the weight), ``params``,
    ``prunable``, ``reason`` (why it can't be pruned, else ``None``) and ``output_shape``
    (only filled in when ``example_input`` is given and the layer runs in ``forward()``).
    """
    prunable, skipped = analyze(model)
    layers = [(n, m) for n, m in model.named_modules() if type(m) in (nn.Conv2d, nn.Linear)]
    output_shapes = _output_shapes(model, layers, example_input) if example_input is not None else {}

    rows = []
    for name, layer in layers:
        rows.append({
            "name": name,
            "type": type(layer).__name__,
            "shape": tuple(layer.weight.shape),
            "params": sum(p.numel() for p in layer.parameters()),
            "prunable": name in prunable,
            "reason": None if name in prunable else skipped.get(name),
            "output_shape": output_shapes.get(name),
        })
    print(format_summary(rows))
    return rows


def _output_shapes(model: nn.Module, layers, example_input: Any) -> Dict[str, Tuple[int, ...]]:
    """Run one forward pass in eval mode and record each layer's output shape."""
    shapes: Dict[str, Tuple[int, ...]] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            shapes[name] = tuple(output.shape)
        return hook

    handles = [layer.register_forward_hook(make_hook(name)) for name, layer in layers]
    was_training = model.training
    model.eval()  # keeps BatchNorm running stats untouched
    try:
        with torch.no_grad():
            model(*example_input) if isinstance(example_input, tuple) else model(example_input)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return shapes


def format_summary(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "no Conv2d/Linear layers found"

    def shape(s: Optional[Tuple[int, ...]]) -> str:
        return "-" if s is None else "(" + ", ".join(str(d) for d in s) + ")"

    with_output = any(r["output_shape"] is not None for r in rows)
    header = ["layer", "type", "shape"] + (["output"] if with_output else []) + ["params", "prunable"]
    params_col = header.index("params")
    table = []
    for r in rows:
        cells = [r["name"], r["type"], shape(r["shape"])]
        if with_output:
            cells.append(shape(r["output_shape"]))
        cells += [f"{r['params']:,}", "yes" if r["prunable"] else f"no: {r['reason']}"]
        table.append(cells)

    n_prunable = sum(r["prunable"] for r in rows)
    total = [""] * len(header)
    total[0] = "total"
    total[params_col] = f"{sum(r['params'] for r in rows):,}"
    total[-1] = f"{n_prunable} of {len(rows)} layers prunable"

    # the last column is free text, so it is not padded and doesn't widen the rules
    widths = [max(len(r[i]) for r in table + [header, total]) for i in range(len(header) - 1)]

    def fmt(cells: List[str]) -> str:
        padded = [c.rjust(w) if i == params_col else c.ljust(w) for i, (c, w) in enumerate(zip(cells, widths))]
        return "  ".join(padded + [cells[-1]]).rstrip()

    line = "-" * max(len(fmt(r)) for r in table + [header, total])
    return "\n".join([fmt(header), line] + [fmt(r) for r in table] + [line, fmt(total)])
