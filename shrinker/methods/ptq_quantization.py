"""Static INT8 post-training quantization with a small calibration set.

Observers record real activation ranges while a handful of loader batches run
through the model, after which ``torch.fx`` converts supported operations to
CPU INT8 kernels.
Layers the backend cannot convert stay FP32. If tracing, preparing, calibrating or
converting fails, the model is returned unchanged and the reason is logged.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn

from shrinker.methods._quantization_common import (
    _as_args,
    _make_views_layout_safe,
    _quantized_engine,
    _select_backend,
)
from shrinker.registry import register

logger = logging.getLogger(__name__)


def _calibration_inputs(batch: Any) -> Optional[torch.Tensor]:
    """Return the input tensor of a calibration batch, or ``None`` if unusable.

    Labels are not needed for calibration, so an iterable of bare Tensors works
    as well as the ``(inputs, labels)`` batches the other methods expect.
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
        return batch[0]
    return None


@register(kind="quantization")
def apply(
    model: nn.Module,
    loader,
    example_input: Any = None,
    backend: str = "auto",
    device: str = "cpu",
    max_batches: int = 32,
) -> nn.Module:
    """Return an INT8 CPU model calibrated on a small data sample.

    The method traces the model with ``torch.fx``, inserts observers, runs at
    most ``max_batches`` batches from ``loader`` through it so they record real
    activation ranges, and converts supported graph operations to
    backend-specific integer kernels.  Observers only watch forward passes, so
    no training data labels or optimizer are involved.

    ``loader`` yields Tensors or ``(inputs, labels)`` batches.  ``example_input``
    supplies the input shape used to trace the graph; the first loader batch is
    used when it is omitted.  ``backend='auto'`` selects an available x86 CPU
    backend on desktop machines and QNNPACK on ARM.  Operations unsupported by
    the selected backend remain FP32 with quantization boundaries around them.
    """
    if loader is None:
        raise ValueError("ptq_quantization requires a calibration DataLoader via loader=")
    if not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError(f"max_batches must be a positive integer, got {max_batches!r}")
    if device != "cpu":
        raise ValueError("ptq_quantization currently supports device='cpu' only")

    try:
        first_inputs = _calibration_inputs(next(iter(loader)))
    except Exception as exc:  # noqa: BLE001 - malformed loaders are not quantizable
        logger.info("Skipping static PTQ: loader produced no usable batch: %s", exc)
        return model
    if first_inputs is None:
        logger.info("Skipping static PTQ: loader must yield Tensors or (inputs, labels) batches")
        return model

    if example_input is None:
        example_input = first_inputs
    example_args = _as_args(example_input)
    if not all(isinstance(value, torch.Tensor) for value in example_args):
        logger.info("Skipping static PTQ: example_input must be a Tensor or tuple of Tensors")
        return model

    selected_backend = _select_backend(backend)
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    except ImportError as exc:
        raise RuntimeError(
            "ptq_quantization requires PyTorch FX graph-mode quantization (torch.ao.quantization)"
        ) from exc

    try:
        qconfig_mapping = get_default_qconfig_mapping(selected_backend)
        with _quantized_engine(selected_backend):
            prepared = prepare_fx(model.cpu().eval(), qconfig_mapping, example_args)
    except Exception as exc:  # noqa: BLE001 - a model may not be FX traceable
        logger.info("Skipping static PTQ: could not trace and prepare this model for FX quantization: %s", exc)
        return model

    try:
        calibrated = 0
        with torch.no_grad():
            for batch in loader:
                calibration_inputs = _calibration_inputs(batch)
                if calibration_inputs is None:
                    raise ValueError("calibration batches must be Tensors or (inputs, labels) tuples")
                prepared(*_as_args(calibration_inputs))
                calibrated += 1
                if calibrated >= max_batches:
                    break
    except Exception as exc:  # noqa: BLE001 - bad calibration data is not quantizable
        logger.info("Skipping static PTQ: calibration failed on this loader: %s", exc)
        return model

    try:
        with _quantized_engine(selected_backend):
            result = convert_fx(prepared.eval())
    except Exception as exc:  # noqa: BLE001 - leave models unchanged if a backend cannot lower them
        logger.info("Skipping static PTQ: backend conversion failed: %s", exc)
        return model

    _make_views_layout_safe(result)
    quantized = sum("quantized" in type(module).__module__ for module in result.modules())
    if quantized:
        logger.info(
            "Static PTQ converted %d quantized modules from %d calibration batches using the %s backend",
            quantized,
            calibrated,
            selected_backend,
        )
    else:
        logger.info("Static PTQ conversion retained this model in FP32 because no supported operators were found")
    return result
