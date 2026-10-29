"""Quantization-aware training that produces a CPU INT8 model."""

from __future__ import annotations

import logging
from typing import Any

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


def _to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, tuple):
        return tuple(value.to(device) for value in inputs)
    return inputs.to(device)


def _freeze_qat_statistics(module: nn.Module) -> None:
    """Stop observer and fused-BatchNorm updates once ranges have stabilized."""
    from torch.ao.quantization import disable_observer

    module.apply(disable_observer)
    try:
        from torch.ao.nn.intrinsic.qat import freeze_bn_stats
    except ImportError:
        return
    module.apply(freeze_bn_stats)


@register(kind="quantization")
def apply(
    model: nn.Module,
    loader,
    example_input: Any = None,
    epochs: int = 5,
    lr: float = 1e-4,
    backend: str = "auto",
    device: str = "cpu",
    observer_freeze_fraction: float = 0.8,
) -> nn.Module:
    """Return an INT8 CPU model after quantization-aware fine-tuning.

    The method prepares the copied model for FX graph-mode quantization-aware
    training, fine-tunes it with cross-entropy, and converts supported graph
    operations to backend-specific integer operators.  Fake quantization in
    training simulates signed per-channel INT8 weights and asymmetric INT8
    activations while preserving FP32 weights for optimization.

    ``loader`` yields ``(inputs, class_labels)`` batches.  ``example_input``
    supplies the input shape used to trace the graph; the first loader batch is
    used when it is omitted.  ``backend='auto'`` selects an available x86 CPU
    backend on desktop machines and QNNPACK on ARM.  Operations unsupported by
    the selected backend remain FP32 with quantization boundaries around them.
    """
    if loader is None:
        raise ValueError("qat_quantization requires a labeled DataLoader via loader=")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not 0.0 <= observer_freeze_fraction <= 1.0:
        raise ValueError("observer_freeze_fraction must be in [0, 1]")
    if device != "cpu":
        raise ValueError("qat_quantization currently supports device='cpu' only")

    try:
        first_batch = next(iter(loader))
        inputs, _ = first_batch
    except Exception as exc:  # noqa: BLE001 - malformed loaders are not quantizable
        logger.info("Skipping QAT quantization: loader did not yield (inputs, labels): %s", exc)
        return model

    if example_input is None:
        example_input = inputs
    example_args = _as_args(example_input)
    if not all(isinstance(value, torch.Tensor) for value in example_args):
        logger.info("Skipping QAT quantization: example_input must be a Tensor or tuple of Tensors")
        return model

    selected_backend = _select_backend(backend)
    try:
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
    except ImportError as exc:
        raise RuntimeError(
            "qat_quantization requires PyTorch FX graph-mode QAT (torch.ao.quantization)"
        ) from exc

    try:
        qconfig_mapping = get_default_qat_qconfig_mapping(selected_backend)
        with _quantized_engine(selected_backend):
            prepared = prepare_qat_fx(model.cpu().train(), qconfig_mapping, example_args)
    except Exception as exc:  # noqa: BLE001 - a model may not be FX traceable
        logger.info("Skipping QAT quantization: could not prepare this model for FX QAT: %s", exc)
        return model

    try:
        optimizer = torch.optim.Adam(prepared.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        try:
            batches_per_epoch = len(loader)
        except TypeError:
            batches_per_epoch = 0
        freeze_after = int(epochs * batches_per_epoch * observer_freeze_fraction)
        step = 0
        frozen = False

        prepared.train()
        for _ in range(epochs):
            for batch_inputs, targets in loader:
                if not frozen and step >= freeze_after:
                    _freeze_qat_statistics(prepared)
                    frozen = True
                batch_inputs = _to_device(batch_inputs, torch.device("cpu"))
                targets = targets.to("cpu")
                optimizer.zero_grad()
                output = prepared(*_as_args(batch_inputs))
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                step += 1
        if not frozen:
            _freeze_qat_statistics(prepared)
    except Exception as exc:  # noqa: BLE001 - unsupported training output/data is left unchanged
        logger.info("Skipping QAT quantization: training failed for this model or loader: %s", exc)
        return model

    try:
        with _quantized_engine(selected_backend):
            result = convert_fx(prepared.eval())
    except Exception as exc:  # noqa: BLE001 - leave models unchanged if a backend cannot lower them
        logger.info("Skipping QAT quantization: backend conversion failed: %s", exc)
        return model

    _make_views_layout_safe(result)
    quantized = sum("quantized" in type(module).__module__ for module in result.modules())
    if quantized:
        logger.info("QAT converted %d quantized modules using the %s backend", quantized, selected_backend)
    else:
        logger.info("QAT conversion retained this model in FP32 because no supported operators were found")
    return result
