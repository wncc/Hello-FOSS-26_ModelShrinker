"""Knowledge distillation.

A teacher model teaches a smaller student to match its soft probability outputs
as well as the true labels.

By default, the student is a copy of the teacher with safely prunable layers
reduced by ``ratio``. If no layer can be pruned safely, the method raises a
clear error. You can provide your own smaller model with ``student=``.

The teacher is not modified. The trained student is returned. Optionally, a
fraction of the student's weakest weights is zeroed at the end (``sparsity``).
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from shrinker.benchmark import count_parameters
from shrinker.core import CompressionError
from shrinker.registry import register
from shrinker.utils import (
    get_module,
    keep_indices,
    prunable_layers,
    prune_channels,
    skipped_layers,
)

logger = logging.getLogger(__name__)


def _soft_target_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Calculate KL-divergence between teacher and student soft distributions."""
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    return (
        F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        * temperature**2
    )


def _zero_smallest_weights(model: nn.Module, sparsity: float) -> None:
    """Zero the smallest ``sparsity`` fraction of weights across the model."""
    if sparsity <= 0.0:
        return

    all_abs = []
    weight_refs = []

    for _, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:
            all_abs.append(param.detach().abs().flatten())
            weight_refs.append(param)

    if not all_abs:
        return

    flat = torch.cat(all_abs)
    n_zero = max(1, int(flat.numel() * sparsity))

    if n_zero >= flat.numel():
        return

    threshold = torch.topk(flat, n_zero, largest=False).values[-1]

    with torch.no_grad():
        for param in weight_refs:
            mask = param.abs() > threshold
            param.mul_(mask)


def _explain_skipped(skipped: Dict[str, str], limit: int = 5) -> str:
    """Format skipped-layer reasons while keeping the error message concise."""
    parts = [
        f"{name}: {reason}"
        for name, reason in list(skipped.items())[:limit]
    ]

    if len(skipped) > limit:
        parts.append(f"and {len(skipped) - limit} more")

    return "; ".join(parts) or "the model has no Conv2d/Linear layers"


def _build_default_student(model: nn.Module, ratio: float) -> nn.Module:
    """Copy the model and remove ``ratio`` of each safely prunable layer."""
    student = copy.deepcopy(model)

    # Find the safely prunable layers before changing the model.
    names = [name for name, _ in prunable_layers(student)]

    if not names:
        reasons = _explain_skipped(skipped_layers(student))
        raise CompressionError(
            "knowledge_distillation failed: no layer of this model can be "
            "pruned safely, so a smaller student cannot be built automatically "
            f"({reasons}). Pass your own smaller model with student=..."
        )

    logger.info(
        "knowledge_distillation: pruning %d layer(s) by %.0f%%: %s",
        len(names),
        ratio * 100,
        ", ".join(names),
    )

    # Keep the channels with the largest L1 norm, following l1_pruning.
    for name in names:
        # Fetch the current layer because previous pruning may replace layers.
        layer = get_module(student, name)
        scores = layer.weight.detach().abs().flatten(1).sum(dim=1)
        prune_channels(student, name, keep_indices(scores, ratio))

    if count_parameters(student) >= count_parameters(model):
        raise CompressionError(
            "knowledge_distillation failed: pruning did not make the student "
            f"smaller (layers tried: {', '.join(names)}). "
            "Pass your own smaller model with student=..."
        )

    return student


@register(
    kind="other",
    paper="Hinton et al. 2015, Distilling the Knowledge in a Neural Network",
)
def apply(
    model: nn.Module,
    loader: DataLoader,
    temperature: float = 4.0,
    alpha: float = 0.7,
    epochs: int = 5,
    lr: float = 1e-3,
    sparsity: float = 0.0,
    student: Optional[nn.Module] = None,
    ratio: float = 0.5,
) -> nn.Module:
    """Train a smaller student to mimic the teacher using knowledge distillation.

    ``loader`` must yield batches of ``(inputs, labels)``.

    If ``student`` is omitted, a copy of the teacher is structurally pruned.
    A supplied student is copied before training so the caller's model is not
    modified. ``ratio`` controls pruning of the default student and is ignored
    when a custom student is supplied. ``sparsity`` optionally zeros weak
    student weights after training.
    """
    if student is None:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        student = _build_default_student(model, ratio)
    else:
        if not isinstance(student, nn.Module):
            raise TypeError(
                f"student must be an nn.Module, got {type(student).__name__}"
            )

        n_student = count_parameters(student)
        n_model = count_parameters(model)

        if n_student >= n_model:
            raise ValueError(
                f"student must be smaller than the model: it has "
                f"{n_student:,} parameters, the model has {n_model:,}"
            )

        student = copy.deepcopy(student)

    teacher = model

    try:
        device = next(teacher.parameters()).device
    except StopIteration as exc:
        raise ValueError("teacher must have at least one parameter") from exc

    student.to(device)

    # Build the optimizer after creating/pruning the student.
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    was_training = teacher.training
    teacher.eval()
    student.train()

    try:
        # Check output shapes before training, including when epochs=0.
        try:
            check_x, _ = next(iter(loader))
        except StopIteration as exc:
            raise ValueError(
                "loader must contain a batch to validate the student's output shape"
            ) from exc

        check_x = check_x.to(device)
        student.eval()

        with torch.no_grad():
            teacher_check = teacher(check_x)
            student_check = student(check_x)

        if student_check.shape != teacher_check.shape:
            raise ValueError(
                f"The student's output shape {tuple(student_check.shape)} does not "
                f"match the teacher's {tuple(teacher_check.shape)}; both must "
                "give the same outputs."
            )

        student.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            seen = 0

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                with torch.no_grad():
                    teacher_logits = teacher(x)

                student_logits = student(x)

                if student_logits.shape != teacher_logits.shape:
                    raise ValueError(
                        f"The student's output shape {tuple(student_logits.shape)} "
                        f"does not match the teacher's "
                        f"{tuple(teacher_logits.shape)}; both must give the same "
                        "outputs."
                    )

                soft_loss = _soft_target_loss(
                    student_logits,
                    teacher_logits,
                    temperature,
                )
                hard_loss = F.cross_entropy(student_logits, y)
                loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.numel()
                seen += y.numel()

            logger.info(
                "distillation epoch %d/%d: loss %.4f",
                epoch,
                epochs,
                total_loss / max(seen, 1),
            )
    finally:
        teacher.train(was_training)

    student.train(was_training)
    _zero_smallest_weights(student, sparsity)

    return student
