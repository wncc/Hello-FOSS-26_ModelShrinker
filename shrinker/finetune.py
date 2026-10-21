from __future__ import annotations

import torch
from torch import nn
from torch.optim import lr_scheduler as lrs

# Schedulers that must be stepped once per batch. finetune() only steps once per epoch, rejects others
_PER_BATCH = (lrs.OneCycleLR, lrs.CyclicLR)
_SCHEDULER_BASE = getattr(lrs, "LRScheduler", None) or lrs._LRScheduler
 
 
def _build_scheduler(factory, opt):
    """Call ``factory(opt)``, validate the result, return ``(scheduler, needs_metric)``."""
    if not callable(factory):
        raise TypeError(
            "scheduler must be a callable that takes the optimizer and returns an LR "
            "scheduler, e.g. `lambda opt: StepLR(opt, step_size=3)`; got "
            f"{type(factory).__name__}. Pass a factory, not a scheduler instance."
        )
    sched = factory(opt)
    if not isinstance(sched, (_SCHEDULER_BASE, lrs.ReduceLROnPlateau)):
        raise TypeError(
            "scheduler factory must return a torch.optim.lr_scheduler object, "
            f"got {type(sched).__name__}"
        )
    if isinstance(sched, _PER_BATCH):
        raise ValueError(
            f"{type(sched).__name__} must be stepped per batch, which finetune() does not "
            "support. Use a per-epoch scheduler (StepLR, CosineAnnealingLR, ...) or "
            "ReduceLROnPlateau."
        )
    if getattr(sched, "optimizer", None) is not opt:
        raise ValueError(
            "scheduler factory must build the scheduler on the optimizer it receives"
        )
    return sched, isinstance(sched, lrs.ReduceLROnPlateau)


def finetune(
    model: nn.Module,
    train_loader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
    optimizer: str = "adam",
    loss_fn= None,
    scheduler=None,
    weight_decay: float=0.0,
    verbose: bool = True,
) -> nn.Module:
    """Train ``model`` in place with cross-entropy to recover accuracy after compression."""
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")
    if loss_fn is None:
        loss_fn=torch.nn.CrossEntropyLoss() #keeping the default from the previous implementation
    elif not callable(loss_fn):
        raise TypeError(f"loss_fn must be a callable (output, target) -> loss, got {type(loss_fn).__name__}")
    if scheduler is not None:
        sched, needs_metric= _build_scheduler(scheduler,opt)
    was_training = model.training
    model.to(device).train()
    if isinstance(loss_fn, nn.Module):
        loss_fn.to(device) #For losses like cross entropy with weights would give errors
    for epoch in range(1, epochs + 1):
        total_loss = correct = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError(
                    "loss_fn must return a scalar tensor; got "
                    f"{type(loss).__name__} with shape {tuple(getattr(loss, 'shape', ()))}. "
                    "If using reduction='none', reduce it first."
                )
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            seen += y.numel()
        if scheduler is not None:
            if needs_metric:
                sched.step(total_loss / max(seen, 1))
            else:
                sched.step()

        if verbose:
            print(f"finetune epoch {epoch}/{epochs}: loss {total_loss / max(seen, 1):.4f}, "
                  f"train acc {correct / max(seen, 1) * 100:.2f}%")
    model.train(was_training)
    return model
