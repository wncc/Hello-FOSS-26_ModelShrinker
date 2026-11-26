from __future__ import annotations

import contextlib
import io
import statistics
import time
from typing import Any, Dict, Iterator, Optional

import torch
from torch import nn

from .attention import PrunedMultiheadAttention
from .utils import unfused_transformer_layers


def _is_quantized(m: nn.Module, kind: str) -> bool:
    """True for the converted INT8 ``Conv2d`` / ``Linear`` kernels, including fused ConvReLU2d / LinearReLU."""
    try:
        import torch.ao.nn.quantized as nnq
    except ImportError:  # the torch.ao quantization stack is not available
        return False
    return isinstance(m, getattr(nnq, kind))


def _packed_tensors(model: nn.Module) -> Iterator[torch.Tensor]:
    """Weights and biases of quantized layers. They are packed INT8 data, not ``nn.Parameter``s,
    so ``model.parameters()`` doesn't see them."""
    for m in model.modules():
        if _is_quantized(m, "Conv2d") or _is_quantized(m, "Linear"):
            yield m.weight().dequantize()
            bias = m.bias()
            # packing fills in an all-zero bias for a layer that had none; that is not a parameter
            if bias is not None and bool(torch.count_nonzero(bias)):
                yield bias


def _tensors(model: nn.Module) -> Iterator[torch.Tensor]:
    yield from model.parameters()
    yield from _packed_tensors(model)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in _tensors(model))


def count_nonzero_parameters(model: nn.Module) -> int:
    """Parameter count minus zeroed entries of weight tensors (biases/BatchNorm always count)."""
    return sum(int(torch.count_nonzero(p)) if p.dim() > 1 else p.numel() for p in _tensors(model))


def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@contextlib.contextmanager
def _eval_on(model: nn.Module, device: str):
    was_training = model.training
    first = next(model.parameters(), None)
    old_device = first.device if first is not None else torch.device("cpu")
    moved = torch.device(device) != old_device
    if moved:
        model.to(device)
    model.eval()
    try:
        yield
    finally:
        if moved:
            model.to(old_device)
        model.train(was_training)


def _run(model: nn.Module, x: Any):
    return model(*x) if isinstance(x, tuple) else model(x)


def _to(x: Any, device: str):
    if isinstance(x, tuple):
        return tuple(t.to(device) for t in x)
    return x.to(device)


def measure_latency(
    model: nn.Module, example_input: Any, warmup: int = 10, runs: int = 50, device: str = "cpu"
) -> float:
    """Median wall-clock time of one forward pass, in milliseconds."""
    x = _to(example_input, device)
    cuda = torch.device(device).type == "cuda"
    with _eval_on(model, device), torch.no_grad():
        for _ in range(warmup):
            _run(model, x)
        times = []
        for _ in range(runs):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _run(model, x)
            if cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def count_flops(model: nn.Module, example_input: Any) -> Optional[int]:
    """FLOPs of the Conv2d, Linear, LSTM/GRU and attention layers (float or quantized) for one forward pass.

    Returns ``None`` when no such layer ran, so an unmeasured model is not reported as 0 FLOPs.
    """
    total = [0]
    counted = [False]

    def conv_hook(m, inp, out):
        counted[0] = True
        total[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        counted[0] = True
        total[0] += out.numel() * m.in_features

    def rnn_hook(m, inp, out):
        # Every time step multiplies each gate by [layer input, previous hidden state].
        counted[0] = True
        x = inp[0]
        steps = x.data.shape[0] if isinstance(x, nn.utils.rnn.PackedSequence) else x.shape[0] * x.shape[1]
        gates = 4 if isinstance(m, nn.LSTM) else 3
        directions = 2 if m.bidirectional else 1
        out_size = getattr(m, "proj_size", 0) or m.hidden_size
        for layer in range(m.num_layers):
            layer_in = m.input_size if layer == 0 else out_size * directions
            total[0] += directions * steps * gates * m.hidden_size * (layer_in + out_size)
            if out_size != m.hidden_size:
                total[0] += directions * steps * m.hidden_size * out_size

    def attention_hook(m, args, kwargs, out):
        counted[0] = True
        query = args[0] if args else kwargs["query"]
        key = args[1] if len(args) > 1 else kwargs.get("key", query)
        B, L = (query.shape[0], query.shape[1]) if m.batch_first else (query.shape[1], query.shape[0])
        S = key.shape[1] if m.batch_first else key.shape[0]
        width = m.num_heads * m.head_dim
        if isinstance(m, nn.MultiheadAttention):  # its projections are not nn.Linear layers
            total[0] += B * (2 * L + 2 * S) * m.embed_dim * m.embed_dim
        total[0] += 2 * B * L * S * width  # scores and weighted sum

    handles = []
    for m in model.modules():
        if type(m) is nn.Conv2d or _is_quantized(m, "Conv2d"):
            handles.append(m.register_forward_hook(conv_hook))
        elif type(m) is nn.Linear or _is_quantized(m, "Linear"):
            handles.append(m.register_forward_hook(linear_hook))
        elif type(m) in (nn.LSTM, nn.GRU):
            handles.append(m.register_forward_hook(rnn_hook))
        elif type(m) in (nn.MultiheadAttention, PrunedMultiheadAttention):
            handles.append(m.register_forward_hook(attention_hook, with_kwargs=True))
    try:
        with _eval_on(model, "cpu"), unfused_transformer_layers(_to(example_input, "cpu")) as x:
            _run(model, x)
    finally:
        for h in handles:
            h.remove()
    return 2 * total[0] if counted[0] else None


def evaluate(model: nn.Module, loader, device: str = "cpu") -> float:
    """Top-1 accuracy (0..1)."""
    correct = total = 0
    with _eval_on(model, device), torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1)
            correct += (pred == y.to(device)).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def measure(
    model: nn.Module, example_input=None, val_loader=None, device: str = "cpu", warmup: int = 10, runs: int = 50
) -> Dict[str, Optional[float]]:
    try:
        flops = count_flops(model, example_input)
    except Exception:  # noqa: BLE001
        flops = None
    return {
        "params": count_parameters(model),
        "nonzero_params": count_nonzero_parameters(model),
        "size_mb": model_size_mb(model),
        "latency_ms": measure_latency(model, example_input, warmup, runs, device) if example_input is not None else None,
        "flops": flops,
        "accuracy": evaluate(model, val_loader, device) if val_loader is not None else None,
    }


def compare(
    original: nn.Module,
    compressed: nn.Module,
    example_input: Any,
    val_loader=None,
    device: str = "cpu",
    warmup: int = 10,
    runs: int = 50,
    verbose: bool = True,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Measure both models, print a table, and return ``{"original": {...}, "compressed": {...}}``."""
    report = {
        "original": measure(original, example_input, val_loader, device, warmup, runs),
        "compressed": measure(compressed, example_input, val_loader, device, warmup, runs),
    }
    if verbose:
        print(format_report(report))
    return report


def format_report(report: Dict[str, Dict[str, Optional[float]]]) -> str:
    a, b = report["original"], report["compressed"]

    def pct(x, y):
        return f"{(y - x) / x * 100:+.1f}%" if x else "-"

    rows = [("params", f"{a['params']:,}", f"{b['params']:,}", pct(a["params"], b["params"]))]
    if a["nonzero_params"] < a["params"] or b["nonzero_params"] < b["params"]:
        rows.append(("non-zero params", f"{a['nonzero_params']:,}", f"{b['nonzero_params']:,}",
                     pct(a["nonzero_params"], b["nonzero_params"])))
    rows.append(("size (MB)", f"{a['size_mb']:.3f}", f"{b['size_mb']:.3f}", pct(a["size_mb"], b["size_mb"])))
    if a["latency_ms"] is not None and b["latency_ms"] is not None:
        speedup = a["latency_ms"] / b["latency_ms"] if b["latency_ms"] else float("inf")
        rows.append(("latency (ms)", f"{a['latency_ms']:.3f}", f"{b['latency_ms']:.3f}", f"{speedup:.2f}x speed"))
    if a["flops"] is not None and b["flops"] is not None:
        rows.append(("MFLOPs", f"{a['flops'] / 1e6:.2f}", f"{b['flops'] / 1e6:.2f}", pct(a["flops"], b["flops"])))
    if a["accuracy"] is not None:
        rows.append(
            (
                "accuracy",
                f"{a['accuracy'] * 100:.2f}%",
                f"{b['accuracy'] * 100:.2f}%",
                f"{(b['accuracy'] - a['accuracy']) * 100:+.2f} pts",
            )
        )
    header = ("", "original", "compressed", "change")
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(4)]
    fmt = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))  # noqa: E731
    line = "-" * len(fmt(header))
    return "\n".join([fmt(header), line] + [fmt(r) for r in rows])
