from types import SimpleNamespace

import torch

import shrinker.methods._quantization_common as qc


def test_engine_is_restored_after_use():
    before = torch.backends.quantized.engine
    with qc._quantized_engine(torch.backends.quantized.supported_engines[0]):
        pass
    assert torch.backends.quantized.engine == before


def test_restore_is_skipped_when_the_previous_engine_is_unsupported(monkeypatch):
    """Some platforms default to 'none', which PyTorch refuses to assign back (issue #14)."""
    fake = SimpleNamespace(engine="none", supported_engines=["qnnpack"])
    monkeypatch.setattr(qc.torch, "backends", SimpleNamespace(quantized=fake))

    with qc._quantized_engine("qnnpack"):
        assert fake.engine == "qnnpack"

    assert fake.engine == "qnnpack"  # left alone instead of raising on 'none'
