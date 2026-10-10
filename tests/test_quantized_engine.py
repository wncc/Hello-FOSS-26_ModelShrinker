import pytest
import torch

from shrinker.methods.qat_quantization import _quantized_engine as qat_engine
from shrinker.methods.dfq_quantization import _quantized_engine as dfq_engine


@pytest.mark.parametrize("engine_ctx", [qat_engine, dfq_engine])
def test_engine_restore_does_not_raise(engine_ctx):
    supported = torch.backends.quantized.supported_engines
    backend = next(e for e in supported if e != "none")
    with engine_ctx(backend):
        pass
