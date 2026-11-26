import pytest
import torch

from shrinker.methods._calibration import ridge


def test_ridge_recovers_a_linear_map():
    X = torch.randn(200, 5)
    W, b = torch.randn(3, 5), torch.randn(3)
    fit = ridge(X, X @ W.T + b, lam=0.0)
    assert torch.allclose(fit[0].float(), W, atol=1e-4) and torch.allclose(fit[1].float(), b, atol=1e-4)


@pytest.mark.parametrize("rows", [1, 10, 23])
def test_ridge_needs_enough_rows(rows):
    assert ridge(torch.randn(rows, 5), torch.randn(rows, 3)) is None
