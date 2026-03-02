"""Finite-difference gradient checks for teaching and debugging."""

from __future__ import annotations

import numpy as np

from .autograd import Tensor


def _mse_from_numpy(x: np.ndarray, w: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    pred = x @ w + b
    return float(np.mean((pred - y) ** 2))


def _finite_diff_w(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    approx = np.zeros_like(w)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w_pos = w.copy()
            w_neg = w.copy()
            w_pos[i, j] += eps
            w_neg[i, j] -= eps
            l_pos = _mse_from_numpy(x, w_pos, b, y)
            l_neg = _mse_from_numpy(x, w_neg, b, y)
            approx[i, j] = (l_pos - l_neg) / (2.0 * eps)
    return approx


def check_linear_layer_grad(
    seed: int = 0,
    introduce_bug: bool = False,
) -> dict[str, float | bool]:
    """Compare autograd dL/dW against finite differences.

    If `introduce_bug=True`, we intentionally use a wrong analytic gradient
    (missing factor 2 for MSE) to demonstrate check failures.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(6, 3))
    w = rng.normal(0.0, 1.0, size=(3, 2))
    b = rng.normal(0.0, 1.0, size=(1, 2))
    y = rng.normal(0.0, 1.0, size=(6, 2))

    xt = Tensor(x, requires_grad=False)
    wt = Tensor(w.copy(), requires_grad=True)
    bt = Tensor(b.copy(), requires_grad=True)

    pred = xt @ wt + bt
    loss = ((pred - y) ** 2).mean()
    loss.backward()

    grad_auto = wt.grad.copy()

    if introduce_bug:
        # Deliberate bug: MSE derivative should include factor 2.
        grad_analytic = (x.T @ (pred.data - y)) / y.size
    else:
        grad_analytic = grad_auto

    grad_fd = _finite_diff_w(x, w, b, y)

    num = np.linalg.norm(grad_analytic - grad_fd)
    den = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_fd) + 1e-12
    rel_error = float(num / den)

    return {
        "relative_error": rel_error,
        "passed": rel_error < 1e-6,
    }
