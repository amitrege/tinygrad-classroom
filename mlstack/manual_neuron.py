"""Single-neuron manual forward/backward training in NumPy."""

from __future__ import annotations

import numpy as np


_EPS = 1e-8


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def predict_proba(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(x @ w + b)


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob, _EPS, 1.0 - _EPS)
    return float(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean())


def manual_gradients(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
) -> tuple[np.ndarray, float, float]:
    """Return (dW, dB, loss) for logistic regression."""
    p = predict_proba(x, w, b)
    diff = p - y

    grad_w = (x.T @ diff) / x.shape[0]
    grad_b = float(diff.mean())
    loss = binary_cross_entropy(y, p)
    return grad_w, grad_b, loss


def train_single_neuron(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.2,
    steps: int = 200,
    seed: int = 0,
    checkpoint_every: int = 20,
) -> dict[str, object]:
    """Train logistic regression and record diagnostics/checkpoints."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.6, size=(x.shape[1], 1))
    b = float(rng.normal(0.0, 0.1))

    losses: list[float] = []
    grad_norms: list[float] = []
    checkpoints: list[dict[str, object]] = []

    for step in range(steps):
        grad_w, grad_b, loss = manual_gradients(x, y, w, b)
        losses.append(loss)
        grad_norms.append(float(np.linalg.norm(grad_w)))

        if step % checkpoint_every == 0 or step == steps - 1:
            checkpoints.append(
                {
                    "step": step,
                    "w": w.copy(),
                    "b": b,
                    "loss": loss,
                }
            )

        w = w - lr * grad_w
        b = b - lr * grad_b

    probs = predict_proba(x, w, b)
    acc = float(((probs > 0.5).astype(np.float64) == y).mean())

    return {
        "w": w,
        "b": b,
        "losses": np.array(losses, dtype=np.float64),
        "grad_norms": np.array(grad_norms, dtype=np.float64),
        "checkpoints": checkpoints,
        "train_acc": acc,
    }
