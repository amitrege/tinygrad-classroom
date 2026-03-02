"""Training utilities for tiny NumPy neural nets."""

from __future__ import annotations

import numpy as np

from .autograd import Tensor
from .nn import Adam, Linear, ReLU, SGD, Sequential, binary_accuracy_from_logits, binary_cross_entropy_with_logits


def build_mlp(in_dim: int = 2, hidden_dim: int = 16, seed: int = 0) -> Sequential:
    return Sequential(
        Linear(in_dim, hidden_dim, seed=seed),
        ReLU(),
        Linear(hidden_dim, 1, seed=seed + 1),
    )


def _grad_norm(model: Sequential) -> float:
    norms = []
    for p in model.parameters():
        if p.grad is None:
            continue
        norms.append(np.linalg.norm(p.grad))
    if not norms:
        return 0.0
    return float(np.sqrt(np.sum(np.square(norms))))


def train_binary_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 16,
    lr: float = 0.05,
    steps: int = 350,
    optimizer_name: str = "adam",
    batch_size: int = 64,
    seed: int = 0,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    model = build_mlp(in_dim=x.shape[1], hidden_dim=hidden_dim, seed=seed)

    params = model.parameters()
    if optimizer_name.lower() == "sgd":
        optimizer = SGD(params, lr=lr, momentum=0.0)
    elif optimizer_name.lower() == "momentum":
        optimizer = SGD(params, lr=lr, momentum=0.9)
    else:
        optimizer = Adam(params, lr=lr)

    losses = []
    accuracies = []
    grad_norms = []

    n = x.shape[0]
    for _step in range(steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        xb = x[idx]
        yb = y[idx]

        logits = model(Tensor(xb, requires_grad=False))
        loss_out = binary_cross_entropy_with_logits(logits, yb)
        loss = loss_out.loss

        optimizer.zero_grad()
        loss.backward()
        grad_norms.append(_grad_norm(model))
        optimizer.step()

        with_logits = model(Tensor(x, requires_grad=False)).data
        train_acc = binary_accuracy_from_logits(with_logits, y)

        losses.append(float(loss.data))
        accuracies.append(train_acc)

    return {
        "model": model,
        "losses": np.array(losses, dtype=np.float64),
        "accuracies": np.array(accuracies, dtype=np.float64),
        "grad_norms": np.array(grad_norms, dtype=np.float64),
    }


def predict_logits(model: Sequential, x: np.ndarray) -> np.ndarray:
    return model(Tensor(x, requires_grad=False)).data
