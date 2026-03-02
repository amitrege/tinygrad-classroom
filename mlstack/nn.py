"""Minimal NN module and optimizers built on top of tiny autograd."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .autograd import Tensor


class Module:
    def parameters(self) -> list[Tensor]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        # Kaiming-like scaling works well for ReLU MLPs.
        scale = np.sqrt(2.0 / max(in_features, 1))
        w_data = rng.normal(0.0, scale, size=(in_features, out_features))
        b_data = np.zeros((1, out_features), dtype=np.float64)
        self.weight = Tensor(w_data, requires_grad=True, label="W")
        self.bias = Tensor(b_data, requires_grad=True, label="b")

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class SGD:
    def __init__(self, params: list[Tensor], lr: float = 0.1, momentum: float = 0.0) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.momentum > 0.0:
                self.velocities[idx] = self.momentum * self.velocities[idx] - self.lr * p.grad
                p.data += self.velocities[idx]
            else:
                p.data -= self.lr * p.grad


class Adam:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        self.t += 1
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad
            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[idx] / (1.0 - self.beta1**self.t)
            v_hat = self.v[idx] / (1.0 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class LossOutput:
    loss: Tensor
    probs: np.ndarray


def binary_cross_entropy_with_logits(logits: Tensor, targets: np.ndarray) -> LossOutput:
    """Stable BCE-with-logits with custom backward for clarity and stability.

    Forward: mean(log(1 + exp(z)) - y * z)
    Backward wrt z: (sigmoid(z) - y) / N
    """
    y = np.array(targets, dtype=np.float64).reshape(logits.data.shape)
    z = logits.data

    # numerically stable softplus
    loss_matrix = np.logaddexp(0.0, z) - y * z
    mean_loss = np.array(loss_matrix.mean(), dtype=np.float64)

    out = Tensor(
        mean_loss,
        requires_grad=logits.requires_grad,
        _prev=(logits,),
        _op="bce_with_logits",
    )

    z_clip = np.clip(z, -60.0, 60.0)
    probs = 1.0 / (1.0 + np.exp(-z_clip))

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return
        grad_logits = (probs - y) / y.size
        logits.grad += grad_logits * out.grad

    out._backward = _backward
    return LossOutput(loss=out, probs=probs)


def binary_accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    preds = (logits > 0.0).astype(np.float64)
    return float((preds == y_true).mean())
