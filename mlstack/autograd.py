"""Tiny reverse-mode autodiff over NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayLike = np.ndarray | float | int


def _to_array(x: ArrayLike) -> np.ndarray:
    return np.array(x, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum gradient axes that were broadcast in forward pass."""
    g = grad
    while g.ndim > len(shape):
        g = g.sum(axis=0)

    for axis, dim in enumerate(shape):
        if dim == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)

    return g.reshape(shape)


@dataclass
class Tensor:
    data: np.ndarray
    requires_grad: bool = False
    _prev: tuple["Tensor", ...] = ()
    _op: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        self.data = _to_array(self.data)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    @staticmethod
    def ensure(other: ArrayLike | "Tensor") -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False)

    def __add__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = Tensor.ensure(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self + other

    def __sub__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self + (-Tensor.ensure(other))

    def __rsub__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return Tensor.ensure(other) - self

    def __neg__(self) -> "Tensor":
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="neg",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = Tensor.ensure(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = Tensor.ensure(other)
        return self * other.pow(-1.0)

    def __rtruediv__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return Tensor.ensure(other) / self

    def pow(self, exponent: float) -> "Tensor":
        out = Tensor(
            np.power(self.data, exponent),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op=f"pow({exponent})",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad * exponent * np.power(self.data, exponent - 1.0)

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> "Tensor":
        return self.pow(exponent)

    def matmul(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = Tensor.ensure(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self.matmul(other)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad

            if axis is None:
                grad = np.broadcast_to(grad, self.data.shape)
            else:
                axes = (axis,) if isinstance(axis, int) else axis
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)
                grad = np.broadcast_to(grad, self.data.shape)

            self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denom = self.data.size
        elif isinstance(axis, int):
            denom = self.data.shape[axis]
        else:
            denom = int(np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        out = Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="transpose",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(0.0, self.data),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad * (self.data > 0.0)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="tanh",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad * (1.0 - t * t)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        e = np.exp(np.clip(self.data, -60.0, 60.0))
        out = Tensor(e, requires_grad=self.requires_grad, _prev=(self,), _op="exp")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad * e

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        safe = np.clip(self.data, 1e-12, None)
        out = Tensor(np.log(safe), requires_grad=self.requires_grad, _prev=(self,), _op="log")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad / safe

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        z = np.clip(self.data, -60.0, 60.0)
        s = 1.0 / (1.0 + np.exp(-z))
        out = Tensor(s, requires_grad=self.requires_grad, _prev=(self,), _op="sigmoid")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad * s * (1.0 - s)

        out._backward = _backward
        return out

    def backward(self, grad: ArrayLike | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Called backward on a tensor that does not require gradients.")

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("Grad must be provided for non-scalar outputs.")
            grad_arr = np.ones_like(self.data)
        else:
            grad_arr = _to_array(grad)

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad_arr

        for node in reversed(topo):
            node._backward()


def as_tensor(x: ArrayLike | Tensor, requires_grad: bool = False) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=requires_grad)
