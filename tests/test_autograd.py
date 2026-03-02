"""Unit tests for tiny autograd engine."""

from __future__ import annotations

import unittest

import numpy as np

from mlstack.autograd import Tensor


class TestAutograd(unittest.TestCase):
    def test_scalar_gradients(self) -> None:
        x = Tensor(np.array(2.0), requires_grad=True)
        y = Tensor(np.array(3.0), requires_grad=True)
        z = x * y + x
        z.backward()

        self.assertAlmostEqual(float(x.grad), 4.0, places=7)
        self.assertAlmostEqual(float(y.grad), 2.0, places=7)

    def test_broadcast_add_backward(self) -> None:
        x = Tensor(np.ones((3, 4)), requires_grad=True)
        b = Tensor(np.ones((1, 4)), requires_grad=True)
        y = (x + b).sum()
        y.backward()

        np.testing.assert_allclose(x.grad, np.ones((3, 4)))
        np.testing.assert_allclose(b.grad, np.full((1, 4), 3.0))

    def test_matmul_backward_shapes(self) -> None:
        rng = np.random.default_rng(0)
        x = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
        w = Tensor(rng.normal(size=(3, 2)), requires_grad=True)

        out = (x @ w).mean()
        out.backward()

        self.assertEqual(x.grad.shape, x.data.shape)
        self.assertEqual(w.grad.shape, w.data.shape)

    def test_relu_backward(self) -> None:
        x = Tensor(np.array([[-1.0, 0.0, 3.0]]), requires_grad=True)
        y = x.relu().sum()
        y.backward()
        np.testing.assert_allclose(x.grad, np.array([[0.0, 0.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
