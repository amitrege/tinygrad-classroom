"""Unit tests for finite-difference gradient checks."""

from __future__ import annotations

import unittest

from mlstack.gradcheck import check_linear_layer_grad


class TestGradCheck(unittest.TestCase):
    def test_gradcheck_passes_without_bug(self) -> None:
        res = check_linear_layer_grad(seed=1, introduce_bug=False)
        self.assertTrue(res["passed"])

    def test_gradcheck_fails_with_intentional_bug(self) -> None:
        res = check_linear_layer_grad(seed=1, introduce_bug=True)
        self.assertFalse(res["passed"])


if __name__ == "__main__":
    unittest.main()
