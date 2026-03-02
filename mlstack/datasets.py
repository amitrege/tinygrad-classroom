"""Synthetic datasets used in class demos."""

from __future__ import annotations

import numpy as np


def standardize_features(x: np.ndarray) -> np.ndarray:
    """Return zero-mean unit-variance features."""
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sigma


def make_linearly_separable(
    n_samples: int = 240,
    seed: int = 0,
    noise: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary dataset that a single neuron can solve well."""
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    x0 = rng.normal(loc=[-1.2, -1.0], scale=noise, size=(n0, 2))
    x1 = rng.normal(loc=[1.2, 1.0], scale=noise, size=(n1, 2))

    x = np.vstack([x0, x1]).astype(np.float64)
    y = np.vstack([np.zeros((n0, 1)), np.ones((n1, 1))]).astype(np.float64)

    perm = rng.permutation(n_samples)
    x = x[perm]
    y = y[perm]

    return standardize_features(x), y


def make_two_moons(
    n_samples: int = 300,
    noise: float = 0.08,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-moons dataset without external dependencies."""
    rng = np.random.default_rng(seed)
    n_top = n_samples // 2
    n_bottom = n_samples - n_top

    theta_top = rng.uniform(0.0, np.pi, size=n_top)
    theta_bottom = rng.uniform(0.0, np.pi, size=n_bottom)

    top = np.column_stack((np.cos(theta_top), np.sin(theta_top)))
    bottom = np.column_stack((1.0 - np.cos(theta_bottom), -np.sin(theta_bottom) - 0.45))

    x = np.vstack([top, bottom]).astype(np.float64)
    x += rng.normal(0.0, noise, size=x.shape)

    y = np.vstack([np.zeros((n_top, 1)), np.ones((n_bottom, 1))]).astype(np.float64)

    perm = rng.permutation(n_samples)
    x = x[perm]
    y = y[perm]

    return standardize_features(x), y
