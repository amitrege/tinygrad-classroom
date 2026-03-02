"""Loop vs vectorized examples for teaching performance and shapes."""

from __future__ import annotations

import time

import numpy as np


def loop_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Slow Python-loop implementation of y = x @ w + b."""
    n, d = x.shape
    out_dim = w.shape[1]
    out = np.zeros((n, out_dim), dtype=np.float64)

    for i in range(n):
        for j in range(out_dim):
            s = 0.0
            for k in range(d):
                s += x[i, k] * w[k, j]
            out[i, j] = s + b[0, j]
    return out


def vectorized_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w + b


def benchmark_forward(
    batch_size: int = 1024,
    in_dim: int = 64,
    out_dim: int = 32,
    repeats: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    """Return loop and vectorized runtimes in milliseconds."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(batch_size, in_dim))
    w = rng.normal(0.0, 1.0, size=(in_dim, out_dim))
    b = rng.normal(0.0, 1.0, size=(1, out_dim))

    loop_times = []
    vec_times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = loop_forward(x, w, b)
        loop_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = vectorized_forward(x, w, b)
        vec_times.append((time.perf_counter() - t0) * 1000.0)

    loop_ms = float(np.mean(loop_times))
    vec_ms = float(np.mean(vec_times))
    speedup = loop_ms / max(vec_ms, 1e-12)

    return {
        "loop_ms": loop_ms,
        "vectorized_ms": vec_ms,
        "speedup": speedup,
    }
