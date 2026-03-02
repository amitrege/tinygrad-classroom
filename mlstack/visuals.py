"""Plot helpers used by class demos and app."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def grid_for_points(
    x: np.ndarray,
    padding: float = 0.8,
    resolution: int = 220,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
    y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    return xx, yy, grid


def plot_decision_surface(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    probs: np.ndarray,
    title: str,
) -> None:
    p = probs.reshape(xx.shape)
    contour = ax.contourf(xx, yy, p, levels=20, cmap="RdYlBu", alpha=0.35)
    ax.contour(xx, yy, p, levels=[0.5], colors="black", linewidths=1.8)
    ax.scatter(x[:, 0], x[:, 1], c=y.ravel(), cmap="RdYlBu", edgecolors="k", s=28)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.figure.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)


def plot_curve(ax: Axes, values: np.ndarray, title: str, ylabel: str) -> None:
    ax.plot(values, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
