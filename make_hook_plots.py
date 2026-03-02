"""Generate hook visuals for lecture intro: common training failure modes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mlstack.datasets import make_linearly_separable
from mlstack.manual_neuron import train_single_neuron


OUT_DIR = Path("hook_plots")


def _save_plot(values: np.ndarray, title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(values, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x, y = make_linearly_separable(n_samples=240, seed=0)

    stable = train_single_neuron(x, y, lr=0.2, steps=140, seed=0)
    oscillating = train_single_neuron(x, y, lr=1.4, steps=140, seed=0)

    # Simulated flat-accuracy case by shuffling labels.
    rng = np.random.default_rng(0)
    y_bad = y.copy()
    rng.shuffle(y_bad)
    flat = train_single_neuron(x, y_bad, lr=0.2, steps=140, seed=0)

    # Create an artificial NaN-like failure trace for hook slide.
    nan_loss = stable["losses"].copy()
    nan_loss[80:] = np.nan

    _save_plot(stable["losses"], "Healthy training", "BCE loss", "healthy_loss.png")
    _save_plot(oscillating["losses"], "Oscillating/diverging training (high LR)", "BCE loss", "oscillating_loss.png")
    _save_plot(flat["losses"], "Loss on corrupted labels (won't improve much)", "BCE loss", "flat_loss.png")
    _save_plot(nan_loss, "Numerical failure (NaN loss)", "BCE loss", "nan_loss.png")

    print(f"Saved plots to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
