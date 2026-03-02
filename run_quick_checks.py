"""Fast smoke checks for Class 1 package."""

from __future__ import annotations

import numpy as np

from mlstack.datasets import make_linearly_separable, make_two_moons
from mlstack.gradcheck import check_linear_layer_grad
from mlstack.manual_neuron import train_single_neuron
from mlstack.train import train_binary_mlp


def main() -> None:
    print("[1/4] Dataset generation...")
    x_lin, y_lin = make_linearly_separable(n_samples=200, seed=0)
    x_moon, y_moon = make_two_moons(n_samples=240, seed=7)
    print(f"  linear: X={x_lin.shape}, y={y_lin.shape}")
    print(f"  moons : X={x_moon.shape}, y={y_moon.shape}")

    print("[2/4] Manual neuron training...")
    manual = train_single_neuron(x_lin, y_lin, lr=0.2, steps=120, seed=0)
    print(f"  final loss={manual['losses'][-1]:.4f}, acc={manual['train_acc']:.3f}")

    print("[3/4] Gradient check...")
    grad_ok = check_linear_layer_grad(seed=0, introduce_bug=False)
    grad_bug = check_linear_layer_grad(seed=0, introduce_bug=True)
    print(f"  clean relative error={grad_ok['relative_error']:.2e} (pass={grad_ok['passed']})")
    print(f"  bug   relative error={grad_bug['relative_error']:.2e} (pass={grad_bug['passed']})")

    print("[4/4] Tiny MLP training...")
    out = train_binary_mlp(x_moon, y_moon, hidden_dim=12, lr=0.03, steps=120, optimizer_name="adam", seed=7)
    print(f"  final loss={out['losses'][-1]:.4f}, acc={out['accuracies'][-1]:.3f}")

    print("All quick checks completed.")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
