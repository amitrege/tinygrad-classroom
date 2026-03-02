"""Interactive Class 1 app: first-principles progression from scalar neuron to multivariable models."""

from __future__ import annotations

import importlib.util

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from mlstack.autograd import Tensor
from mlstack.datasets import make_linearly_separable, make_two_moons
from mlstack.gradcheck import check_linear_layer_grad
from mlstack.manual_neuron import predict_proba, train_single_neuron
from mlstack.nn import Adam, Linear, ReLU, SGD, Sequential, binary_accuracy_from_logits, binary_cross_entropy_with_logits
from mlstack.train import predict_logits, train_binary_mlp
from mlstack.vectorization import benchmark_forward, loop_forward, vectorized_forward
from mlstack.visuals import grid_for_points, plot_curve, plot_decision_surface


st.set_page_config(page_title="Class 1 ML Fundamentals", layout="wide")

STAGES = [
    "1) Scalar Neuron Fundamentals",
    "2) Scalar Losses and Manual Gradients",
    "3) Computational Graph and Topo Sort",
    "4) Tiny Autograd Implementation",
    "5) NumPy -> PyTorch Mirror",
    "6) Move to Multivariable",
    "7) End-to-End Demos",
]


# =========================
# Core helpers
# =========================
def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def _scalar_example_details(x: float, y: float, w: float, b: float, loss_name: str) -> dict[str, float]:
    """Return full scalar forward/backward details for one sample."""
    z = w * x + b
    p = float(_sigmoid_np(np.array(z)))

    if loss_name == "mse":
        loss = (p - y) ** 2
        dL_dp = 2.0 * (p - y)
        dp_dz = p * (1.0 - p)
        dL_dz = dL_dp * dp_dz
    else:
        eps = 1e-12
        p_safe = min(max(p, eps), 1.0 - eps)
        loss = -(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))
        dL_dp = -(y / p_safe) + ((1.0 - y) / (1.0 - p_safe))
        dp_dz = p_safe * (1.0 - p_safe)
        dL_dz = p_safe - y  # Simplified exact result for BCE+sigmoid.

    dz_dw = x
    dz_db = 1.0

    dL_dw = dL_dz * dz_dw
    dL_db = dL_dz * dz_db

    return {
        "x": x,
        "y": y,
        "w": w,
        "b": b,
        "z": z,
        "p": p,
        "loss": float(loss),
        "dL_dp": float(dL_dp),
        "dp_dz": float(dp_dz),
        "dL_dz": float(dL_dz),
        "dz_dw": float(dz_dw),
        "dz_db": float(dz_db),
        "dL_dw": float(dL_dw),
        "dL_db": float(dL_db),
    }


def _train_scalar_dataset(
    x: np.ndarray,
    y: np.ndarray,
    loss_name: str,
    lr: float,
    steps: int,
    seed: int,
    checkpoint_every: int,
) -> dict[str, object]:
    """Train scalar logistic model (1 feature) with manual gradients."""
    rng = np.random.default_rng(seed)
    w = float(rng.normal(0.0, 0.6))
    b = float(rng.normal(0.0, 0.1))

    losses: list[float] = []
    grad_norms: list[float] = []
    checkpoints: list[dict[str, float]] = []

    for step in range(steps):
        z = x * w + b
        p = _sigmoid_np(z)

        if loss_name == "mse":
            loss = float(np.mean((p - y) ** 2))
            dz = 2.0 * (p - y) * p * (1.0 - p)
        else:
            p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)
            loss = float(np.mean(-(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))))
            dz = p - y

        dw = float(np.mean(dz * x))
        db = float(np.mean(dz))

        losses.append(loss)
        grad_norms.append(float(np.sqrt(dw * dw + db * db)))

        if step % checkpoint_every == 0 or step == steps - 1:
            checkpoints.append({"step": float(step), "w": w, "b": b, "loss": loss})

        w -= lr * dw
        b -= lr * db

    probs = _sigmoid_np(x * w + b)
    acc = float(((probs >= 0.5).astype(np.float64) == y).mean())

    return {
        "w": w,
        "b": b,
        "losses": np.array(losses, dtype=np.float64),
        "grad_norms": np.array(grad_norms, dtype=np.float64),
        "checkpoints": checkpoints,
        "train_acc": acc,
    }


# =========================
# Cached data and demos
# =========================
@st.cache_data(show_spinner=False)
def get_scalar_data(seed: int = 11, n_samples: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """One-feature binary classification dataset."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n_samples, 1)).astype(np.float64)
    logits = 1.9 * x - 0.2 + rng.normal(0.0, 0.45, size=(n_samples, 1))
    y = (logits > 0.0).astype(np.float64)
    return x, y


@st.cache_data(show_spinner=False)
def get_linear_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return make_linearly_separable(n_samples=260, seed=seed, noise=0.26)


@st.cache_data(show_spinner=False)
def get_moons_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    return make_two_moons(n_samples=320, noise=0.09, seed=seed)


@st.cache_data(show_spinner=False)
def run_manual_demo(lr: float, steps: int, seed: int) -> dict[str, object]:
    x, y = get_linear_data(seed=seed)
    return train_single_neuron(x, y, lr=lr, steps=steps, seed=seed, checkpoint_every=max(1, steps // 4))


@st.cache_data(show_spinner=False)
def run_scalar_manual_demo(loss_name: str, lr: float, steps: int, seed: int) -> dict[str, object]:
    x, y = get_scalar_data(seed=seed)
    out = _train_scalar_dataset(
        x=x,
        y=y,
        loss_name=loss_name,
        lr=lr,
        steps=steps,
        seed=seed,
        checkpoint_every=max(1, steps // 4),
    )
    out["x"] = x
    out["y"] = y
    return out


@st.cache_data(show_spinner=False)
def run_single_neuron_on_moons(lr: float, steps: int, seed: int) -> dict[str, object]:
    x, y = get_moons_data(seed=seed)
    result = train_single_neuron(x, y, lr=lr, steps=steps, seed=seed, checkpoint_every=max(1, steps // 4))

    xx, yy, grid = grid_for_points(x, padding=0.8, resolution=220)
    probs_grid = predict_proba(grid, result["w"], float(result["b"]))

    return {
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
        "probs_grid": probs_grid,
        "losses": result["losses"],
        "train_acc": result["train_acc"],
    }


@st.cache_data(show_spinner=False)
def run_vector_benchmark(batch: int, in_dim: int, out_dim: int) -> dict[str, float]:
    return benchmark_forward(batch_size=batch, in_dim=in_dim, out_dim=out_dim, repeats=6, seed=0)


@st.cache_data(show_spinner=False)
def run_mlp_demo(
    hidden_dim: int,
    lr: float,
    steps: int,
    optimizer_name: str,
    seed: int,
) -> dict[str, np.ndarray]:
    x, y = get_moons_data(seed=seed)
    result = train_binary_mlp(
        x,
        y,
        hidden_dim=hidden_dim,
        lr=lr,
        steps=steps,
        optimizer_name=optimizer_name,
        batch_size=64,
        seed=seed,
    )

    model = result["model"]
    xx, yy, grid = grid_for_points(x, padding=0.8, resolution=220)
    logits_grid = predict_logits(model, grid)
    probs_grid = 1.0 / (1.0 + np.exp(-np.clip(logits_grid, -60.0, 60.0)))

    return {
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
        "probs_grid": probs_grid,
        "losses": result["losses"],
        "accuracies": result["accuracies"],
        "grad_norms": result["grad_norms"],
    }


@st.cache_data(show_spinner=False)
def run_torch_scalar(seed: int, lr: float, steps: int) -> dict[str, object] | None:
    """PyTorch scalar-feature logistic training demo."""
    if importlib.util.find_spec("torch") is None:
        return None

    import torch

    x_np, y_np = get_scalar_data(seed=seed)

    torch.manual_seed(seed)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    losses: list[float] = []
    grad_norms: list[float] = []
    weight_trace: list[float] = []
    bias_trace: list[float] = []
    with torch.enable_grad():
        for _ in range(steps):
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()

            # Track gradient norm so we can visualize optimization dynamics.
            grad_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_sq += float(torch.sum(p.grad * p.grad).detach().cpu().item())
            grad_norms.append(float(np.sqrt(grad_sq)))

            opt.step()
            losses.append(float(loss.detach().cpu().item()))

            with torch.no_grad():
                weight_trace.append(float(model.weight.detach().cpu().reshape(()).item()))
                bias_trace.append(float(model.bias.detach().cpu().reshape(()).item()))

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        acc = float(((probs > 0.5).float() == y).float().mean().item())

    w = float(model.weight.detach().cpu().numpy().reshape(()).item())
    b = float(model.bias.detach().cpu().numpy().reshape(()).item())

    return {
        "x": x_np,
        "y": y_np,
        "w": w,
        "b": b,
        "losses": np.array(losses, dtype=np.float64),
        "grad_norms": np.array(grad_norms, dtype=np.float64),
        "weight_trace": np.array(weight_trace, dtype=np.float64),
        "bias_trace": np.array(bias_trace, dtype=np.float64),
        "acc": acc,
    }


@st.cache_data(show_spinner=False)
def run_torch_scalar_autograd_demo(
    *,
    x: float,
    y: float,
    w: float,
    b: float,
    loss_name: str,
) -> dict[str, object] | None:
    """Compare manual scalar derivatives vs PyTorch autograd for one sample."""
    if importlib.util.find_spec("torch") is None:
        return None

    import torch

    dtype = torch.float64

    x_t = torch.tensor(x, dtype=dtype)
    y_t = torch.tensor(y, dtype=dtype)
    w_t = torch.tensor(w, dtype=dtype, requires_grad=True)
    b_t = torch.tensor(b, dtype=dtype, requires_grad=True)

    z_t = w_t * x_t + b_t
    p_t = torch.sigmoid(z_t)

    if loss_name == "mse":
        loss_t = (p_t - y_t) ** 2
    else:
        p_safe = torch.clamp(p_t, 1e-12, 1.0 - 1e-12)
        loss_t = -(y_t * torch.log(p_safe) + (1.0 - y_t) * torch.log(1.0 - p_safe))

    dL_dz_t = torch.autograd.grad(loss_t, z_t, retain_graph=True)[0]
    loss_t.backward()

    manual = _scalar_example_details(x=x, y=y, w=w, b=b, loss_name=loss_name)
    torch_out = {
        "z": float(z_t.detach().cpu().item()),
        "p": float(p_t.detach().cpu().item()),
        "loss": float(loss_t.detach().cpu().item()),
        "dL_dz": float(dL_dz_t.detach().cpu().item()),
        "dL_dw": float(w_t.grad.detach().cpu().item()),
        "dL_db": float(b_t.grad.detach().cpu().item()),
    }

    abs_errors = {
        "z": abs(manual["z"] - torch_out["z"]),
        "p": abs(manual["p"] - torch_out["p"]),
        "loss": abs(manual["loss"] - torch_out["loss"]),
        "dL_dz": abs(manual["dL_dz"] - torch_out["dL_dz"]),
        "dL_dw": abs(manual["dL_dw"] - torch_out["dL_dw"]),
        "dL_db": abs(manual["dL_db"] - torch_out["dL_db"]),
    }

    return {"manual": manual, "torch": torch_out, "abs_errors": abs_errors}


@st.cache_data(show_spinner=False)
def run_torch_batch_compare(
    *,
    seed: int,
    lr: float,
    epochs: int,
    mini_batch_size: int,
) -> dict[str, object] | None:
    """Compare mini-batch vs full-batch training dynamics in PyTorch."""
    if importlib.util.find_spec("torch") is None:
        return None

    import torch

    x_np, y_np = get_scalar_data(seed=seed)
    x_t = torch.tensor(x_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)
    n = x_t.shape[0]

    def train_with_batch(batch_size: int) -> tuple[np.ndarray, float]:
        # Reset seed before each run so both runs start from identical initialization.
        torch.manual_seed(seed)
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.SGD(model.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(x_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        epoch_losses: list[float] = []
        for _ in range(epochs):
            weighted_sum = 0.0
            sample_count = 0

            for xb, yb in loader:
                logits = model(xb)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

                batch_n = int(xb.shape[0])
                weighted_sum += float(loss.detach().cpu().item()) * batch_n
                sample_count += batch_n

            epoch_losses.append(weighted_sum / max(1, sample_count))

        with torch.no_grad():
            probs = torch.sigmoid(model(x_t))
            acc = float(((probs > 0.5).float() == y_t).float().mean().item())

        return np.array(epoch_losses, dtype=np.float64), acc

    mini = max(1, min(int(mini_batch_size), n))
    mini_losses, mini_acc = train_with_batch(mini)
    full_losses, full_acc = train_with_batch(n)

    return {
        "mini_batch_size": mini,
        "mini_losses": mini_losses,
        "mini_acc": mini_acc,
        "full_losses": full_losses,
        "full_acc": full_acc,
    }


def _dataset_by_name(dataset_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return requested binary dataset by chapter-friendly name."""
    if dataset_name == "linear":
        return get_linear_data(seed=seed)
    if dataset_name == "moons":
        return get_moons_data(seed=seed)
    raise ValueError(f"Unknown dataset_name={dataset_name!r}")


def _tiny_grad_norm(params: list[Tensor]) -> float:
    """L2 norm over all parameter gradients in tiny framework."""
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(np.sum(p.grad * p.grad))
    return float(np.sqrt(total))


@st.cache_data(show_spinner=False)
def run_tiny_end_to_end_demo(
    *,
    dataset_name: str,
    model_kind: str,
    hidden_dim: int,
    lr: float,
    steps: int,
    optimizer_name: str,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    """Train tiny-framework model and return diagnostics + decision surface."""
    x, y = _dataset_by_name(dataset_name, seed=seed)

    if model_kind == "linear":
        # One linear layer -> one decision hyperplane.
        model = Sequential(Linear(x.shape[1], 1, seed=seed))
    elif model_kind == "mlp":
        # Two-layer MLP -> nonlinear boundary capacity.
        model = Sequential(
            Linear(x.shape[1], hidden_dim, seed=seed),
            ReLU(),
            Linear(hidden_dim, 1, seed=seed + 1),
        )
    else:
        raise ValueError(f"Unknown model_kind={model_kind!r}")

    params = model.parameters()
    opt_name = optimizer_name.lower()
    if opt_name == "sgd":
        optimizer = SGD(params, lr=lr, momentum=0.0)
    elif opt_name == "momentum":
        optimizer = SGD(params, lr=lr, momentum=0.9)
    else:
        optimizer = Adam(params, lr=lr)

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    bsz = max(1, min(batch_size, n))

    losses: list[float] = []
    accuracies: list[float] = []
    grad_norms: list[float] = []

    for _step in range(steps):
        # Sample a mini-batch so update noise and speed resemble practical training.
        if bsz < n:
            idx = rng.choice(n, size=bsz, replace=False)
            xb = x[idx]
            yb = y[idx]
        else:
            xb = x
            yb = y

        # Forward pass on tiny autograd graph.
        logits = model(Tensor(xb, requires_grad=False))
        loss_out = binary_cross_entropy_with_logits(logits, yb)
        loss = loss_out.loss

        # Standard optimization step: zero -> backward -> step.
        optimizer.zero_grad()
        loss.backward()
        grad_norms.append(_tiny_grad_norm(params))
        optimizer.step()

        # Track full-dataset accuracy to compare learning curves cleanly.
        full_logits = model(Tensor(x, requires_grad=False)).data
        train_acc = binary_accuracy_from_logits(full_logits, y)
        losses.append(float(loss.data))
        accuracies.append(train_acc)

    xx, yy, grid = grid_for_points(x, padding=0.8, resolution=220)
    logits_grid = predict_logits(model, grid)
    probs_grid = 1.0 / (1.0 + np.exp(-np.clip(logits_grid, -60.0, 60.0)))

    return {
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
        "probs_grid": probs_grid,
        "losses": np.array(losses, dtype=np.float64),
        "accuracies": np.array(accuracies, dtype=np.float64),
        "grad_norms": np.array(grad_norms, dtype=np.float64),
        "train_acc": float(accuracies[-1]) if accuracies else 0.0,
    }


@st.cache_data(show_spinner=False)
def run_torch_end_to_end_demo(
    *,
    dataset_name: str,
    model_kind: str,
    hidden_dim: int,
    lr: float,
    steps: int,
    optimizer_name: str,
    batch_size: int,
    seed: int,
) -> dict[str, object] | None:
    """Train PyTorch model and return diagnostics + decision surface."""
    if importlib.util.find_spec("torch") is None:
        return None

    import torch
    import torch.nn.functional as F

    x_np, y_np = _dataset_by_name(dataset_name, seed=seed)
    x_t = torch.tensor(x_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)

    torch.manual_seed(seed)
    if model_kind == "linear":
        model = torch.nn.Linear(x_t.shape[1], 1)
    elif model_kind == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(x_t.shape[1], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
    else:
        raise ValueError(f"Unknown model_kind={model_kind!r}")

    opt_name = optimizer_name.lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rng = np.random.default_rng(seed)
    n = x_t.shape[0]
    bsz = max(1, min(batch_size, int(n)))

    losses: list[float] = []
    accuracies: list[float] = []
    grad_norms: list[float] = []

    for _step in range(steps):
        # Match tiny-framework batching policy for a fair conceptual comparison.
        if bsz < n:
            idx = rng.choice(int(n), size=bsz, replace=False)
            idx_t = torch.from_numpy(idx).long()
            xb = x_t[idx_t]
            yb = y_t[idx_t]
        else:
            xb = x_t
            yb = y_t

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)

        optimizer.zero_grad()
        loss.backward()

        grad_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq += float(torch.sum(p.grad * p.grad).detach().cpu().item())
        grad_norms.append(float(np.sqrt(grad_sq)))

        optimizer.step()

        with torch.no_grad():
            full_logits = model(x_t)
            preds = (full_logits > 0.0).float()
            train_acc = float((preds == y_t).float().mean().item())

        losses.append(float(loss.detach().cpu().item()))
        accuracies.append(train_acc)

    xx, yy, grid = grid_for_points(x_np, padding=0.8, resolution=220)
    grid_t = torch.tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        probs_grid = torch.sigmoid(model(grid_t)).detach().cpu().numpy()

    return {
        "x": x_np,
        "y": y_np,
        "xx": xx,
        "yy": yy,
        "probs_grid": probs_grid,
        "losses": np.array(losses, dtype=np.float64),
        "accuracies": np.array(accuracies, dtype=np.float64),
        "grad_norms": np.array(grad_norms, dtype=np.float64),
        "train_acc": float(accuracies[-1]) if accuracies else 0.0,
    }


# =========================
# Graph + visualization helpers
# =========================
def _collect_topo(root: Tensor) -> list[Tensor]:
    topo: list[Tensor] = []
    visited: set[int] = set()

    def dfs(node: Tensor) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        for parent in node._prev:
            dfs(parent)
        topo.append(node)

    dfs(root)
    return topo


def _node_name(node: Tensor) -> str:
    if node.label:
        return node.label
    if node._op:
        return node._op
    return "input"


def _node_operator(node: Tensor) -> str:
    return node._op if node._op else "leaf"


def _node_scalar_value(node: Tensor) -> float:
    if node.data.size == 1:
        return float(node.data.reshape(()).item())
    return float(np.mean(node.data))


def _draw_tensor_graph(root: Tensor, highlight_node_id: int | None = None) -> plt.Figure:
    """Draw computational graph with optional highlighted node."""
    topo = _collect_topo(root)

    depth_cache: dict[int, int] = {}

    def depth(node: Tensor) -> int:
        node_id = id(node)
        if node_id in depth_cache:
            return depth_cache[node_id]
        if not node._prev:
            depth_cache[node_id] = 0
            return 0
        d = 1 + max(depth(p) for p in node._prev)
        depth_cache[node_id] = d
        return d

    layers: dict[int, list[Tensor]] = {}
    for node in topo:
        d = depth(node)
        layers.setdefault(d, []).append(node)

    positions: dict[int, tuple[float, float]] = {}
    max_layer_size = 1
    for d, nodes in layers.items():
        max_layer_size = max(max_layer_size, len(nodes))
        for idx, node in enumerate(nodes):
            y = -(idx - (len(nodes) - 1) / 2.0)
            positions[id(node)] = (float(d), float(y))

    fig_w = min(14.0, 3.2 + 1.3 * (max(layers.keys()) + 1))
    fig_h = min(9.0, 3.2 + 0.9 * max_layer_size)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Edges first (explicit parent -> child direction).
    for child in topo:
        cx, cy = positions[id(child)]
        for parent in child._prev:
            px, py = positions[id(parent)]
            dx, dy = (cx - px), (cy - py)
            norm = float(np.sqrt(dx * dx + dy * dy) + 1e-12)

            # Offset line endpoints so arrowheads are visible outside node circles.
            start = (px + 0.18 * dx / norm, py + 0.18 * dy / norm)
            end = (cx - 0.26 * dx / norm, cy - 0.26 * dy / norm)
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=2.0,
                    color="#111827",
                    mutation_scale=18,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=2,
            )

    # Nodes
    for node in topo:
        x, y = positions[id(node)]
        is_highlight = highlight_node_id is not None and id(node) == highlight_node_id
        color = "#f59e0b" if is_highlight else "#93c5fd"
        edge = "#92400e" if is_highlight else "#1e3a8a"

        ax.scatter([x], [y], s=2400, c=color, edgecolors=edge, linewidths=2.0, zorder=3)

        grad_txt = "None"
        if node.grad is not None and node.grad.size == 1:
            grad_txt = f"{float(node.grad.reshape(()).item()):.4f}"

        text = (
            f"{_node_name(node)}\n"
            f"op={_node_operator(node)}\n"
            f"val={_node_scalar_value(node):.4f}\n"
            f"grad={grad_txt}"
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=8.1, zorder=4)

    ax.text(
        0.02,
        0.98,
        "edge direction: parent -> child",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#111827",
    )

    ax.set_title("Computational graph (forward edges, backward uses reverse topological order)")
    fig.tight_layout()
    return fig


def _progressive_code_block(
    *,
    key: str,
    title: str | None = None,
    code: str,
    language: str = "python",
    expanded: bool = False,
) -> None:
    """Render code in a collapsed dropdown-style expander."""
    label = title if title is not None else f"Show Code: {key}"
    with st.expander(label, expanded=expanded):
        st.code(code.strip("\n"), language=language)


def _evaluate_expression(expr: str, x: float, w: float, b: float, y: float) -> dict[str, object]:
    """Evaluate a scalar Tensor expression safely in a restricted environment."""
    x_t = Tensor(np.array(x), requires_grad=True, label="x")
    w_t = Tensor(np.array(w), requires_grad=True, label="w")
    b_t = Tensor(np.array(b), requires_grad=True, label="b")
    y_t = Tensor(np.array(y), requires_grad=False, label="y")

    def sigmoid(t: Tensor) -> Tensor:
        return t.sigmoid()

    def relu(t: Tensor) -> Tensor:
        return t.relu()

    def tanh(t: Tensor) -> Tensor:
        return t.tanh()

    local_env = {
        "x": x_t,
        "w": w_t,
        "b": b_t,
        "y": y_t,
        "sigmoid": sigmoid,
        "relu": relu,
        "tanh": tanh,
    }

    loss = eval(expr, {"__builtins__": {}}, local_env)
    if not isinstance(loss, Tensor):
        loss = Tensor(np.array(float(loss)), requires_grad=False, label="loss")

    if loss.data.size != 1:
        loss = loss.mean()

    loss.label = "L"
    loss.backward()

    return {
        "loss": loss,
        "x": x_t,
        "w": w_t,
        "b": b_t,
        "y": y_t,
    }


# =========================
# Plot helpers
# =========================
def _plot_single_neuron_surface(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, title: str) -> plt.Figure:
    xx, yy, grid = grid_for_points(x, padding=0.8, resolution=220)
    probs = predict_proba(grid, w, b)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    plot_decision_surface(ax, x, y, xx, yy, probs, title=title)
    fig.tight_layout()
    return fig


def _plot_checkpoint_boundaries(x: np.ndarray, y: np.ndarray, checkpoints: list[dict[str, object]]) -> plt.Figure:
    picks = [checkpoints[0], checkpoints[len(checkpoints) // 2], checkpoints[-1]]
    titles = ["Early", "Middle", "Final"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    xx, yy, grid = grid_for_points(x, padding=0.8, resolution=160)

    for ax, ckpt, label in zip(axes, picks, titles):
        probs = predict_proba(grid, ckpt["w"], float(ckpt["b"]))
        plot_decision_surface(
            ax,
            x,
            y,
            xx,
            yy,
            probs,
            title=f"{label}: step={int(ckpt['step'])}, loss={ckpt['loss']:.3f}",
        )

    fig.tight_layout()
    return fig


def _plot_scalar_prob_curve_on_ax(ax: plt.Axes, x: np.ndarray, y: np.ndarray, w: float, b: float, title: str) -> None:
    x1 = x.reshape(-1)
    y1 = y.reshape(-1)
    grid = np.linspace(x1.min() - 0.8, x1.max() + 0.8, 250)
    probs = _sigmoid_np(grid * w + b)

    ax.plot(grid, probs, linewidth=2.2, color="#2563eb", label="p(y=1|x)")
    ax.scatter(x1, y1, c=y1, cmap="coolwarm", edgecolors="k", s=26, alpha=0.8, label="data")
    ax.axhline(0.5, color="#374151", linestyle="--", linewidth=1.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.grid(alpha=0.2)


def _plot_scalar_checkpoints(x: np.ndarray, y: np.ndarray, checkpoints: list[dict[str, float]]) -> plt.Figure:
    picks = [checkpoints[0], checkpoints[len(checkpoints) // 2], checkpoints[-1]]
    labels = ["Early", "Middle", "Final"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, ckpt, label in zip(axes, picks, labels):
        _plot_scalar_prob_curve_on_ax(
            ax,
            x,
            y,
            ckpt["w"],
            ckpt["b"],
            title=f"{label}: step={int(ckpt['step'])}, loss={ckpt['loss']:.3f}",
        )
    fig.tight_layout()
    return fig


def _plot_sigmoid_shape(z_value: float) -> plt.Figure:
    z_grid = np.linspace(-8.0, 8.0, 300)
    p_grid = _sigmoid_np(z_grid)
    p_val = float(_sigmoid_np(np.array(z_value)))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(z_grid, p_grid, linewidth=2.3, color="#2563eb")
    ax.scatter([z_value], [p_val], color="#dc2626", s=65, zorder=4)
    ax.axhline(0.5, linestyle="--", color="#6b7280")
    ax.axvline(0.0, linestyle="--", color="#6b7280")
    ax.set_xlabel("z")
    ax.set_ylabel("sigma(z)")
    ax.set_title("Sigmoid curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# =========================
# Stage renderers
# =========================
def _render_stage_scalar_fundamentals() -> None:
    st.subheader("Stage 1: Scalar Neuron Fundamentals")
    subtabs = st.tabs(["1A) What and Why", "1B) Math", "1C) Worked Numbers", "1D) NumPy", "1E) Visual"])

    with subtabs[0]:
        st.markdown("We start with the smallest complete learning unit: one neuron with one input feature.")
        st.markdown(
            "If this unit is clear, then every larger network is just many copies of the same ideas (weighted sum, nonlinearity, loss, gradients)."
        )
        st.markdown("**Goal of this stage**: understand exactly what the neuron computes before any training.")

    with subtabs[1]:
        st.markdown("For one input feature `x`, one weight `w`, and bias `b`:")
        st.latex(r"z = wx + b")
        st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
        st.latex(r"\hat{y} = \sigma(z)")
        st.markdown("Interpretation in plain language:")
        st.markdown("- `z` is the raw score.")
        st.markdown("- `sigma(z)` turns score into probability between 0 and 1.")
        st.markdown("- Decision threshold 0.5 corresponds to `z = 0`.")

    with subtabs[2]:
        st.markdown("Step-by-step numeric forward pass:")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x = st.slider("x", -3.0, 3.0, 1.5, 0.1, key="s1_x")
        with c2:
            w = st.slider("w", -3.0, 3.0, 1.2, 0.1, key="s1_w")
        with c3:
            b = st.slider("b", -3.0, 3.0, 0.2, 0.1, key="s1_b")
        with c4:
            y_true = st.selectbox("true label y", [0.0, 1.0], index=1, key="s1_y")

        z = w * x + b
        p = float(_sigmoid_np(np.array(z)))
        pred = 1.0 if p >= 0.5 else 0.0

        st.table(
            [
                {"step": "weighted sum", "result": f"z = ({w:.2f} × {x:.2f}) + {b:.2f} = {z:.4f}"},
                {"step": "sigmoid", "result": f"p = 1 / (1 + exp(-{z:.4f})) = {p:.4f}"},
                {"step": "class decision", "result": f"pred = {pred:.0f} (threshold 0.5)"},
                {"step": "match?", "result": "correct" if pred == y_true else "incorrect"},
            ]
        )

    with subtabs[3]:
        st.markdown("NumPy implementation of the same forward computation:")
        st.code(
            """import numpy as np


def sigmoid(z):
    # clipping protects exp(-z) from overflow for very large |z|
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, w, b):
    # X: (N, 1), w: scalar or (1, 1), b: scalar
    return sigmoid(X * w + b)
""",
            language="python",
        )

        st.markdown("Why clip `z` in sigmoid?")
        st.markdown("- `exp(-z)` can overflow for very large positive/negative `z`.")
        st.markdown("- Clipping keeps computation numerically stable.")
        st.markdown("- For large `|z|`, sigmoid is already saturated, so clipping does not change practical behavior.")

    with subtabs[4]:
        st.markdown("Interactive scalar-feature classifier view:")
        x_data, y_data = get_scalar_data(seed=11)

        c1, c2 = st.columns(2)
        with c1:
            w_demo = st.slider("demo w", -4.0, 4.0, 1.2, 0.05, key="s1_demo_w")
        with c2:
            b_demo = st.slider("demo b", -4.0, 4.0, 0.0, 0.05, key="s1_demo_b")

        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        _plot_scalar_prob_curve_on_ax(ax, x_data, y_data, w_demo, b_demo, title="1D probability curve and labeled points")
        st.pyplot(fig)

def _render_stage_scalar_gradients() -> None:
    st.subheader("Stage 2: Scalar Losses and Manual Gradients")
    subtabs = st.tabs(["2A) Why Learn", "2B) MSE by Hand", "2C) BCE by Hand", "2D) NumPy Loop", "2E) Training Demo"])

    with subtabs[0]:
        st.markdown(
            "So far, `w` and `b` were fixed by us. That is useful for intuition, but it does not scale to real data."
        )
        st.markdown("Now we learn `w` and `b` directly from data using gradients.")
        st.markdown("Training loop idea in plain language:")
        st.markdown("1. Predict")
        st.markdown("2. Measure error with a loss")
        st.markdown("3. Compute gradients")
        st.markdown("4. Update parameters")

    with subtabs[1]:
        st.markdown("MSE with one sample: `L = (p - y)^2`, where `p = sigma(z)` and `z = wx+b`.")
        st.markdown("First derive everything symbolically, then plug in numbers.")

        st.markdown("Symbolic derivatives:")
        st.latex(r"L(p) = (p-y)^2 \quad \Rightarrow \quad \frac{\partial L}{\partial p} = 2(p-y)")
        st.latex(
            r"p(z) = \sigma(z) = \frac{1}{1+e^{-z}} \quad \Rightarrow \quad "
            r"\frac{\partial p}{\partial z} = \frac{e^{-z}}{(1+e^{-z})^2} = p(1-p)"
        )
        st.latex(r"z(w,b) = wx+b \quad \Rightarrow \quad \frac{\partial z}{\partial w}=x,\;\frac{\partial z}{\partial b}=1")
        st.markdown("Compose with chain rule:")
        st.latex(
            r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial p}\frac{\partial p}{\partial z}\frac{\partial z}{\partial w}"
            r" = 2(p-y)\,p(1-p)\,x"
        )
        st.latex(
            r"\frac{\partial L}{\partial b} = \frac{\partial L}{\partial p}\frac{\partial p}{\partial z}\frac{\partial z}{\partial b}"
            r" = 2(p-y)\,p(1-p)"
        )
        st.latex(
            r"\text{Expanded form: }"
            r"\frac{\partial L}{\partial w}=2(\sigma(wx+b)-y)\,\sigma(wx+b)\,[1-\sigma(wx+b)]\,x"
        )
        st.markdown("Now evaluate the same formulas numerically:")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x = st.slider("x", -3.0, 3.0, 1.4, 0.1, key="s2_mse_x")
        with c2:
            w = st.slider("w", -3.0, 3.0, 0.8, 0.1, key="s2_mse_w")
        with c3:
            b = st.slider("b", -3.0, 3.0, -0.1, 0.1, key="s2_mse_b")
        with c4:
            y = st.selectbox("y", [0.0, 1.0], index=1, key="s2_mse_y")

        d = _scalar_example_details(x=x, y=float(y), w=w, b=b, loss_name="mse")

        st.table(
            [
                {"term": "z = wx+b", "value": f"{d['z']:.6f}"},
                {"term": "p = sigma(z)", "value": f"{d['p']:.6f}"},
                {"term": "L = (p-y)^2", "value": f"{d['loss']:.6f}"},
                {"term": "dL/dp = 2(p-y)", "value": f"{d['dL_dp']:.6f}"},
                {"term": "dp/dz = p(1-p)", "value": f"{d['dp_dz']:.6f}"},
                {"term": "dL/dz", "value": f"{d['dL_dz']:.6f}"},
                {"term": "dz/dw = x", "value": f"{d['dz_dw']:.6f}"},
                {"term": "dz/db = 1", "value": f"{d['dz_db']:.6f}"},
                {"term": "dL/dw", "value": f"{d['dL_dw']:.6f}"},
                {"term": "dL/db", "value": f"{d['dL_db']:.6f}"},
            ]
        )

    with subtabs[2]:
        st.markdown("BCE example with one sample: `L = -(y log p + (1-y) log(1-p))`.")
        st.markdown("This is the standard loss for binary classification with sigmoid output.")
        st.markdown("Again: derive symbolically first, then evaluate with numbers.")

        st.markdown("Symbolic derivatives:")
        st.latex(
            r"L(p) = -\big[y\log p + (1-y)\log(1-p)\big]"
            r"\quad \Rightarrow \quad"
            r"\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}"
        )
        st.latex(r"p(z)=\sigma(z) \quad \Rightarrow \quad \frac{\partial p}{\partial z}=p(1-p)")
        st.latex(r"z(w,b)=wx+b \quad \Rightarrow \quad \frac{\partial z}{\partial w}=x,\;\frac{\partial z}{\partial b}=1")

        st.markdown("Chain-rule composition and simplification:")
        st.latex(
            r"\frac{\partial L}{\partial z}"
            r"=\frac{\partial L}{\partial p}\frac{\partial p}{\partial z}"
            r"=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p)"
        )
        st.latex(r"\frac{\partial L}{\partial z} = -y(1-p)+(1-y)p = p-y")
        st.latex(r"\frac{\partial L}{\partial w} = (p-y)x,\qquad \frac{\partial L}{\partial b}=p-y")
        st.markdown("So BCE+sigmoid gives a clean gradient term: `p - y`.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x = st.slider("x", -3.0, 3.0, 1.1, 0.1, key="s2_bce_x")
        with c2:
            w = st.slider("w", -3.0, 3.0, 0.9, 0.1, key="s2_bce_w")
        with c3:
            b = st.slider("b", -3.0, 3.0, 0.1, 0.1, key="s2_bce_b")
        with c4:
            y = st.selectbox("y", [0.0, 1.0], index=1, key="s2_bce_y")

        d = _scalar_example_details(x=x, y=float(y), w=w, b=b, loss_name="bce")

        st.markdown("Numeric evaluation of the symbolic result:")

        st.table(
            [
                {"term": "z", "value": f"{d['z']:.6f}"},
                {"term": "p", "value": f"{d['p']:.6f}"},
                {"term": "BCE loss", "value": f"{d['loss']:.6f}"},
                {"term": "dL/dp", "value": f"{d['dL_dp']:.6f}"},
                {"term": "dp/dz", "value": f"{d['dp_dz']:.6f}"},
                {"term": "dL/dz = p-y", "value": f"{d['dL_dz']:.6f}"},
                {"term": "dL/dw", "value": f"{d['dL_dw']:.6f}"},
                {"term": "dL/db", "value": f"{d['dL_db']:.6f}"},
            ]
        )

    with subtabs[3]:
        st.markdown("How this becomes code (same 4-step loop every iteration):")
        st.code(
            """# scalar-feature logistic training (batch form)
for step in range(steps):
    z = X * w + b
    p = sigmoid(z)

    # choose loss: BCE or MSE
    if loss_name == "bce":
        dz = p - y
    else:  # mse
        dz = 2.0 * (p - y) * p * (1.0 - p)

    dw = np.mean(dz * X)
    db = np.mean(dz)

    w = w - lr * dw
    b = b - lr * db
""",
            language="python",
        )
        st.markdown("Why do we use `mean` in `dw = np.mean(dz * X)` and `db = np.mean(dz)`?")
        st.latex(r"L = \frac{1}{N}\sum_{i=1}^{N} L_i \quad \Rightarrow \quad \frac{\partial L}{\partial w}=\frac{1}{N}\sum_{i=1}^{N}\frac{\partial L_i}{\partial w}")
        st.markdown(
            "- `dz * X` gives per-sample gradient contributions for `w`.\n"
            "- `np.mean(...)` averages those contributions across the batch.\n"
            "- Averaging keeps gradient scale roughly consistent when batch size changes.\n"
            "- If we used `sum`, gradient magnitude would grow with batch size and learning-rate tuning would be less stable."
        )
        st.markdown("Key intuition: gradients give direction; learning rate sets step size.")

    with subtabs[4]:
        st.markdown("Choose loss function:")
        st.markdown(
            "- **BCE (recommended for binary classification):** stronger and better-behaved gradients for probabilities/logits.\n"
            "- **MSE:** useful for learning mechanics and sometimes regression-style tasks, but can train slower for classification.\n"
            "- Practical default for binary class problems is BCE (or BCE-with-logits)."
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            loss_name = st.selectbox("loss", ["bce", "mse"], index=0, key="s2_demo_loss")
        with c2:
            lr = st.slider("learning rate", 0.001, 1.0, 0.08, 0.001, key="s2_demo_lr")
        with c3:
            steps = st.slider("steps", 40, 700, 260, 20, key="s2_demo_steps")
        with c4:
            seed = st.number_input("seed", min_value=0, max_value=1000, value=11, step=1, key="s2_demo_seed")

        result = run_scalar_manual_demo(loss_name=loss_name, lr=lr, steps=steps, seed=int(seed))
        x, y = result["x"], result["y"]

        left, right = st.columns([1.2, 1.0])
        with left:
            fig = _plot_scalar_checkpoints(x, y, result["checkpoints"])
            st.pyplot(fig)
        with right:
            fig2, axes = plt.subplots(2, 1, figsize=(6.4, 5.6))
            plot_curve(axes[0], result["losses"], f"{loss_name.upper()} loss", "loss")
            plot_curve(axes[1], result["grad_norms"], "Gradient norm", "||g||")
            fig2.tight_layout()
            st.pyplot(fig2)
            st.metric("Final accuracy", f"{100.0 * result['train_acc']:.1f}%")

def _render_stage_graph_topo() -> None:
    st.subheader("Stage 3: Computational Graph and Topological Order")
    subtabs = st.tabs(
        [
            "3A) Recall Forward/Backward",
            "3B) Why DAG + Traversal",
            "3C) Topo Sort in Code",
            "3D) Formula-Driven Playground",
        ]
    )

    with subtabs[0]:
        st.markdown(
            "From Chapter 2: we first did a **forward calculation** to compute loss, "
            "then a **backward calculation** to compute gradients."
        )
        st.markdown(
            "For a simple scalar example we can write this as:"
        )
        st.latex(r"z = wx + b,\quad p=\sigma(z),\quad L=(p-y)^2")
        st.markdown(
            "Forward pass computes values (`x,w,b -> z -> p -> L`). "
            "Backward pass sends gradient information from `L` back to parameters."
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            x = st.slider("x", -3.0, 3.0, 1.2, 0.1, key="s3a_x")
        with c2:
            w = st.slider("w", -3.0, 3.0, 0.9, 0.1, key="s3a_w")
        with c3:
            b = st.slider("b", -3.0, 3.0, -0.3, 0.1, key="s3a_b")
        with c4:
            y = st.selectbox("label y", [0.0, 1.0], index=1, key="s3a_y")
        with c5:
            use_sigmoid = st.toggle("use sigmoid", value=True, key="s3a_sig")

        x_t = Tensor(np.array(x), requires_grad=True, label="x")
        w_t = Tensor(np.array(w), requires_grad=True, label="w")
        b_t = Tensor(np.array(b), requires_grad=True, label="b")
        y_t = Tensor(np.array(float(y)), requires_grad=False, label="y")

        z_t = w_t * x_t + b_t
        z_t.label = "z"
        pred_t = z_t.sigmoid() if use_sigmoid else z_t
        pred_t.label = "p" if use_sigmoid else "z"
        err_t = pred_t - y_t
        err_t.label = "e"
        loss_t = err_t * err_t
        loss_t.label = "L"
        loss_t.backward()

        fig = _draw_tensor_graph(loss_t)
        st.pyplot(fig)
        st.markdown(
            "This graph is small enough to inspect manually. In larger graphs we need a traversal algorithm."
        )
        st.write(
            {
                "loss": float(loss_t.data),
                "dL/dw": float(w_t.grad),
                "dL/db": float(b_t.grad),
                "dL/dx": float(x_t.grad),
            }
        )

    with subtabs[1]:
        st.markdown("A computational graph for neural nets is a **DAG** (Directed Acyclic Graph):")
        st.markdown("- Directed: edges have direction (inputs -> outputs).")
        st.markdown("- Acyclic: no path loops back to itself.")
        st.markdown(
            "Dependencies create levels: some nodes can only be processed after others are available."
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x = st.slider("x", -3.0, 3.0, 1.1, 0.1, key="s3b_x")
        with c2:
            w = st.slider("w", -3.0, 3.0, 0.8, 0.1, key="s3b_w")
        with c3:
            b = st.slider("b", -3.0, 3.0, -0.2, 0.1, key="s3b_b")
        with c4:
            y = st.selectbox("label y", [0.0, 1.0], index=1, key="s3b_y")

        # Branching graph to show dependency levels and grad accumulation:
        # a = w*x, z1 = a+b, z2 = a-y, L = z1*z2
        x_t = Tensor(np.array(x), requires_grad=True, label="x")
        w_t = Tensor(np.array(w), requires_grad=True, label="w")
        b_t = Tensor(np.array(b), requires_grad=True, label="b")
        y_t = Tensor(np.array(float(y)), requires_grad=False, label="y")

        a_t = w_t * x_t
        a_t.label = "a"
        z1_t = a_t + b_t
        z1_t.label = "z1"
        z2_t = a_t - y_t
        z2_t.label = "z2"
        loss_t = z1_t * z2_t
        loss_t.label = "L"
        loss_t.backward()

        topo = _collect_topo(loss_t)

        depth_cache: dict[int, int] = {}

        def depth(node: Tensor) -> int:
            node_id = id(node)
            if node_id in depth_cache:
                return depth_cache[node_id]
            if not node._prev:
                depth_cache[node_id] = 0
                return 0
            d = 1 + max(depth(p) for p in node._prev)
            depth_cache[node_id] = d
            return d

        level_rows = []
        level_map: dict[int, list[str]] = {}
        for node in topo:
            lvl = depth(node)
            level_rows.append(
                {
                    "node": _node_name(node),
                    "level": lvl,
                    "depends_on": ", ".join(_node_name(p) for p in node._prev) if node._prev else "-",
                }
            )
            level_map.setdefault(lvl, []).append(_node_name(node))

        fig = _draw_tensor_graph(loss_t)
        st.pyplot(fig)

        st.markdown("Level grouping (forward dependency levels):")
        for lvl in sorted(level_map):
            st.markdown(f"- Level {lvl}: {', '.join(level_map[lvl])}")

        st.dataframe(level_rows, hide_index=True, use_container_width=True)
        st.markdown(
            "Because of these dependencies, we need an order that respects the DAG structure. "
            "That order is produced by **topological sort**."
        )

    with subtabs[2]:
        st.markdown("Topological sort (DFS) gives a valid order that respects dependencies in a DAG.")
        st.markdown("Why we need it:")
        st.markdown(
            "- A node can only be computed after its parent nodes are computed.\n"
            "- Backward pass also needs a dependency-safe order (reverse topo)."
        )
        st.markdown("Symbolic dependency example:")
        st.latex(r"a = wx,\quad z_1 = a + b,\quad z_2 = a - y,\quad L = z_1 z_2")
        st.markdown(
            "Here `a` must be available before `z1` and `z2`, and both `z1`/`z2` must be available before `L`."
        )
        st.markdown(
            "A valid forward topological order is: `x, w, b, y, a, z1, z2, L` "
            "(inputs can appear in any order before their dependents)."
        )
        st.code(
            """def topo_sort(root):
    topo = []
    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for parent in node.parents:
            dfs(parent)
        topo.append(node)  # parent nodes are already appended

    dfs(root)
    return topo

# forward order: topo
# backward order: reversed(topo)
""",
            language="python",
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x = st.slider("x", -3.0, 3.0, 1.2, 0.1, key="s3c_x")
        with c2:
            w = st.slider("w", -3.0, 3.0, 0.9, 0.1, key="s3c_w")
        with c3:
            b = st.slider("b", -3.0, 3.0, -0.3, 0.1, key="s3c_b")
        with c4:
            y = st.selectbox("label y", [0.0, 1.0], index=1, key="s3c_y")

        x_t = Tensor(np.array(x), requires_grad=True, label="x")
        w_t = Tensor(np.array(w), requires_grad=True, label="w")
        b_t = Tensor(np.array(b), requires_grad=True, label="b")
        y_t = Tensor(np.array(float(y)), requires_grad=False, label="y")

        z_t = w_t * x_t + b_t
        z_t.label = "z"
        p_t = z_t.sigmoid()
        p_t.label = "p"
        e_t = p_t - y_t
        e_t.label = "e"
        loss_t = e_t * e_t
        loss_t.label = "L"
        loss_t.backward()

        topo = _collect_topo(loss_t)
        backward_order = list(reversed(topo))

        view = st.radio("View order", ["forward topo", "backward (reverse topo)"], horizontal=True, key="s3c_view")
        order = topo if view == "forward topo" else backward_order

        step_key = "s3c_step_idx"
        if step_key not in st.session_state:
            st.session_state[step_key] = 0
        if st.session_state[step_key] >= len(order):
            st.session_state[step_key] = len(order) - 1

        n1, n2, n3 = st.columns([1, 1, 2])
        with n1:
            if st.button("Previous Step", key="s3c_prev_btn", use_container_width=True):
                st.session_state[step_key] = max(0, st.session_state[step_key] - 1)
        with n2:
            if st.button("Next Step", key="s3c_next_btn", use_container_width=True):
                st.session_state[step_key] = min(len(order) - 1, st.session_state[step_key] + 1)
        with n3:
            st.metric("Current Step", f"{st.session_state[step_key]}/{len(order) - 1}")

        if st.button("Reset Step", key="s3c_reset_btn"):
            st.session_state[step_key] = 0

        step = st.session_state[step_key]
        current = order[step]

        fig = _draw_tensor_graph(loss_t, highlight_node_id=id(current))
        st.pyplot(fig)

        rows = []
        for idx, node in enumerate(order):
            rows.append(
                {
                    "step": idx,
                    "node": _node_name(node),
                    "value": f"{_node_scalar_value(node):.5f}",
                    "current": "<--" if idx == step else "",
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.markdown("This applies topological sort directly to the created graph.")

    with subtabs[3]:
        st.markdown("Pick a formula template. The symbolic form and graph are shown together.")
        st.markdown(
            "`y` is the **label** (ground-truth target). "
            "Without a label, we cannot compute supervised loss because loss compares prediction vs target."
        )

        templates = {
            "Linear + MSE": {
                "expr": "(w*x + b - y) ** 2",
                "symbolic": [
                    r"z = wx + b",
                    r"L = (z-y)^2",
                    r"\frac{\partial L}{\partial w}=2(z-y)x,\;\frac{\partial L}{\partial b}=2(z-y)",
                ],
            },
            "Sigmoid + MSE": {
                "expr": "(sigmoid(w*x + b) - y) ** 2",
                "symbolic": [
                    r"z = wx + b,\quad p=\sigma(z)",
                    r"L = (p-y)^2",
                    r"\frac{\partial L}{\partial z}=2(p-y)p(1-p)",
                ],
            },
            "Sigmoid + BCE": {
                "expr": "-(y * sigmoid(w*x + b).log() + (1 - y) * (1 - sigmoid(w*x + b)).log())",
                "symbolic": [
                    r"z = wx + b,\quad p=\sigma(z)",
                    r"L = -\left(y\log p + (1-y)\log(1-p)\right)",
                    r"\frac{\partial L}{\partial z}=p-y",
                ],
            },
        }

        template_name = st.selectbox("formula template", list(templates.keys()), index=1, key="s3d_template")
        template = templates[template_name]

        st.markdown("Symbolic formulas for selected template:")
        for line in template["symbolic"]:
            st.latex(line)

        expr = template["expr"]
        manual_edit = st.toggle("edit expression manually", value=False, key="s3d_edit")
        if manual_edit:
            expr = st.text_input("expression", expr, key="s3d_expr_edit")
        else:
            st.code(expr, language="python")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x = st.slider("x", -3.0, 3.0, 1.2, 0.1, key="s3d_x")
        with c2:
            w = st.slider("w", -3.0, 3.0, 0.9, 0.1, key="s3d_w")
        with c3:
            b = st.slider("b", -3.0, 3.0, -0.3, 0.1, key="s3d_b")
        with c4:
            y = st.selectbox("label y", [0.0, 1.0], index=1, key="s3d_y")

        try:
            out = _evaluate_expression(expr=expr, x=x, w=w, b=b, y=float(y))
            loss_t = out["loss"]
            fig = _draw_tensor_graph(loss_t)

            left, right = st.columns([1.3, 1.0])
            with left:
                st.pyplot(fig)
            with right:
                st.metric("Loss", f"{float(loss_t.data):.6f}")
                st.write(
                    {
                        "dL/dx": float(out["x"].grad),
                        "dL/dw": float(out["w"].grad),
                        "dL/db": float(out["b"].grad),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Expression could not be evaluated: {exc}")

def _render_stage_tiny_autograd() -> None:
    st.subheader("Stage 4: Tiny Autograd Implementation")
    subtabs = st.tabs(
        [
            "4A) Why Autograd",
            "4B) General Class Design",
            "4C) Tensor Core Code",
            "4D) Operator Forward/Backward",
            "4E) Backward Engine",
            "4F) Tests and Validation",
            "4G) Old vs New Interface",
            "4H) Debugging Code",
        ]
    )

    with subtabs[0]:
        st.markdown(
            "Before autograd, we manually derived gradients for each new equation. "
            "That works for tiny examples, but it becomes unmanageable as expressions get deeper."
        )
        st.markdown("Why autograd is needed:")
        st.markdown("- Neural network losses are compositions of many operators.")
        st.markdown("- Manual derivative code for every new expression is slow and error-prone.")
        st.markdown("- We want a reusable system that handles any expression built from known operators.")
        st.markdown("Key idea:")
        st.markdown(
            "If we implement forward + local backward rule for each operator (`+`, `*`, `matmul`, `sigmoid`, ...), "
            "then any composed equation gets gradients automatically by graph traversal."
        )
        st.markdown("Symbolic examples that share operator building blocks:")
        st.latex(r"L_1 = (wx+b-y)^2")
        st.latex(r"L_2 = (\sigma(wx+b)-y)^2")
        st.latex(r"L_3 = -\left[y\log(\sigma(wx+b)) + (1-y)\log(1-\sigma(wx+b))\right]")
        st.markdown("All three are just combinations of primitive operators.")

    with subtabs[1]:
        st.markdown("We implement autograd with a **general Tensor class** so the same engine works for all equations.")
        st.table(
            [
                {"field/method": "`data`", "why needed": "Stores numeric value from forward pass."},
                {"field/method": "`grad`", "why needed": "Stores accumulated gradient from backward pass."},
                {"field/method": "`_prev`", "why needed": "Parents in computation graph for traversal."},
                {"field/method": "`_op`", "why needed": "Operator label for introspection/debugging."},
                {"field/method": "`_backward`", "why needed": "Local derivative rule for this operator."},
                {"field/method": "`backward()`", "why needed": "Runs reverse-topological gradient propagation."},
                {"field/method": "`zero_grad()`", "why needed": "Prevents stale gradients across iterations."},
            ]
        )
        st.markdown("Extensibility principle:")
        st.markdown("- Add a new operator once (forward + backward).")
        st.markdown("- Instantly supports all future formulas that use that operator.")

    with subtabs[2]:
        st.markdown("Core Tensor class skeleton:")
        _progressive_code_block(
            key="s4_tensor_core",
            title="Show Tensor Core Class",
            code="""
class Tensor:
    def __init__(self, data, requires_grad=False, _prev=(), _op=""):
        # `data` is the forward value at this graph node.
        # We force NumPy float arrays so arithmetic behavior is consistent across ops.
        self.data = np.array(data, dtype=np.float64)

        # `requires_grad=True` means this tensor is a leaf/parameter (or intermediate)
        # for which we want dL/d(data) during backprop.
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

        # `_prev` stores parent nodes used to produce this tensor.
        # This is how we recover graph structure for reverse traversal.
        self._prev = _prev

        # `_op` is a human-readable operator tag ("add", "mul", "matmul", ...).
        # It is optional for math, but critical for debugging/graph visualization.
        self._op = _op

        # `_backward` is a closure set by each operator implementation.
        # When called, it pushes this node's gradient contribution to parents.
        self._backward = lambda: None

    def zero_grad(self):
        # Gradients accumulate by design (`+=` during backprop).
        # Therefore we must clear them between optimization steps.
        if self.grad is not None:
            self.grad.fill(0.0)
""",
        )
        st.markdown("Small but important helper functions:")
        _progressive_code_block(
            key="s4_helpers",
            title="Show Helper Utilities (`ensure` and `_unbroadcast`)",
            code="""
def ensure(x):
    # Utility: normalize non-Tensor inputs.
    # This lets expressions like `tensor + 3.0` work by promoting `3.0` to Tensor.
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

def _unbroadcast(grad, shape):
    # During forward pass, NumPy may broadcast small tensors to larger shapes.
    # Example: bias shape (1, d) added to batch shape (n, d).
    # Backward must invert that expansion so returned gradient matches original parameter shape.

    # Step 1: remove extra leading dimensions introduced by broadcasting.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Step 2: for each axis where original dim == 1,
    # all copied entries correspond to one original value.
    # So gradients from those copies must be summed.
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    # Step 3: final reshape for exact shape parity.
    return grad.reshape(shape)
""",
        )
        st.markdown("Why `_unbroadcast` matters: without it, parameter gradients have wrong shape/magnitude.")

    with subtabs[3]:
        st.markdown("Operators seen so far, with both forward and backward rules.")
        st.markdown("`add` and `mul`:")
        _progressive_code_block(
            key="s4_add_mul",
            title="Show `add` and `mul` Operators (Forward + Backward)",
            code="""
def __add__(self, other):
    # Normalize so this operator supports Tensor+Tensor and Tensor+scalar uniformly.
    other = ensure(other)

    # Forward: out = a + b
    out = Tensor(
        self.data + other.data,
        requires_grad=self.requires_grad or other.requires_grad,
        _prev=(self, other),
        _op="add",
    )

    def _backward():
        # If no upstream gradient reached `out`, there is no contribution to propagate.
        if out.grad is None:
            return

        # Local derivatives:
        # d(a+b)/da = 1
        # d(a+b)/db = 1
        # Chain rule: dL/da += dL/dout * 1
        if self.requires_grad:
            self.grad += _unbroadcast(out.grad, self.data.shape)

        # Same propagation rule for the second parent.
        if other.requires_grad:
            other.grad += _unbroadcast(out.grad, other.data.shape)

    out._backward = _backward
    return out

def __mul__(self, other):
    # Normalize mixed input types.
    other = ensure(other)

    # Forward: out = a * b
    out = Tensor(
        self.data * other.data,
        requires_grad=self.requires_grad or other.requires_grad,
        _prev=(self, other),
        _op="mul",
    )

    def _backward():
        if out.grad is None:
            return

        # Local derivatives for product:
        # d(a*b)/da = b
        # d(a*b)/db = a
        # Chain rule:
        # dL/da += dL/dout * b
        # dL/db += dL/dout * a
        if self.requires_grad:
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
        if other.requires_grad:
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

    out._backward = _backward
    return out
""",
        )

        st.markdown("`matmul` for multivariable chapter:")
        _progressive_code_block(
            key="s4_matmul",
            title="Show `matmul` Operator (Forward + Backward)",
            code="""
def matmul(self, other):
    # Support Tensor @ scalar/array combinations by normalizing inputs first.
    other = ensure(other)

    # Forward: out = A @ B
    out = Tensor(
        self.data @ other.data,
        requires_grad=self.requires_grad or other.requires_grad,
        _prev=(self, other),
        _op="matmul",
    )

    def _backward():
        if out.grad is None:
            return

        # For matrix product C = A @ B:
        # dL/dA = dL/dC @ B^T
        # dL/dB = A^T @ dL/dC
        # This is the vectorized chain rule for linear maps.
        if self.requires_grad:
            self.grad += out.grad @ other.data.T
        if other.requires_grad:
            other.grad += self.data.T @ out.grad

    out._backward = _backward
    return out
""",
        )

        st.markdown("Nonlinearities and reductions:")
        _progressive_code_block(
            key="s4_nonlinear_reduce",
            title="Show `sigmoid`, `relu`, and `sum` Operators",
            code="""
def sigmoid(self):
    # Forward: sigma(z) = 1 / (1 + exp(-z))
    # We clip z to keep exp numerically stable for large-magnitude values.
    s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
    out = Tensor(s, requires_grad=self.requires_grad, _prev=(self,), _op="sigmoid")

    def _backward():
        if out.grad is None or not self.requires_grad:
            return
        # Local derivative:
        # d sigma / dz = sigma(z) * (1 - sigma(z))
        # Chain rule:
        # dL/dz += dL/dout * d(out)/dz
        self.grad += out.grad * s * (1.0 - s)

    out._backward = _backward
    return out

def relu(self):
    out = Tensor(
        np.maximum(0.0, self.data),
        requires_grad=self.requires_grad,
        _prev=(self,),
        _op="relu",
    )

    def _backward():
        if out.grad is None or not self.requires_grad:
            return
        # Local derivative for ReLU:
        # d relu(z)/dz = 1 if z > 0 else 0
        self.grad += out.grad * (self.data > 0.0)

    out._backward = _backward
    return out

def sum(self, axis=None, keepdims=False):
    # Forward: collapse values by summing across selected axes.
    out = Tensor(
        self.data.sum(axis=axis, keepdims=keepdims),
        requires_grad=self.requires_grad,
        _prev=(self,),
        _op="sum",
    )

    def _backward():
        if out.grad is None or not self.requires_grad:
            return

        # Backward intuition:
        # Every input element contributes linearly to the sum with coefficient 1.
        # Therefore each input receives the upstream gradient, but we may need shape repair
        # when forward collapsed one or more axes.
        grad = out.grad
        if axis is None:
            # Sum over all entries -> same scalar gradient copied to every cell.
            grad = np.broadcast_to(grad, self.data.shape)
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            if not keepdims:
                # If forward removed dimensions, reinsert them before broadcasting.
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            # Broadcast to original input shape.
            grad = np.broadcast_to(grad, self.data.shape)
        self.grad += grad

    out._backward = _backward
    return out
""",
        )
        st.markdown(
            "These operators already cover all expressions from scalar chapters and prepare for vectorized models."
        )

    with subtabs[4]:
        st.markdown("Backward engine (graph traversal + local rules):")
        _progressive_code_block(
            key="s4_backward_engine",
            title="Show `backward()` Engine (Topological Traversal)",
            code="""
def backward(self, grad=None):
    # Goal: compute gradients for all ancestors of `self`.
    # Strategy: reverse-mode autodiff = reverse traversal of computational DAG.

    # 1) Build topological order so each node appears AFTER all its parents.
    # This ensures dependencies are resolved when we later traverse in reverse.
    topo = []
    visited = set()

    def build(node):
        # Avoid revisiting shared subgraphs.
        if id(node) in visited:
            return
        visited.add(id(node))

        # DFS into parents first (dependency-first recursion).
        for p in node._prev:
            build(p)

        # Post-order append gives parent-before-child ordering.
        topo.append(node)

    build(self)

    # 2) Seed output gradient.
    # For scalar loss L, default seed is dL/dL = 1.
    # For non-scalar outputs, caller should pass custom upstream gradient.
    if grad is None:
        grad = np.ones_like(self.data)
    self.grad += grad

    # 3) Reverse topological traversal.
    # Child nodes run before parents, so each parent has complete accumulated
    # downstream gradient by the time its own `_backward` executes.
    for node in reversed(topo):
        node._backward()
""",
        )
        st.markdown(
            "Why reverse topo? A node can push gradient to parents only after it has received all downstream contributions."
        )

        st.markdown("Minimal optimizer helpers (yes, this is where `step()` comes from):")
        _progressive_code_block(
            key="s4_opt_helpers",
            title="Show `zero_grad` and `step` Helpers",
            code="""
def zero_grad(params):
    # Framework-level utility called once per optimization iteration.
    # Because gradients accumulate (+=), not clearing would mix old and new gradients.
    for p in params:
        p.zero_grad()

def step(params, lr):
    # Minimal SGD update rule:
    # theta <- theta - lr * dL/dtheta
    # `params` can hold scalars, vectors, or matrices; operation is elementwise.
    for p in params:
        if p.grad is None:
            # Skip constants or tensors not marked for gradient tracking.
            continue
        p.data -= lr * p.grad
""",
        )
        st.markdown(
            "This is the same role as `optimizer.step()` in PyTorch. "
            "Later we can replace this with richer optimizers (momentum, Adam) without changing model/loss code."
        )

    with subtabs[5]:
        st.markdown("Why tests are essential:")
        st.markdown("- Autograd bugs are often silent (wrong values but no crash).")
        st.markdown("- Unit tests catch shape and local-rule errors quickly.")
        st.markdown("- Finite-difference checks validate analytic gradients numerically.")

        st.markdown("Example unit tests:")
        _progressive_code_block(
            key="s4_tests_unit",
            title="Show Unit Test Examples",
            code="""
def test_broadcast_add_backward():
    # Test case: bias vector is broadcast over batch rows.
    # If `_unbroadcast` is wrong, this test fails immediately.
    x = Tensor(np.ones((3, 4)), requires_grad=True)
    b = Tensor(np.ones((1, 4)), requires_grad=True)

    # Forward: y = sum(x + b)
    y = (x + b).sum()

    # Backward should recover correct gradients for both inputs.
    y.backward()

    # Since y is sum of all elements, each x_ij has local derivative 1.
    assert_allclose(x.grad, np.ones((3, 4)))

    # Bias entry b_1j is reused in 3 rows -> gradients add to 3.
    assert_allclose(b.grad, np.full((1, 4), 3.0))

def test_matmul_backward_shapes():
    # Shape consistency test for matrix multiplication gradients.
    # Wrong transpose order usually appears here as shape mismatch.
    x = Tensor(randn(5, 3), requires_grad=True)
    w = Tensor(randn(3, 2), requires_grad=True)
    out = (x @ w).mean()
    out.backward()
    assert x.grad.shape == x.data.shape
    assert w.grad.shape == w.data.shape
""",
        )

        st.markdown("Finite-difference validation:")
        _progressive_code_block(
            key="s4_tests_fd",
            title="Show Finite-Difference Gradient Check",
            code="""
def central_diff(f, w, eps=1e-5):
    # Central-difference numerical derivative:
    # f'(w) ≈ [f(w + eps) - f(w - eps)] / (2 * eps)
    # This is more accurate than forward difference for small eps.
    return (f(w + eps) - f(w - eps)) / (2 * eps)

# Compare numerical gradient vs autograd gradient at same point.
g_fd = central_diff(loss_fn, w_value)
g_auto = autograd_grad

# Scale-aware mismatch metric:
# rel_error = ||g_auto - g_fd|| / (||g_auto|| + ||g_fd|| + tiny_constant)
# Good implementations usually give very small relative error.
rel_error = np.linalg.norm(g_auto - g_fd) / (
    np.linalg.norm(g_auto) + np.linalg.norm(g_fd) + 1e-12
)
assert rel_error < 1e-6
""",
        )

        bug = st.toggle("Inject deliberate bug (missing factor 2 in MSE derivative)", value=False, key="s4_bug")
        check = check_linear_layer_grad(seed=0, introduce_bug=bug)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Relative error", f"{check['relative_error']:.2e}")
        with c2:
            st.metric("Check status", "PASS" if check["passed"] else "FAIL")

    with subtabs[6]:
        st.markdown("From manual style to framework style:")
        left, right = st.columns(2)

        with left:
            st.markdown("Before (manual derivatives):")
            _progressive_code_block(
                key="s4_old_manual",
                title="Show Manual Loop (Before Autograd)",
                code="""
# Manual training loop (single-neuron binary classifier).
# Every new model/loss pair requires re-deriving and re-implementing these gradient lines.
z = X * w + b              # Forward 1: linear logit for each sample.
p = sigmoid(z)             # Forward 2: map logit -> probability in [0, 1].
dz = p - y                 # Backward 1: dL/dz for BCE-with-sigmoid form.
dw = np.mean(dz * X)       # Backward 2: dL/dw = mean_i(dz_i * x_i).
db = np.mean(dz)           # Backward 3: dL/db = mean_i(dz_i * 1).
w -= lr * dw               # SGD update for weight parameter.
b -= lr * db               # SGD update for bias parameter.
""",
            )

        with right:
            st.markdown("After (tiny autograd):")
            _progressive_code_block(
                key="s4_new_framework",
                title="Show Autograd Loop (After Refactor)",
                code="""
# Autograd loop style:
# We only write forward expression; gradient logic lives in Tensor operators.
X_t = Tensor(X, requires_grad=False)      # Inputs are constants for this optimization step.
w_t = Tensor(w, requires_grad=True)       # Learnable weight (needs gradient).
b_t = Tensor(b, requires_grad=True)       # Learnable bias (needs gradient).
params = [w_t, b_t]                       # Centralized parameter list for optimizer utilities.

pred = (X_t * w_t + b_t).sigmoid()        # Forward graph construction happens here.
loss = ((pred - y) ** 2).mean()           # Loss graph node (scalar objective).

zero_grad(params)                         # Prevent stale gradient accumulation from prior iteration.
loss.backward()                           # Reverse traversal computes grads for all required params.
step(params, lr)                          # Parameter update; interface matches optimizer.step concept.
""",
            )

        st.markdown(
            "Benefit: the interface stays simple and extensible. You focus on model/loss expression, "
            "not re-deriving every gradient each time."
        )

    with subtabs[7]:
        st.markdown("Where is debugging applied: internally vs by you?")
        st.table(
            [
                {
                    "type": "internal (inside engine)",
                    "examples": "`_unbroadcast`, local `_backward` rules, topo traversal order",
                    "who writes it": "framework implementer",
                    "when": "once while building the engine",
                },
                {
                    "type": "external (during model training)",
                    "examples": "NaN guards, grad norms, shape checks, gradient checks",
                    "who writes it": "you (model/training code)",
                    "when": "every project/experiment",
                },
            ]
        )

        st.markdown("Symbolic formulas for debugging signals:")
        st.latex(r"g_{fd}(w) \approx \frac{f(w+\epsilon)-f(w-\epsilon)}{2\epsilon}")
        st.latex(
            r"\text{relative error} = \frac{\|g_{auto} - g_{fd}\|}{\|g_{auto}\| + \|g_{fd}\| + \delta}"
        )
        st.latex(r"\|g\|_2 = \sqrt{\sum_i g_i^2}")
        st.markdown(
            "Interpretation:\n"
            "- very small relative error means local backward rule is likely correct\n"
            "- exploding/vanishing `||g||` indicates optimization instability\n"
            "- NaN/Inf checks catch numerical failures early"
        )

        st.markdown("How to **use** `check_linear_layer_grad` in practice (not just how it is implemented):")
        _progressive_code_block(
            key="s4_gradcheck_usage",
            title="Show `check_linear_layer_grad` Usage Example",
            code="""
from mlstack.gradcheck import check_linear_layer_grad

# Step 1: Run a pre-flight gradient check before real training.
# Why: if this fails, training curves are untrustworthy no matter how long you train.
res = check_linear_layer_grad(seed=0, introduce_bug=False)
print("relative_error:", res["relative_error"])
print("passed:", res["passed"])

# Step 2: Enforce a hard gate in development workflow.
# Relative error should be tiny; otherwise local backward rule is likely incorrect.
assert res["passed"], "Fix backward implementation before training"

# Step 3: Optional lecture/demo mode.
# Intentionally inject a bug so students see a failing check and large mismatch.
res_bug = check_linear_layer_grad(seed=0, introduce_bug=True)
print("bug mode passed:", res_bug["passed"])  # expected False
print("bug mode relative_error:", res_bug["relative_error"])
""",
        )
        st.markdown("The function is defined in `mlstack/gradcheck.py` and returns:")
        st.markdown("- `relative_error`: numeric mismatch score")
        st.markdown("- `passed`: boolean threshold check")

        st.markdown("Training-loop debug hooks you should actively add:")
        _progressive_code_block(
            key="s4_debug_patterns",
            title="Show Training-Time Debug Pattern",
            code="""
from mlstack.gradcheck import check_linear_layer_grad

# One-time pre-flight: verify local backward rules before expensive runs.
gradcheck = check_linear_layer_grad(seed=0, introduce_bug=False)
assert gradcheck["passed"], f"Gradcheck failed: {gradcheck['relative_error']:.2e}"

for step in range(num_steps):
    # Forward pass: compute predictions and scalar objective.
    pred = model(X)
    loss = compute_loss(pred, y)

    # 1) Reset stale grads from previous iteration.
    #    Without this, gradients accumulate across steps and updates become incorrect.
    zero_grad(params)

    # 2) Backward pass through computational graph.
    loss.backward()

    # 3) External runtime checks (project code, not framework internals).
    #    Catch numerical failure immediately instead of discovering it much later.
    assert np.isfinite(loss.data).all(), "loss has NaN/Inf"
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            assert np.isfinite(p.grad).all(), "gradient has NaN/Inf"
            total_norm += np.sum(p.grad ** 2)
    total_norm = np.sqrt(total_norm)

    # 4) Optional logging cadence for diagnostics and lecture visualization.
    if step % 20 == 0:
        print(f"step={step} loss={float(loss.data):.4f} grad_norm={total_norm:.4f}")

    # 5) Parameter update.
    step(params, lr)
""",
        )
        st.markdown("Use this pattern directly in assignments/projects to avoid silent failure modes.")

def _render_stage_pytorch_mirror() -> None:
    st.subheader("Stage 5: Introduce PyTorch Early")
    torch_ready = importlib.util.find_spec("torch") is not None
    if torch_ready:
        st.success("PyTorch is available. Live demos in this chapter will execute locally on CPU.")
    else:
        st.warning(
            "PyTorch is not installed in this environment yet. Concept sections still work. "
            "For live PyTorch demos, run `./setup_env.sh` and refresh."
        )

    subtabs = st.tabs(
        [
            "5A) Why PyTorch Now",
            "5B) Same Core, New API",
            "5C) Tensor Basics",
            "5D) Autograd Parity Check",
            "5E) Modules and Loop",
            "5F) Batching with DataLoader",
            "5G) PyTorch Debugging",
            "5H) Live Training Demo",
            "5I) Bridge to Next Chapters",
        ]
    )

    with subtabs[0]:
        st.markdown(
            "You already built the core ideas manually: forward pass, backward pass, and parameter updates. "
            "PyTorch does **not** replace those ideas; it packages them into a reliable system."
        )
        st.markdown("Why switch now:")
        st.markdown("- Your tiny engine gives intuition but has limited operator coverage.")
        st.markdown("- PyTorch gives tested operators, modules, optimizers, and data pipeline tools.")
        st.markdown("- You can move faster while keeping the same mathematical mental model.")
        st.table(
            [
                {"need in real projects": "many operators", "tiny engine": "add by hand", "PyTorch": "already implemented"},
                {"need in real projects": "fast execution", "tiny engine": "Python-level overhead", "PyTorch": "optimized kernels"},
                {"need in real projects": "safe training loop", "tiny engine": "you write everything", "PyTorch": "standard patterns"},
                {"need in real projects": "ecosystem reuse", "tiny engine": "local only", "PyTorch": "large ecosystem"},
            ]
        )

    with subtabs[1]:
        st.markdown("Concept continuity (this is the key point):")
        st.latex(r"\theta \leftarrow \theta - \eta \nabla_\theta L")
        st.markdown(
            "The update rule above is unchanged. What changes is **how much boilerplate you must write** "
            "to compute gradients and manage parameters."
        )
        st.table(
            [
                {"concept": "tracked value", "tiny engine": "`Tensor(data, grad, _prev)`", "PyTorch": "`torch.Tensor`"},
                {"concept": "graph build", "tiny engine": "operator overloads", "PyTorch": "operator overloads"},
                {"concept": "backprop", "tiny engine": "`loss.backward()`", "PyTorch": "`loss.backward()`"},
                {"concept": "clear gradients", "tiny engine": "`zero_grad(params)`", "PyTorch": "`optimizer.zero_grad()`"},
                {"concept": "update", "tiny engine": "`step(params, lr)`", "PyTorch": "`optimizer.step()`"},
            ]
        )

        left, right = st.columns(2)
        with left:
            st.markdown("Tiny autograd style")
            _progressive_code_block(
                key="s5_tiny_loop",
                title="Show Tiny Autograd Training Step",
                code="""
# Tiny engine from previous chapter.
X_t = Tensor(X, requires_grad=False)      # Data tensor (no gradient needed).
w_t = Tensor(w, requires_grad=True)       # Learnable parameter.
b_t = Tensor(b, requires_grad=True)       # Learnable parameter.
params = [w_t, b_t]

pred = (X_t * w_t + b_t).sigmoid()        # Build forward graph.
loss = ((pred - y) ** 2).mean()           # Scalar objective.

zero_grad(params)                         # Clear old gradients.
loss.backward()                           # Reverse graph traversal computes grads.
step(params, lr)                          # SGD update.
""",
            )
        with right:
            st.markdown("PyTorch style")
            _progressive_code_block(
                key="s5_torch_loop",
                title="Show PyTorch Equivalent Training Step",
                code="""
import torch
import torch.nn.functional as F

model = torch.nn.Linear(1, 1)                 # Parameter container + forward op.
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

logits = model(x)                              # Forward pass.
loss = F.binary_cross_entropy_with_logits(logits, y)  # Stable BCE for logits.

optimizer.zero_grad()                          # Clear old grads on parameters.
loss.backward()                                # Autograd computes gradients.
optimizer.step()                               # Optimizer updates parameters.
""",
            )

    with subtabs[2]:
        st.markdown(
            "Before full models, practice with raw tensors: shape, dtype, `requires_grad`, and broadcasting. "
            "These are the most common sources of beginner bugs."
        )
        _progressive_code_block(
            key="s5_tensor_basics_code",
            title="Show Basic Tensor Sandbox Code",
            code="""
import torch

# Build a tensor with explicit dtype.
x = torch.linspace(-1.5, 1.5, steps=6, dtype=torch.float32).reshape(2, 3)
x = x.clone().detach().requires_grad_(True)   # Track gradients on x.

# Compose a simple expression.
y = ((1.8 * x + 0.4) ** 2).mean()

# Backprop to get dy/dx.
y.backward()
print("x shape:", x.shape)
print("y scalar:", y.item())
print("x.grad shape:", x.grad.shape)
""",
            )

        if torch_ready:
            import torch

            col1, col2, col3 = st.columns(3)
            with col1:
                rows = st.slider("rows", 1, 5, 2, 1, key="s5_rows")
            with col2:
                cols = st.slider("cols", 1, 6, 3, 1, key="s5_cols")
            with col3:
                dtype_name = st.selectbox("dtype", ["float32", "float64"], index=0, key="s5_dtype")

            req_grad = st.toggle("Track gradients (`requires_grad=True`)", value=True, key="s5_req_grad")
            scale = st.slider("scale in y = ((scale*x + shift)^2).mean()", 0.5, 3.0, 1.8, 0.1, key="s5_scale")
            shift = st.slider("shift in y = ((scale*x + shift)^2).mean()", -2.0, 2.0, 0.4, 0.1, key="s5_shift")

            dtype = torch.float32 if dtype_name == "float32" else torch.float64
            base = torch.linspace(-1.5, 1.5, steps=rows * cols, dtype=dtype).reshape(rows, cols)
            x_t = base.clone().detach().requires_grad_(req_grad)
            y_t = ((scale * x_t + shift) ** 2).mean()

            if req_grad:
                y_t.backward()

            left, right = st.columns(2)
            with left:
                st.write("Tensor values")
                st.write(x_t.detach().cpu().numpy())
                st.caption(f"shape={tuple(x_t.shape)}, dtype={x_t.dtype}, requires_grad={x_t.requires_grad}")
            with right:
                st.metric("Output scalar y", f"{float(y_t.detach().cpu().item()):.6f}")
                if req_grad:
                    st.write("Gradient dy/dx")
                    st.write(x_t.grad.detach().cpu().numpy())
                else:
                    st.info("Gradient is not tracked because `requires_grad=False`.")
        else:
            st.info("Install PyTorch to run the tensor sandbox interactively.")

    with subtabs[3]:
        st.markdown(
            "Now verify that PyTorch autograd returns the same derivatives you derived manually "
            "for the one-sample scalar neuron."
        )
        st.markdown("Manual pipeline being checked:")
        st.latex(r"z = wx+b,\quad p=\sigma(z),\quad L\in\{\text{MSE},\text{BCE}\}")
        st.latex(r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z}\frac{\partial z}{\partial w},\quad \frac{\partial z}{\partial w}=x")

        c1, c2, c3 = st.columns(3)
        with c1:
            x_val = st.slider("x", -3.0, 3.0, 1.2, 0.1, key="s5_chk_x")
        with c2:
            w_val = st.slider("w", -3.0, 3.0, -0.7, 0.1, key="s5_chk_w")
        with c3:
            b_val = st.slider("b", -2.0, 2.0, 0.2, 0.1, key="s5_chk_b")

        c4, c5 = st.columns(2)
        with c4:
            y_val = st.selectbox("label y", [0.0, 1.0], index=1, key="s5_chk_y")
        with c5:
            loss_name = st.selectbox("loss", ["mse", "bce"], index=1, key="s5_chk_loss")

        manual = _scalar_example_details(x=x_val, y=y_val, w=w_val, b=b_val, loss_name=loss_name)
        if torch_ready:
            comp = run_torch_scalar_autograd_demo(x=x_val, y=y_val, w=w_val, b=b_val, loss_name=loss_name)
        else:
            comp = None

        if comp is None:
            st.warning("PyTorch is required for parity check execution. Showing manual values only.")
            st.write(
                {
                    "z": round(manual["z"], 6),
                    "p": round(manual["p"], 6),
                    "loss": round(manual["loss"], 6),
                    "dL/dz": round(manual["dL_dz"], 6),
                    "dL/dw": round(manual["dL_dw"], 6),
                    "dL/db": round(manual["dL_db"], 6),
                }
            )
        else:
            rows = []
            for key, label in [
                ("z", "z"),
                ("p", "p"),
                ("loss", "L"),
                ("dL_dz", "dL/dz"),
                ("dL_dw", "dL/dw"),
                ("dL_db", "dL/db"),
            ]:
                rows.append(
                    {
                        "quantity": label,
                        "manual": f"{comp['manual'][key]:.8f}",
                        "pytorch": f"{comp['torch'][key]:.8f}",
                        "abs error": f"{comp['abs_errors'][key]:.2e}",
                    }
                )
            st.table(rows)

            max_abs = max(comp["abs_errors"].values())
            st.metric("Max absolute error", f"{max_abs:.2e}")

            fig, ax = plt.subplots(figsize=(6.8, 3.8))
            keys = ["dL_dz", "dL_dw", "dL_db"]
            vals = [comp["abs_errors"][k] for k in keys]
            ax.bar(["|dL/dz| err", "|dL/dw| err", "|dL/db| err"], vals, color=["#93c5fd", "#60a5fa", "#2563eb"])
            ax.set_title("Manual vs PyTorch gradient mismatch")
            ax.set_ylabel("absolute error")
            st.pyplot(fig)

        _progressive_code_block(
            key="s5_autograd_parity_code",
            title="Show PyTorch Autograd Parity Code",
            code="""
import torch

# Scalar values from sliders.
x = torch.tensor(x_val, dtype=torch.float64)
y = torch.tensor(y_val, dtype=torch.float64)
w = torch.tensor(w_val, dtype=torch.float64, requires_grad=True)
b = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)

z = w * x + b
p = torch.sigmoid(z)

if loss_name == "mse":
    loss = (p - y) ** 2
else:
    p_safe = torch.clamp(p, 1e-12, 1.0 - 1e-12)
    loss = -(y * torch.log(p_safe) + (1.0 - y) * torch.log(1.0 - p_safe))

# Optional: direct gradient for intermediate node z.
dL_dz = torch.autograd.grad(loss, z, retain_graph=True)[0]

# Backprop to parameters.
loss.backward()
print("dL/dw:", w.grad.item(), "dL/db:", b.grad.item())
""",
            )

    with subtabs[4]:
        st.markdown(
            "Next abstraction: `nn.Module` packages parameters and forward logic, "
            "so model definition is reusable and clean."
        )
        _progressive_code_block(
            key="s5_module_code",
            title="Show Minimal `nn.Module` for Single Neuron",
            code="""
import torch
import torch.nn as nn

class SingleNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Includes weight and bias tensors.

    def forward(self, x):
        # Return logits; keep sigmoid outside for numerically stable BCEWithLogits.
        return self.linear(x)
""",
        )
        _progressive_code_block(
            key="s5_train_step_code",
            title="Show Canonical PyTorch Training Loop Skeleton",
            code="""
import torch.nn.functional as F

model = SingleNeuron()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for step in range(num_steps):
    model.train()                                         # Training mode (important when dropout/bn appear).
    logits = model(x_batch)                               # Forward pass.
    loss = F.binary_cross_entropy_with_logits(logits, y_batch)

    optimizer.zero_grad()                                 # Clear stale gradients.
    loss.backward()                                       # Autograd computes gradients.
    optimizer.step()                                      # Parameter update.
""",
        )
        st.table(
            [
                {"chapter 4 (tiny engine)": "`Tensor` + operator overloads", "PyTorch": "`torch.Tensor` + autograd"},
                {"chapter 4 (tiny engine)": "manual parameter list", "PyTorch": "`model.parameters()`"},
                {"chapter 4 (tiny engine)": "`step(params, lr)`", "PyTorch": "optimizer object (`SGD`, `Adam`)"},
                {"chapter 4 (tiny engine)": "custom debug checks", "PyTorch": "same checks still needed"},
            ]
        )

    with subtabs[5]:
        st.markdown(
            "We now add batching. Same model and loss, but data arrives in mini-batches from a `DataLoader`."
        )
        st.latex(r"\nabla_\theta L_{\text{batch}} = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta L_i")
        _progressive_code_block(
            key="s5_dataloader_code",
            title="Show Dataset + DataLoader Pattern",
            code="""
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for xb, yb in loader:
    logits = model(xb)
    loss = F.binary_cross_entropy_with_logits(logits, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""",
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            seed = st.number_input("seed", min_value=0, max_value=1000, value=11, step=1, key="s5_batch_seed")
        with c2:
            lr = st.slider("learning rate", 0.005, 0.4, 0.08, 0.005, key="s5_batch_lr")
        with c3:
            epochs = st.slider("epochs", 10, 160, 60, 5, key="s5_batch_epochs")
        with c4:
            mini_batch_size = st.slider("mini batch size", 4, 128, 16, 4, key="s5_batch_size")

        batch_out = run_torch_batch_compare(
            seed=int(seed),
            lr=float(lr),
            epochs=int(epochs),
            mini_batch_size=int(mini_batch_size),
        )

        if batch_out is None:
            st.info("Install PyTorch to run the mini-batch vs full-batch comparison.")
        else:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.plot(batch_out["mini_losses"], label=f"mini-batch ({batch_out['mini_batch_size']})", linewidth=2.0)
            ax.plot(batch_out["full_losses"], label="full-batch", linewidth=2.0)
            ax.set_title("Epoch loss: mini-batch vs full-batch")
            ax.set_xlabel("epoch")
            ax.set_ylabel("BCE loss")
            ax.legend()
            st.pyplot(fig)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Mini-batch final accuracy", f"{100.0 * batch_out['mini_acc']:.1f}%")
            with m2:
                st.metric("Full-batch final accuracy", f"{100.0 * batch_out['full_acc']:.1f}%")

            st.markdown(
                "Interpretation: mini-batch updates are noisier but usually faster per epoch on larger datasets; "
                "full-batch is smoother but less scalable."
            )

    with subtabs[6]:
        st.markdown(
            "PyTorch removes boilerplate, but debugging discipline remains your responsibility. "
            "Use this checklist every time you train."
        )
        st.markdown("- Verify one batch overfits (bug detector).")
        st.markdown("- Log loss and gradient norm.")
        st.markdown("- Check `NaN`/`Inf` in loss and gradients.")
        st.markdown("- Match model output shape to label shape.")
        st.markdown("- Use `model.train()` and `model.eval()` correctly.")
        _progressive_code_block(
            key="s5_debug_code",
            title="Show PyTorch Debug-Safe Training Template",
            code="""
import torch
import torch.nn.functional as F

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for step, (xb, yb) in enumerate(loader):
    model.train()
    logits = model(xb)
    loss = F.binary_cross_entropy_with_logits(logits, yb)

    optimizer.zero_grad()
    loss.backward()

    # Runtime guards: fail fast when numerics break.
    assert torch.isfinite(loss).all(), "loss has NaN/Inf"
    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "gradient has NaN/Inf"
            grad_sq += float(torch.sum(p.grad * p.grad).item())
    grad_norm = grad_sq ** 0.5

    if step % 20 == 0:
        print(f"step={step} loss={loss.item():.4f} grad_norm={grad_norm:.4f}")

    optimizer.step()
""",
        )

    with subtabs[7]:
        st.markdown("Live local CPU demo: PyTorch logistic regression on the same 1D dataset.")
        c1, c2, c3 = st.columns(3)
        with c1:
            seed = st.number_input("seed", min_value=0, max_value=1000, value=11, step=1, key="s5_live_seed")
        with c2:
            lr = st.slider("lr", 0.001, 1.0, 0.08, 0.001, key="s5_live_lr")
        with c3:
            steps = st.slider("steps", 40, 700, 260, 20, key="s5_live_steps")

        out = run_torch_scalar(seed=int(seed), lr=lr, steps=steps)
        if out is None:
            st.warning("PyTorch is not installed in this environment. Run `./setup_env.sh` and refresh.")
        else:
            top_left, top_right = st.columns([1.2, 1.0])
            with top_left:
                fig, ax = plt.subplots(figsize=(7.4, 4.4))
                _plot_scalar_prob_curve_on_ax(ax, out["x"], out["y"], out["w"], out["b"], title="PyTorch 1D classifier")
                st.pyplot(fig)
            with top_right:
                fig2, ax2 = plt.subplots(figsize=(6.4, 3.2))
                plot_curve(ax2, out["losses"], "PyTorch BCE loss", "loss")
                st.pyplot(fig2)

                fig3, ax3 = plt.subplots(figsize=(6.4, 3.2))
                plot_curve(ax3, out["grad_norms"], "Gradient norm", "||g||")
                st.pyplot(fig3)

            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("Final accuracy", f"{100.0 * out['acc']:.1f}%")
            with b2:
                st.metric("Final loss", f"{out['losses'][-1]:.4f}")
            with b3:
                st.metric("Final grad norm", f"{out['grad_norms'][-1]:.4f}")

    with subtabs[8]:
        st.markdown("What this chapter prepared you for:")
        st.markdown("- Later networks are still forward graph + backward + optimizer step.")
        st.markdown("- `nn.Module`, optimizers, and `DataLoader` are the three APIs you will reuse constantly.")
        st.markdown("- Debugging habits from Chapter 4 transfer directly to PyTorch.")

def _render_stage_multivariable() -> None:
    st.subheader("Stage 6: Move to Multivariable (Step-by-Step)")
    subtabs = st.tabs(
        [
            "6A) Recall and Motivation",
            "6B) Two Features by Hand",
            "6C) Batch Matrix View",
            "6D) NumPy Implementation",
            "6E) Runtime Motivation",
            "6F) 2D Decision Visual",
            "6G) Pitfalls and Checks",
        ]
    )

    with subtabs[0]:
        st.markdown(
            "Recall from previous chapters: we computed forward pass to get loss, then backward pass for gradients. "
            "That logic does not change in multivariable settings."
        )
        st.markdown("What changes is notation and shape management.")
        st.table(
            [
                {"scalar chapter": r"`x \in R`", "multivariable chapter": r"`x \in R^D`", "interpretation": "one sample now has D features"},
                {"scalar chapter": r"`w \in R`", "multivariable chapter": r"`w \in R^D`", "interpretation": "one weight per feature"},
                {"scalar chapter": r"`z = wx+b`", "multivariable chapter": r"`z = w^Tx + b`", "interpretation": "dot product replaces scalar multiply"},
                {"scalar chapter": "single sample", "multivariable chapter": r"`X \in R^{B x D}`", "interpretation": "B samples processed as a batch"},
            ]
        )
        st.markdown("Forward equation progression:")
        st.latex(r"z = wx+b \;\;\rightarrow\;\; z = w^\top x + b \;\;\rightarrow\;\; Z = XW + b")

    with subtabs[1]:
        st.markdown(
            "Start with **one sample, two features**. This is the smallest meaningful jump beyond scalar math."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            x1 = st.slider("x1", -3.0, 3.0, 1.2, 0.1, key="s6_x1")
            w1 = st.slider("w1", -3.0, 3.0, 0.9, 0.1, key="s6_w1_hand")
        with c2:
            x2 = st.slider("x2", -3.0, 3.0, -0.8, 0.1, key="s6_x2")
            w2 = st.slider("w2", -3.0, 3.0, -1.1, 0.1, key="s6_w2_hand")
        with c3:
            b = st.slider("b", -2.0, 2.0, 0.2, 0.1, key="s6_b_hand")

        z = w1 * x1 + w2 * x2 + b
        p = float(_sigmoid_np(np.array(z)))
        st.latex(
            rf"z = w_1x_1 + w_2x_2 + b = ({w1:.2f})({x1:.2f}) + ({w2:.2f})({x2:.2f}) + ({b:.2f}) = {z:.4f}"
        )
        st.latex(rf"p = \sigma(z) = \frac{{1}}{{1+e^{{-z}}}} = {p:.4f}")
        st.markdown(
            "Intuition: each feature contributes `w_i x_i`; positive weight reinforces the feature, "
            "negative weight suppresses it."
        )

    with subtabs[2]:
        st.markdown("Now stack samples into a matrix and compute all logits at once.")
        st.latex(r"X \in \mathbb{R}^{B\times D},\; W \in \mathbb{R}^{D\times 1},\; b \in \mathbb{R}^{1\times 1},\; Z = XW + b")

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        w = np.array([[0.2], [0.5]], dtype=np.float64)
        b = np.array([[0.1]], dtype=np.float64)
        z = x @ w + b

        st.write("X =", x)
        st.write("W =", w)
        st.write("b =", b)
        st.write("Z = X @ W + b =", z)

        st.markdown("Row-wise interpretation (sample 2):")
        st.latex(
            rf"z_2 = x_{{21}}w_1 + x_{{22}}w_2 + b = (3.0)(0.2) + (4.0)(0.5) + 0.1 = {float(z[1, 0]):.4f}"
        )
        st.markdown("Matrix multiplication computes all rows in one operation; bias is broadcast to each row.")

    with subtabs[3]:
        st.markdown("Translate math directly into NumPy code. First loop version, then vectorized version.")
        _progressive_code_block(
            key="s6_loop_vs_vec_code",
            title="Show Loop vs Vectorized Forward Pass",
            code="""
# Loop implementation (explicit and readable).
out_loop = loop_forward(X, W, b)

# Vectorized implementation (same math, faster).
out_vec = X @ W + b

# Always verify equivalence when refactoring.
assert np.allclose(out_loop, out_vec)
""",
        )
        _progressive_code_block(
            key="s6_batch_grad_code",
            title="Show Batch Gradient Formulas in Code",
            code="""
# For binary logistic neuron with batch dimension B:
# logits = X @ w + b
# probs = sigmoid(logits)
# dz = probs - y                 # shape: (B, 1)
# dw = (X.T @ dz) / B            # shape: (D, 1)
# db = mean(dz)                  # scalar

B = X.shape[0]
dz = probs - y
dw = (X.T @ dz) / B
db = np.mean(dz)
""",
        )

        x_small = np.array([[1.0, 2.0, -1.0], [0.5, -0.2, 1.1], [2.3, 0.7, -0.4]], dtype=np.float64)
        w_small = np.array([[0.2, -0.1], [0.5, 0.3], [-0.4, 0.8]], dtype=np.float64)
        b_small = np.array([[0.1, -0.2]], dtype=np.float64)

        loop_out = loop_forward(x_small, w_small, b_small)
        vec_out = vectorized_forward(x_small, w_small, b_small)
        st.metric("max |loop - vectorized|", f"{np.abs(loop_out - vec_out).max():.2e}")
        st.caption("Near-zero mismatch confirms the vectorized version is mathematically identical.")

    with subtabs[4]:
        st.markdown("Why we insist on vectorization: runtime scales much better.")
        c1, c2, c3 = st.columns(3)
        with c1:
            batch = st.slider("batch size", 128, 4096, 1024, 128, key="s6_batch")
        with c2:
            in_dim = st.slider("input dim", 8, 256, 64, 8, key="s6_in")
        with c3:
            out_dim = st.slider("output dim", 8, 128, 32, 8, key="s6_out")

        bench = run_vector_benchmark(batch=batch, in_dim=in_dim, out_dim=out_dim)
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        ax.bar(["Loop", "Vectorized"], [bench["loop_ms"], bench["vectorized_ms"]], color=["#a5b4fc", "#34d399"])
        ax.set_ylabel("milliseconds")
        ax.set_title("Forward pass runtime")
        for idx, val in enumerate([bench["loop_ms"], bench["vectorized_ms"]]):
            ax.text(idx, val * 1.02, f"{val:.2f}", ha="center")
        st.pyplot(fig)
        st.metric("Speedup", f"{bench['speedup']:.1f}x")

    with subtabs[5]:
        st.markdown("Visual intuition: two-feature neuron creates a linear decision boundary in 2D.")
        x_lin, y_lin = get_linear_data(seed=0)
        c1, c2, c3 = st.columns(3)
        with c1:
            w1 = st.slider("boundary w1", -4.0, 4.0, 1.2, 0.05, key="s6_w1")
        with c2:
            w2 = st.slider("boundary w2", -4.0, 4.0, 1.2, 0.05, key="s6_w2")
        with c3:
            b = st.slider("boundary b", -3.0, 3.0, 0.0, 0.05, key="s6_b")

        w_demo = np.array([[w1], [w2]], dtype=np.float64)
        fig = _plot_single_neuron_surface(x_lin, y_lin, w_demo, b, title="Two-feature neuron decision boundary")
        st.pyplot(fig)
        st.markdown("Move one slider at a time and observe whether boundary rotates (`w1`, `w2`) or shifts (`b`).")

    with subtabs[6]:
        st.markdown("Most multivariable bugs are shape/broadcast issues, not math mistakes.")
        st.markdown("High-value checks:")
        st.markdown("- Assert expected shapes for logits, labels, and gradients.")
        st.markdown("- Check that vectorized and loop outputs match on tiny arrays.")
        st.markdown("- Log `max_abs_diff` when refactoring code.")
        st.markdown("- Use small deterministic test tensors before training on full data.")
        _progressive_code_block(
            key="s6_shape_checks_code",
            title="Show Shape and Broadcasting Guardrails",
            code="""
def safe_forward(X, W, b):
    # Shape guards prevent silent broadcasting bugs.
    assert X.ndim == 2, "X must be (B, D)"
    assert W.ndim == 2, "W must be (D, H)"
    assert b.ndim == 2 and b.shape[0] == 1, "b must be (1, H)"
    assert X.shape[1] == W.shape[0], "feature dimension mismatch"
    assert b.shape[1] == W.shape[1], "output dimension mismatch"

    out = X @ W + b
    assert out.shape == (X.shape[0], W.shape[1])
    return out

# Refactor safety check:
out_loop = loop_forward(X, W, b)
out_vec = safe_forward(X, W, b)
max_abs_diff = np.max(np.abs(out_loop - out_vec))
assert max_abs_diff < 1e-10
""",
        )

def _render_stage_end_to_end() -> None:
    st.subheader("Stage 7: End-to-End Compare (Tiny Framework and PyTorch)")
    torch_ready = importlib.util.find_spec("torch") is not None
    if torch_ready:
        st.success("PyTorch is available: all cross-framework demos will run.")
    else:
        st.warning(
            "PyTorch is not installed in this environment. Tiny-framework demos still run. "
            "Install PyTorch to enable side-by-side runtime comparisons."
        )

    subtabs = st.tabs(
        [
            "7A) Objective",
            "7B) Pipeline Map",
            "7C) Linear Task (Both)",
            "7D) Moons: Linear Fails (Both)",
            "7E) Moons: MLP Works (Both)",
            "7F) Implementation Code",
            "7G) Debug and Portfolio",
        ]
    )

    with subtabs[0]:
        st.markdown(
            "This chapter closes the loop: train full models end-to-end using **our tiny framework** and **PyTorch** "
            "on the same datasets, with the same optimization knobs."
        )
        st.markdown("Learning goals:")
        st.markdown("- Confirm that framework choice changes API, not core optimization math.")
        st.markdown("- See linear-model limits on nonlinear data.")
        st.markdown("- See how adding hidden nonlinear layers changes decision boundaries.")
        st.table(
            [
                {"experiment": "linear data + linear model", "expected result": "both frameworks fit well"},
                {"experiment": "moons data + linear model", "expected result": "both frameworks underfit"},
                {"experiment": "moons data + MLP", "expected result": "both frameworks achieve high accuracy"},
            ]
        )

    with subtabs[1]:
        st.markdown("Shared training pipeline (identical conceptually in both frameworks):")
        st.latex(r"\text{forward} \rightarrow \text{loss} \rightarrow \text{backward} \rightarrow \text{update}")
        st.table(
            [
                {"step": "build model", "tiny framework": "`Sequential(Linear, ReLU, ...)`", "PyTorch": "`torch.nn.Sequential(...)`"},
                {"step": "forward", "tiny framework": "`logits = model(Tensor(x))`", "PyTorch": "`logits = model(x_tensor)`"},
                {"step": "loss", "tiny framework": "`binary_cross_entropy_with_logits`", "PyTorch": "`F.binary_cross_entropy_with_logits`"},
                {"step": "clear grads", "tiny framework": "`optimizer.zero_grad()`", "PyTorch": "`optimizer.zero_grad()`"},
                {"step": "backward", "tiny framework": "`loss.backward()`", "PyTorch": "`loss.backward()`"},
                {"step": "update", "tiny framework": "`optimizer.step()`", "PyTorch": "`optimizer.step()`"},
            ]
        )
        st.markdown(
            "Interpretation: once fundamentals are understood, moving between frameworks is mostly a syntax change."
        )

    with subtabs[2]:
        st.markdown("Experiment 1: linearly separable data + linear model in both frameworks.")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            opt_name = st.selectbox("optimizer", ["adam", "sgd", "momentum"], index=1, key="s7_lin_opt")
        with c2:
            lr = st.slider("lr", 0.005, 0.5, 0.08, 0.005, key="s7_lin_lr")
        with c3:
            steps = st.slider("steps", 40, 600, 220, 20, key="s7_lin_steps")
        with c4:
            batch_size = st.slider("batch size", 8, 256, 64, 8, key="s7_lin_bsz")
        with c5:
            seed = st.number_input("seed", min_value=0, max_value=1000, value=0, step=1, key="s7_lin_seed")

        tiny = run_tiny_end_to_end_demo(
            dataset_name="linear",
            model_kind="linear",
            hidden_dim=8,
            lr=float(lr),
            steps=int(steps),
            optimizer_name=opt_name,
            batch_size=int(batch_size),
            seed=int(seed),
        )
        torch_out = run_torch_end_to_end_demo(
            dataset_name="linear",
            model_kind="linear",
            hidden_dim=8,
            lr=float(lr),
            steps=int(steps),
            optimizer_name=opt_name,
            batch_size=int(batch_size),
            seed=int(seed),
        )

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(6.4, 4.9))
            plot_decision_surface(
                ax,
                tiny["x"],
                tiny["y"],
                tiny["xx"],
                tiny["yy"],
                tiny["probs_grid"],
                title="Tiny framework: linear model",
            )
            st.pyplot(fig)
        with right:
            if torch_out is None:
                st.info("Install PyTorch to render the right-side comparison.")
            else:
                fig, ax = plt.subplots(figsize=(6.4, 4.9))
                plot_decision_surface(
                    ax,
                    torch_out["x"],
                    torch_out["y"],
                    torch_out["xx"],
                    torch_out["yy"],
                    torch_out["probs_grid"],
                    title="PyTorch: linear model",
                )
                st.pyplot(fig)

        fig2, axes = plt.subplots(3, 1, figsize=(7.4, 7.4))
        axes[0].plot(tiny["losses"], label="tiny", linewidth=2.0)
        if torch_out is not None:
            axes[0].plot(torch_out["losses"], label="torch", linewidth=2.0)
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(tiny["accuracies"], label="tiny", linewidth=2.0)
        if torch_out is not None:
            axes[1].plot(torch_out["accuracies"], label="torch", linewidth=2.0)
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[2].plot(tiny["grad_norms"], label="tiny", linewidth=2.0)
        if torch_out is not None:
            axes[2].plot(torch_out["grad_norms"], label="torch", linewidth=2.0)
        axes[2].set_title("Gradient Norm")
        axes[2].legend()
        fig2.tight_layout()
        st.pyplot(fig2)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Tiny final accuracy", f"{100.0 * tiny['train_acc']:.1f}%")
        with m2:
            if torch_out is None:
                st.metric("PyTorch final accuracy", "N/A")
            else:
                st.metric("PyTorch final accuracy", f"{100.0 * torch_out['train_acc']:.1f}%")

    with subtabs[3]:
        st.markdown("Experiment 2: two-moons data + linear model in both frameworks.")
        st.markdown("Expectation: both frameworks struggle because model capacity is linear.")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            opt_name = st.selectbox("optimizer", ["adam", "sgd", "momentum"], index=0, key="s7_moon_lin_opt")
        with c2:
            lr = st.slider("lr", 0.001, 0.3, 0.05, 0.001, key="s7_moon_lin_lr")
        with c3:
            steps = st.slider("steps", 80, 800, 320, 20, key="s7_moon_lin_steps")
        with c4:
            batch_size = st.slider("batch size", 8, 256, 64, 8, key="s7_moon_lin_bsz")
        with c5:
            seed = st.number_input("seed", min_value=0, max_value=1000, value=7, step=1, key="s7_moon_lin_seed")

        tiny = run_tiny_end_to_end_demo(
            dataset_name="moons",
            model_kind="linear",
            hidden_dim=8,
            lr=float(lr),
            steps=int(steps),
            optimizer_name=opt_name,
            batch_size=int(batch_size),
            seed=int(seed),
        )
        torch_out = run_torch_end_to_end_demo(
            dataset_name="moons",
            model_kind="linear",
            hidden_dim=8,
            lr=float(lr),
            steps=int(steps),
            optimizer_name=opt_name,
            batch_size=int(batch_size),
            seed=int(seed),
        )

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(6.4, 4.9))
            plot_decision_surface(
                ax,
                tiny["x"],
                tiny["y"],
                tiny["xx"],
                tiny["yy"],
                tiny["probs_grid"],
                title="Tiny framework: linear model on moons",
            )
            st.pyplot(fig)
        with right:
            if torch_out is None:
                st.info("Install PyTorch to render the right-side comparison.")
            else:
                fig, ax = plt.subplots(figsize=(6.4, 4.9))
                plot_decision_surface(
                    ax,
                    torch_out["x"],
                    torch_out["y"],
                    torch_out["xx"],
                    torch_out["yy"],
                    torch_out["probs_grid"],
                    title="PyTorch: linear model on moons",
                )
                st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(7.2, 3.9))
        ax2.plot(tiny["accuracies"], label="tiny", linewidth=2.0)
        if torch_out is not None:
            ax2.plot(torch_out["accuracies"], label="torch", linewidth=2.0)
        ax2.set_title("Accuracy on two-moons with linear model")
        ax2.set_xlabel("training step")
        ax2.set_ylabel("accuracy")
        ax2.legend()
        st.pyplot(fig2)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Tiny linear accuracy", f"{100.0 * tiny['train_acc']:.1f}%")
        with m2:
            if torch_out is None:
                st.metric("PyTorch linear accuracy", "N/A")
            else:
                st.metric("PyTorch linear accuracy", f"{100.0 * torch_out['train_acc']:.1f}%")

        st.markdown(
            "Interpretation: weak result here is a **model class limitation**, not a framework issue."
        )

    with subtabs[4]:
        st.markdown("Experiment 3: two-moons data + MLP in both frameworks.")
        st.markdown("Expectation: nonlinear hidden layer allows curved decision boundaries.")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            hidden_dim = st.slider("hidden dim", 4, 64, 16, 2, key="s7_mlp_h")
        with c2:
            opt_name = st.selectbox("optimizer", ["adam", "sgd", "momentum"], index=0, key="s7_mlp_opt")
        with c3:
            lr = st.slider("lr", 0.001, 0.3, 0.05, 0.001, key="s7_mlp_lr")
        with c4:
            steps = st.slider("steps", 100, 900, 360, 20, key="s7_mlp_steps")
        with c5:
            batch_size = st.slider("batch size", 8, 256, 64, 8, key="s7_mlp_bsz")
        with c6:
            seed = st.number_input("seed", min_value=0, max_value=1000, value=7, step=1, key="s7_mlp_seed")

        tiny = run_tiny_end_to_end_demo(
            dataset_name="moons",
            model_kind="mlp",
            hidden_dim=int(hidden_dim),
            lr=float(lr),
            steps=int(steps),
            optimizer_name=opt_name,
            batch_size=int(batch_size),
            seed=int(seed),
        )
        torch_out = run_torch_end_to_end_demo(
            dataset_name="moons",
            model_kind="mlp",
            hidden_dim=int(hidden_dim),
            lr=float(lr),
            steps=int(steps),
            optimizer_name=opt_name,
            batch_size=int(batch_size),
            seed=int(seed),
        )

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(6.4, 4.9))
            plot_decision_surface(
                ax,
                tiny["x"],
                tiny["y"],
                tiny["xx"],
                tiny["yy"],
                tiny["probs_grid"],
                title="Tiny framework: MLP on moons",
            )
            st.pyplot(fig)
        with right:
            if torch_out is None:
                st.info("Install PyTorch to render the right-side comparison.")
            else:
                fig, ax = plt.subplots(figsize=(6.4, 4.9))
                plot_decision_surface(
                    ax,
                    torch_out["x"],
                    torch_out["y"],
                    torch_out["xx"],
                    torch_out["yy"],
                    torch_out["probs_grid"],
                    title="PyTorch: MLP on moons",
                )
                st.pyplot(fig)

        fig2, axes = plt.subplots(3, 1, figsize=(7.4, 7.4))
        axes[0].plot(tiny["losses"], label="tiny", linewidth=2.0)
        if torch_out is not None:
            axes[0].plot(torch_out["losses"], label="torch", linewidth=2.0)
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(tiny["accuracies"], label="tiny", linewidth=2.0)
        if torch_out is not None:
            axes[1].plot(torch_out["accuracies"], label="torch", linewidth=2.0)
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[2].plot(tiny["grad_norms"], label="tiny", linewidth=2.0)
        if torch_out is not None:
            axes[2].plot(torch_out["grad_norms"], label="torch", linewidth=2.0)
        axes[2].set_title("Gradient Norm")
        axes[2].legend()
        fig2.tight_layout()
        st.pyplot(fig2)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Tiny MLP accuracy", f"{100.0 * tiny['train_acc']:.1f}%")
        with m2:
            if torch_out is None:
                st.metric("PyTorch MLP accuracy", "N/A")
            else:
                st.metric("PyTorch MLP accuracy", f"{100.0 * torch_out['train_acc']:.1f}%")

    with subtabs[5]:
        st.markdown("Reference implementations (both frameworks) with explicit comments.")
        _progressive_code_block(
            key="s7_tiny_linear_code",
            title="Show Tiny Framework Linear Training Loop",
            code="""
from mlstack.autograd import Tensor
from mlstack.nn import Linear, Sequential, SGD, binary_cross_entropy_with_logits

# ------------------------------------------------------------
# Tiny-framework linear classifier
# ------------------------------------------------------------
# Input shape convention:
#   x_batch: (B, 2)   -> B samples, 2 features each
#   y_batch: (B, 1)   -> binary targets in {0, 1}
#
# Model:
#   logits = x @ W + b, where W shape is (2, 1), b shape is (1, 1)
# We intentionally keep sigmoid outside the model and use BCE-with-logits
# for better numerical stability.
# ------------------------------------------------------------
model = Sequential(Linear(in_features=2, out_features=1, seed=seed))
optimizer = SGD(model.parameters(), lr=lr, momentum=0.0)

for step in range(num_steps):
    # Forward pass through tiny autograd graph.
    logits = model(Tensor(x_batch, requires_grad=False))

    # Stable loss helper:
    #   Forward: mean(log(1 + exp(z)) - y*z)
    #   Backward wrt logits: (sigmoid(z) - y) / B
    loss_out = binary_cross_entropy_with_logits(logits, y_batch)
    loss = loss_out.loss

    # Backprop + update sequence:
    # 1) zero_grad: prevent stale accumulation from previous steps
    # 2) backward: compute dL/dW and dL/db
    # 3) step: apply parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Optional runtime diagnostics:
    #   - track loss scalar
    #   - track grad norm of model.parameters()
    #   - evaluate full-batch accuracy every K steps
""",
        )
        _progressive_code_block(
            key="s7_torch_linear_code",
            title="Show PyTorch Linear Training Loop",
            code="""
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# PyTorch linear classifier (same math as tiny-framework code)
# ------------------------------------------------------------
# Input shape convention:
#   x_batch: torch.Tensor of shape (B, 2), dtype float32/float64
#   y_batch: torch.Tensor of shape (B, 1), same dtype
#
# Model:
#   logits = x @ W + b via nn.Linear(2, 1)
# Loss:
#   BCEWithLogits combines sigmoid + BCE in a numerically stable form.
# ------------------------------------------------------------
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for step in range(num_steps):
    # Forward pass.
    logits = model(x_batch)

    # Stable logistic loss on logits.
    loss = F.binary_cross_entropy_with_logits(logits, y_batch)

    # Backprop + update sequence mirrors tiny framework exactly.
    optimizer.zero_grad()    # Clears old gradients on all model params.
    loss.backward()          # Populates param.grad using autograd graph.
    optimizer.step()         # Updates parameters in-place.

    # Optional runtime diagnostics:
    #   - assert torch.isfinite(loss)
    #   - compute gradient norm from model.parameters()
    #   - log accuracy periodically
""",
        )
        _progressive_code_block(
            key="s7_tiny_mlp_code",
            title="Show Tiny Framework MLP Training Loop",
            code="""
from mlstack.nn import Linear, ReLU, Sequential, Adam, binary_cross_entropy_with_logits

# ------------------------------------------------------------
# Tiny-framework MLP for nonlinear boundaries
# ------------------------------------------------------------
# Architecture:
#   input(2) -> Linear(2, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)
#
# Why this works on two-moons:
#   ReLU introduces nonlinearity, allowing piecewise-linear curved boundaries
#   instead of one global line.
# ------------------------------------------------------------
model = Sequential(
    Linear(2, hidden_dim, seed=seed),
    ReLU(),
    Linear(hidden_dim, 1, seed=seed + 1),
)
optimizer = Adam(model.parameters(), lr=lr)

for step in range(num_steps):
    # Forward pass on current mini-batch.
    logits = model(Tensor(x_batch, requires_grad=False))
    loss_out = binary_cross_entropy_with_logits(logits, y_batch)
    loss = loss_out.loss

    # Backward and optimizer step.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Optional diagnostics:
    #   - monitor gradient norm to catch exploding/vanishing patterns
    #   - monitor train accuracy curve for optimization health
    #   - snapshot boundary every N steps for lecture visuals
""",
        )
        _progressive_code_block(
            key="s7_torch_mlp_code",
            title="Show PyTorch MLP Training Loop",
            code="""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# PyTorch MLP (one-to-one conceptual mirror of tiny-framework MLP)
# ------------------------------------------------------------
# Same architecture and loss family as tiny version, so performance
# differences usually come from implementation details/hyperparameters,
# not from different math.
# ------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(2, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for step in range(num_steps):
    # Forward pass.
    logits = model(x_batch)
    loss = F.binary_cross_entropy_with_logits(logits, y_batch)

    # Backprop + optimizer update.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Optional diagnostics:
    #   - run model.eval() on validation split periodically
    #   - track loss/accuracy/grad_norm in logger
    #   - early-stop when validation metric plateaus
""",
        )

    with subtabs[6]:
        st.markdown("Debug checklist for both frameworks:")
        st.markdown("- Verify data and label shapes (`(B, D)` and `(B, 1)`).")
        st.markdown("- Check if loss decreases at all in first 50-100 updates.")
        st.markdown("- Track gradient norms; inspect for collapse to zero or explosion.")
        st.markdown("- Confirm no NaN/Inf in logits, loss, and gradients.")
        st.markdown("- Overfit a tiny subset to rule out implementation bugs.")
        st.markdown("Portfolio-quality deliverables from this chapter:")
        st.markdown("1. One notebook/report comparing tiny framework vs PyTorch on the same experiments.")
        st.markdown("2. Side-by-side boundary plots + loss/accuracy/grad-norm curves.")
        st.markdown("3. Brief analysis: what changed due to model capacity vs framework choice.")

def _render_stage(idx: int) -> None:
    if idx == 0:
        _render_stage_scalar_fundamentals()
    elif idx == 1:
        _render_stage_scalar_gradients()
    elif idx == 2:
        _render_stage_graph_topo()
    elif idx == 3:
        _render_stage_tiny_autograd()
    elif idx == 4:
        _render_stage_pytorch_mirror()
    elif idx == 5:
        _render_stage_multivariable()
    elif idx == 6:
        _render_stage_end_to_end()


# =========================
# App shell
# =========================
if "current_stage" not in st.session_state:
    st.session_state.current_stage = 0
if "lesson_started" not in st.session_state:
    st.session_state.lesson_started = False

st.title("Class 1: Fundamentals to Tiny Autograd")
st.caption("Systematic flow: scalar first -> autograd internals -> PyTorch mirror -> multivariable demos.")

with st.sidebar:
    st.header("Session Controls")
    mode = st.radio("Mode", ["Guided Lecture", "Free Explore"], index=0)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start Lesson", use_container_width=True):
            st.session_state.lesson_started = True
            st.session_state.current_stage = 0
            st.rerun()
    with col_b:
        if st.button("Reset", use_container_width=True):
            st.session_state.lesson_started = False
            st.session_state.current_stage = 0
            st.rerun()

    can_prev = st.session_state.current_stage > 0
    can_next = st.session_state.current_stage < (len(STAGES) - 1)

    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("Previous", disabled=not can_prev, use_container_width=True):
            st.session_state.current_stage -= 1
            st.rerun()
    with col_d:
        if st.button("Next", disabled=not can_next, use_container_width=True):
            st.session_state.lesson_started = True
            st.session_state.current_stage += 1
            st.rerun()

    stage_choice = st.selectbox(
        "Jump to stage",
        options=list(range(len(STAGES))),
        index=st.session_state.current_stage,
        format_func=lambda i: STAGES[i],
    )
    if stage_choice != st.session_state.current_stage:
        st.session_state.lesson_started = True
        st.session_state.current_stage = stage_choice
        st.rerun()

    progress = (st.session_state.current_stage + 1) / len(STAGES)
    st.progress(progress, text=f"Progress: {st.session_state.current_stage + 1}/{len(STAGES)}")

st.markdown(
    """
Source mapping:
- `mlstack/manual_neuron.py`
- `mlstack/autograd.py`
- `mlstack/gradcheck.py`
- `mlstack/vectorization.py`
- `mlstack/nn.py`, `mlstack/train.py`
"""
)
st.divider()

if mode == "Guided Lecture":
    if not st.session_state.lesson_started:
        st.subheader("Press Start Lesson")
        st.write("Stages are ordered to build intuition gradually before scaling up.")
        st.write("Within each stage, move left-to-right through sublesson tabs.")
    else:
        stage_idx = st.session_state.current_stage
        st.markdown(f"### {STAGES[stage_idx]}")
        _render_stage(stage_idx)

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("Previous Step", disabled=stage_idx == 0, use_container_width=True):
                st.session_state.current_stage -= 1
                st.rerun()
        with col2:
            if st.button("Next Step", disabled=stage_idx == len(STAGES) - 1, use_container_width=True):
                st.session_state.current_stage += 1
                st.rerun()
        with col3:
            st.caption("Each stage includes short prose, equations, worked numbers, code, and visuals.")
else:
    stage_idx = st.selectbox(
        "Select stage",
        options=list(range(len(STAGES))),
        index=st.session_state.current_stage,
        format_func=lambda i: STAGES[i],
    )
    st.session_state.current_stage = stage_idx
    st.markdown(f"### {STAGES[stage_idx]}")
    _render_stage(stage_idx)
