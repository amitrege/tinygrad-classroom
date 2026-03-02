"""Environment verification for reliable lecture execution."""

from __future__ import annotations

import importlib.util
import platform


def _version(module_name: str) -> str:
    if importlib.util.find_spec(module_name) is None:
        return "missing"
    module = __import__(module_name)
    return getattr(module, "__version__", "unknown")


def main() -> None:
    print("[verify] platform:", platform.platform())
    print("[verify] python:", platform.python_version())
    print("[verify] numpy:", _version("numpy"))
    print("[verify] matplotlib:", _version("matplotlib"))
    print("[verify] streamlit:", _version("streamlit"))

    if importlib.util.find_spec("torch") is None:
        print("[verify] torch: missing (PyTorch mirror stage will be disabled)")
        return

    import torch

    device_hint = "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device_hint = "mps"
    elif torch.cuda.is_available():
        device_hint = "cuda"

    print("[verify] torch:", torch.__version__)
    print("[verify] torch default device hint:", device_hint)


if __name__ == "__main__":
    main()
