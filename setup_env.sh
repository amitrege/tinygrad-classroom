#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -n "${PYTHON_BIN:-}" ]; then
  SELECTED_PYTHON="$PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  # Prefer 3.12 because PyTorch wheels are broadly available for it.
  SELECTED_PYTHON="python3.12"
else
  SELECTED_PYTHON="python3"
fi

VENV_DIR="${VENV_DIR:-.venv}"
ALLOW_NO_TORCH="${ALLOW_NO_TORCH:-0}"

echo "[setup] root: $ROOT_DIR"
echo "[setup] python: $SELECTED_PYTHON"

target_py_ver="$($SELECTED_PYTHON - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
echo "[setup] target python version: $target_py_ver"

need_create_venv=0
if [ ! -d "$VENV_DIR" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
  need_create_venv=1
else
  current_py_ver="$($VENV_DIR/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  if [ "$current_py_ver" != "$target_py_ver" ]; then
    echo "[setup] existing venv python=$current_py_ver differs from target=$target_py_ver"
    echo "[setup] recreating virtualenv with $SELECTED_PYTHON"
    "$SELECTED_PYTHON" -m venv --clear "$VENV_DIR"
  fi
fi

if [ "$need_create_venv" -eq 1 ]; then
  echo "[setup] creating virtualenv at $VENV_DIR"
  "$SELECTED_PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[setup] upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing base dependencies"
python -m pip install -r requirements.txt

if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
  echo "[setup] torch already installed"
else
  echo "[setup] installing PyTorch (CPU/local)"
  set +e
  python -m pip install torch
  rc_default=$?
  if [ "$rc_default" -ne 0 ]; then
    echo "[setup] default torch install failed, trying CPU wheel index"
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
    rc_cpu=$?
  else
    rc_cpu=0
  fi
  set -e

  if [ "$rc_default" -ne 0 ] && [ "$rc_cpu" -ne 0 ]; then
    echo "[setup] torch installation failed"
    if [ "$target_py_ver" = "3.13" ] && command -v python3.12 >/dev/null 2>&1; then
      echo "[setup] hint: rerun with Python 3.12 -> PYTHON_BIN=python3.12 ./setup_env.sh"
    fi

    if [ "$ALLOW_NO_TORCH" = "1" ]; then
      echo "[setup] continuing without torch because ALLOW_NO_TORCH=1"
    else
      exit 1
    fi
  fi
fi

echo "[setup] verifying environment"
python verify_env.py

echo "[setup] completed"
