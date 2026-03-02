#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8501}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"
GENERATE_HOOK_PLOTS="${GENERATE_HOOK_PLOTS:-0}"

if [ ! -d ".venv" ]; then
  echo "[run] missing .venv. run ./setup_env.sh first"
  exit 1
fi

# shellcheck disable=SC1090
source ".venv/bin/activate"

# Keep all runtime caches local to avoid permission issues in locked environments.
CACHE_DIR="$ROOT_DIR/.cache"
STREAMLIT_HOME="$ROOT_DIR/.streamlit_home"
mkdir -p "$CACHE_DIR/matplotlib" "$STREAMLIT_HOME/.streamlit"

export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export XDG_CACHE_HOME="$CACHE_DIR"
export STREAMLIT_SERVER_HEADLESS=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Avoid first-run email prompt by writing local streamlit credentials.
cat > "$STREAMLIT_HOME/.streamlit/credentials.toml" <<'TOML'
[general]
email = ""
TOML

cat > "$STREAMLIT_HOME/.streamlit/config.toml" <<TOML
[browser]
gatherUsageStats = false

[server]
headless = false
address = "127.0.0.1"
port = $PORT
TOML

if [ "$OPEN_BROWSER" = "1" ]; then
  python - <<PY &
import os
import time
import webbrowser
port = os.environ.get("PORT", "$PORT")
time.sleep(1.8)
webbrowser.open(f"http://127.0.0.1:{port}")
PY
fi

if [ "$GENERATE_HOOK_PLOTS" = "1" ]; then
  set +e
  python make_hook_plots.py
  set -e
fi

exec env HOME="$STREAMLIT_HOME" streamlit run app.py --server.port "$PORT" --server.address 127.0.0.1
