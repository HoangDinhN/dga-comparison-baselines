#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install 'numpy==1.26.4'
"$VENV_DIR/bin/python" -m pip install 'torch==2.4.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m pytest "$PROJECT_DIR/tests/test_model_smoke.py" -q

echo "Environment ready: $VENV_DIR"
