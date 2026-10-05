#!/usr/bin/env bash
set -euo pipefail

# FANCI RF Official45 150-tree inference environment installer.
# Usage:
#   chmod +x setup_env.sh
#   PYTHON_BIN=python3.11 ./setup_env.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
REQ_FILE="$PROJECT_DIR/requirements.txt"

choose_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            echo "[ERROR] PYTHON_BIN='$PYTHON_BIN' was not found."
            exit 1
        fi
        command -v "$PYTHON_BIN" 2>/dev/null || echo "$PYTHON_BIN"
        return
    fi
    for candidate in python3.11 python3.12 python3.13 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
            then
                command -v "$candidate"
                return
            fi
        fi
    done
    echo ""
}

PYTHON="$(choose_python)"
if [[ -z "$PYTHON" ]]; then
    echo "[ERROR] Python 3.11+ was not found."
    echo "Install Python 3.11 and python3.11-venv, then rerun this script."
    exit 1
fi

echo "============================================================"
echo " FANCI RF Official45 150-tree inference environment setup"
echo "============================================================"
echo "Project : $PROJECT_DIR"
echo "Python  : $PYTHON"
"$PYTHON" --version

if [[ -d "$VENV_DIR" && ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[ERROR] '$VENV_DIR' exists but is not a valid virtual environment."
    exit 1
fi
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[1/4] Creating $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "[1/4] Reusing $VENV_DIR"
fi

VPY="$VENV_DIR/bin/python"
echo "[2/4] Updating pip tooling"
"$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VPY" -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing pinned runtime dependencies"
"$VPY" -m pip install -r "$REQ_FILE"

echo "[4/4] Verifying versions and the 45-feature extractor"
(cd "$PROJECT_DIR" && "$VPY" - <<'PY'
import joblib
import numpy
import pika
import psutil
import sklearn
import yaml

from domain_inference.models.fanci_features import FEATURE_NAMES, OfficialFANCI45Extractor

print("Python package versions")
print("  NumPy       :", numpy.__version__)
print("  scikit-learn:", sklearn.__version__)
print("  joblib      :", joblib.__version__)
print("  pika        :", pika.__version__)
print("  PyYAML      :", yaml.__version__)
print("  psutil      :", psutil.__version__)

assert numpy.__version__ == "2.2.3"
assert sklearn.__version__ == "1.6.1"
assert joblib.__version__ == "1.4.2"
assert len(FEATURE_NAMES) == 45
extractor = OfficialFANCI45Extractor({".de"}, {".de"})
assert extractor.extract("itsec.rwth-aachen.de").shape == (45,)
print("[OK] Runtime and FANCI Official45 extractor verified.")
PY
)

"$VPY" -m pip check

echo
echo "SETUP COMPLETE"
echo "Activate with: source .venv/bin/activate"
echo "The 194 MiB model smoke test is intentionally separate:"
echo "  python -m pytest tests/test_model_smoke.py -q"

