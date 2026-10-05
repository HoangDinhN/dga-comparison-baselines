#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Không tìm thấy $PYTHON_BIN. Project yêu cầu Python 3.10 trở lên; khuyên dùng 3.11."
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "[ERROR] Project yêu cầu Python 3.10 trở lên; khuyên dùng 3.11."
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VPY="$VENV_DIR/bin/python"
if [[ ! -x "$VPY" ]]; then
    echo "[ERROR] $VENV_DIR không phải virtual environment hợp lệ"
    exit 1
fi

if ! "$VPY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "[ERROR] .venv dùng Python quá cũ. Hãy xóa .venv rồi chạy lại bằng Python 3.11."
    exit 1
fi

"$VPY" -m pip install --upgrade pip setuptools wheel

if [[ "${SKIP_TORCH:-0}" != "1" ]]; then
    "$VPY" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1"
fi

"$VPY" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$VPY" -m pip check

cd "$PROJECT_DIR"
"$VPY" - <<'PY'
from domain_inference.models.lstm_mi import LSTMMIDGA

model = LSTMMIDGA("artifact/best_model_LSTM_MI.pt")
results = model.predict_many(["google.com", "ajd82ksla9q.net"])
assert len(results) == 2
assert all(item["prediction"] in (0, 1) for item in results)
print("[OK] LSTM_MI checkpoint loaded and inference completed")
print(results)
PY

echo "[OK] Môi trường LSTM_MI đã sẵn sàng"
echo "Kích hoạt bằng: source $VENV_DIR/bin/activate"
