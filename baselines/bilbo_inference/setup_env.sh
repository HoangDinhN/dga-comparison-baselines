#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Không tìm thấy $PYTHON_BIN"
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

"$VPY" -m pip install --upgrade pip setuptools wheel

# CPU wheel giúp các VM không có GPU cài đặt nhỏ và ổn định hơn.
# Muốn chạy CUDA, cài torch phù hợp từ pytorch.org trước rồi đặt SKIP_TORCH=1.
if [[ "${SKIP_TORCH:-0}" != "1" ]]; then
    "$VPY" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1"
fi

"$VPY" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$VPY" -m pip check

cd "$PROJECT_DIR"
"$VPY" - <<'PY'
from domain_inference.models.bilbo import BilboDGA

model = BilboDGA("artifact/bilbo.pt_ver2")
results = model.predict_many(["google.com", "ajd82ksla9q.net"])
assert len(results) == 2
print("[OK] Bilbo checkpoint loaded and inference completed")
print(results)
PY

echo "[OK] Môi trường Bilbo đã sẵn sàng"
echo "Kích hoạt bằng: source $VENV_DIR/bin/activate"
