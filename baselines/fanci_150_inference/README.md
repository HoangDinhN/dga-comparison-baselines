# FANCI RF Official45 — 150 trees

Project testbed này dùng biến thể 150 cây được lấy trực tiếp từ 150 estimator đầu của artifact FANCI Official45 785 cây. Không có cây nào được train lại. Bộ trích xuất 45 đặc trưng, Public Suffix List, dữ liệu train và toàn bộ tham số của từng cây được giữ nguyên.

```text
test_genAI.csv -> Router passthrough -> jobs.worker
                                      -> FANCI 150-tree workers
                                      -> results -> Controller
```

## Artifact

- File: `artifact/fanci_rf_official45_150trees_bundle.joblib`
- Kích thước: 203.653.273 byte, khoảng 194,22 MiB
- SHA256: `a0251fd62d21c6fe39adc1b3ae9e51715c7398c48931561a6560c12ccfc5e6c9`
- Nguồn 785 cây SHA256: `952032ee5f1c0fec3190720adb1f34df0b522036de2a99f3d423b3a3d380bae5`
- Số cây: 150
- Số đặc trưng: 45
- Threshold triển khai: `0.500924704924705`, được chọn chỉ trên validation
- `model_type`: `fanci_random_forest_official45_150trees`

Metadata trong bundle ghi rõ `retrained: false` và phương pháp `first_n_estimators`. Báo cáo xuất artifact nằm tại `reports/fanci_150_build_report.json`.

## Kết quả kiểm tra

Trên `test_genAI.csv` sau cùng chính sách làm sạch của notebook:

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 90,1396% | 89,0707% | 70,7080% | 78,8342% |

Bản 785 cây đạt Accuracy 90,1745% và F1 78,8261%. Bản 150 cây giảm Accuracy 0,0349 điểm phần trăm, còn F1 chênh +0,0081 điểm phần trăm.

## Phân bố artifact

Chỉ Controller machine-9 giữ artifact lâu dài. Worker tải file khoảng 194 MiB qua HTTP, xác minh SHA256, nạp vào RAM rồi xóa bản tạm. Router không tải artifact.

Bản 150 cây phù hợp hơn với Worker 4 GB RAM. Phép đo nạp artifact trên máy kiểm thử cho RSS ổn định 1.115 MiB, RSS đỉnh 2.076 MiB và thời gian load 3,49 giây. Linux có thể khác, vì vậy vẫn nên bảo đảm mỗi Worker còn ít nhất khoảng 2,5 GiB RAM `available` trước khi khởi động và kiểm tra lại bằng telemetry.

## Cài môi trường trên mỗi máy

```bash
cd ~/hoang/fanci_150_inference
python3.11 --version
chmod +x setup_env.sh
PYTHON_BIN=python3.11 ./setup_env.sh
source .venv/bin/activate
which python
python --version
python -c 'import sklearn, numpy, joblib, pika, yaml, psutil; print("sklearn:", sklearn.__version__, "| numpy:", numpy.__version__, "| joblib:", joblib.__version__)'
python -m pip check
python -m pytest tests/test_features.py -q
```

Kết quả đúng cần có:

- scikit-learn 1.6.1
- NumPy 2.2.3
- joblib 1.4.2
- `pip check` không báo lỗi
- test feature pass

Trên machine-9 chạy thêm smoke test đầy đủ:

```bash
python -m pytest tests/test_model_smoke.py -q
```

Smoke test kiểm tra SHA256, 45 đặc trưng, đúng 150 cây, threshold và dự đoán mẫu.

## Cấu hình

- `config/controller.example.yaml`: kiểm tra IP machine-9 và đường dẫn artifact.
- `config/router-client.example.yaml`: sửa đường dẫn `test_genAI.csv`.
- `config/worker-client.example.yaml`: đặt `client.id` riêng từ `worker-01` đến `worker-07`.
- Controller, Router và mọi Worker phải dùng cùng `DGA_QUEUE_PREFIX` và `DGA_ARTIFACT_TOKEN`.

Mặc định project dùng queue prefix `dga.fanci_150`. Mỗi lần chạy nên tạo prefix mới để tránh message cũ.

## Chạy testbed

### Controller trên machine-9

```bash
cd ~/hoang/fanci_150_inference
source .venv/bin/activate
DGA_QUEUE_PREFIX="dga.fanci_150_test01" DGA_ARTIFACT_TOKEN="fanci-150-test01" \
python run.py controller --config config/controller.example.yaml
```

### Mỗi Worker

Sửa `client.id` trong `config/worker-client.example.yaml`, sau đó:

```bash
cd ~/hoang/fanci_150_inference
source .venv/bin/activate
DGA_QUEUE_PREFIX="dga.fanci_150_test01" DGA_ARTIFACT_TOKEN="fanci-150-test01" \
python run.py client --config config/worker-client.example.yaml
```

### Router chạy cuối cùng

```bash
cd ~/hoang/fanci_150_inference
source .venv/bin/activate
DGA_QUEUE_PREFIX="dga.fanci_150_test01" DGA_ARTIFACT_TOKEN="fanci-150-test01" \
python run.py client --config config/router-client.example.yaml
```

Cấu hình mặc định lặp tập test đến 1.000.000 domain. Muốn chạy đúng một lượt file, đặt:

```yaml
loop_forever: false
max_records: null
```

## Kết quả và telemetry

Controller ghi:

- `var/test_genAI/results_1M.jsonl`
- `var/test_genAI/telemetrys_1M.jsonl`

Phân tích:

```bash
python analyze_results.py var/test_genAI/results_1M.jsonl
python analyze_telemetry.py var/test_genAI/telemetrys_1M.jsonl
```

Phần trích xuất Official45 thường chiếm nhiều thời gian hơn `predict_proba`. Giảm từ 785 xuống 150 cây chủ yếu giải quyết RAM; tổng throughput còn phụ thuộc mạnh vào `feature_n_jobs`, CPU của Worker và backlog RabbitMQ.
