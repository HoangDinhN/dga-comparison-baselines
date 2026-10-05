# LSTM_MI single-branch distributed inference

Project triển khai checkpoint `best_model_LSTM_MI.pt` trong cùng kiểu testbed
Controller/Router/Worker của `inference_domain`, giúp so sánh LSTM_MI với các
baseline khác trên cùng dữ liệu, số worker, batch size và hạ tầng RabbitMQ.

## Luồng dữ liệu

```text
CSV -> Router (passthrough) -> jobs.worker -> LSTM_MI Worker
    -> results queue -> Controller -> results.jsonl
                         \-> telemetrys.jsonl
```

LSTM_MI chỉ có một nhánh. Router không tải model; Controller phân phối checkpoint
cho các Worker qua artifact server.

## Model và tiền xử lý

Worker đọc cấu hình trực tiếp từ checkpoint và kiểm tra format/mapping nhãn:

- làm sạch URL bằng `urlsplit`, lấy toàn bộ hostname, lowercase và chuyển IDNA;
- từ chối IP, URL chứa username/password và scheme ngoài HTTP/HTTPS;
- bảng ký tự `a-z`, `0-9`, `.`, `-`, `_`, có PAD và UNK riêng;
- chiều dài tối đa 75;
- embedding 64, LSTM hidden 128, dropout 0.3;
- đầu ra softmax hai lớp `[Benign, DGA]`;
- threshold lấy trực tiếp từ checkpoint.

Checkpoint: `artifact/best_model_LSTM_MI.pt`

SHA-256:

```text
6BEBECD7C6F8C6CDDDBC2BD7927F1365B51C3BB2AA836BF8BD51BB99737FFF01
```

## Cài môi trường

Trên Controller, Router và từng Worker:

```bash
cd ~/hoang/lstm_mi_inference
chmod +x setup_env.sh
PYTHON_BIN=python3.11 ./setup_env.sh
source .venv/bin/activate
python -m pytest tests/test_model_smoke.py -q
```

## Cấu hình

- Controller: sửa RabbitMQ, `public_base_url`, token, đường dẫn checkpoint và
  `required_model_workers` trong `config/controller.example.yaml`.
- Router: sửa đường dẫn CSV và `max_records` trong `config/router-client.example.yaml`.
- Mỗi Worker: đặt `client.id` riêng (`worker-01` ... `worker-07`) trong
  `config/worker-client.example.yaml`.
- Tất cả máy phải dùng cùng `DGA_QUEUE_PREFIX`.

Với 1 Worker, đổi `bootstrap.required_model_workers` từ `7` thành `1`.

## Chạy

Controller:

```bash
DGA_QUEUE_PREFIX="dga.LSTM_MI_test03" python run.py controller --config config/controller.example.yaml
```

Worker:

```bash
DGA_QUEUE_PREFIX="dga.LSTM_MI_test03" python run.py client --config config/worker-client.example.yaml
```

Router chạy cuối cùng:

```bash
DGA_QUEUE_PREFIX="dga.LSTM_MI_test03" python run.py client --config config/router-client.example.yaml
```

## Phân tích kết quả

```bash
python analyze_results.py var/LSTM_MI/results.jsonl
python analyze_telemetry.py var/LSTM_MI/telemetrys.jsonl
```

Đầu ra giữ cùng schema với các project một nhánh khác để so sánh accuracy/F1,
throughput, inference latency, E2E latency, CPU và RAM.
