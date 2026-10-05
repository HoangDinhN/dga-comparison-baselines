# Bilbo single-branch distributed inference

Project này triển khai checkpoint `bilbo.pt_ver2` trong cùng kiểu testbed
Controller/Router/Worker của `inference_domain` để có thể so sánh trên cùng dữ liệu,
số worker, batch size và hạ tầng RabbitMQ.

## Luồng dữ liệu

```text
CSV -> Router (passthrough) -> jobs.worker -> Bilbo Worker
    -> results queue -> Controller -> results.jsonl
                         \-> telemetrys.jsonl
```

Bilbo chỉ có một nhánh model. Router không tải model và mọi domain đều đi tới
`jobs.worker`. Controller phân phối checkpoint cho các worker qua artifact server.

## Model và tiền xử lý

Worker tái tạo đúng kiến trúc trong `bilbo_v2.ipynb`:

- lấy SLD bằng `tldextract`, chuyển lowercase, bỏ ký tự ngoài `[a-z0-9-]`;
- bảng ký tự `a-z`, `0-9`, `.`, `-`; padding index bằng 0;
- chiều dài đầu vào 32;
- nhánh CNN: embedding 128, 128 filter cho mỗi kernel 2, 3, 4, 5, 6;
- nhánh LSTM: embedding 128, hidden size 256;
- các nhánh đi qua dense 100 và sigmoid;
- dự đoán DGA khi xác suất lớn hơn 0.5, giống notebook.

Checkpoint được đặt tại `artifact/bilbo.pt_ver2`. SHA-256 của bản đã chép:

```text
1E16A5EFE6A7EF4349F7F3DB51CA9D1788DC642350F968727A76C2E688171B75
```

## Chuẩn bị

Trên Controller, Router và từng Worker:

```bash
cd ~/hoang/bilbo_inference
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
```

Sửa trước khi chạy:

- `config/controller.example.yaml`: địa chỉ RabbitMQ, `public_base_url`, token,
  đường dẫn checkpoint và số worker cần chờ;
- `config/router-client.example.yaml`: đường dẫn CSV và số record;
- `config/worker-client.example.yaml`: mỗi máy dùng một `client.id` riêng.

Tất cả máy phải dùng cùng `DGA_QUEUE_PREFIX`.

## Chạy test 3 máy

Với 1 Worker, trước tiên đổi `bootstrap.required_model_workers` trong
`config/controller.example.yaml` từ `7` thành `1`.

Controller:

```bash
DGA_QUEUE_PREFIX="dga.Bilbo_test02" python run.py controller --config config/controller.example.yaml
```

Worker:

```bash
DGA_QUEUE_PREFIX="dga.Bilbo_test02" python run.py client --config config/worker-client.example.yaml
```

Router chạy cuối cùng:

```bash
DGA_QUEUE_PREFIX="dga.Bilbo_test02" python run.py client --config config/router-client.example.yaml
```

Với testbed đầy đủ 7 Worker, giữ `bootstrap.required_model_workers: 7` và đổi ID
từng worker thành `worker-01` tới `worker-07`.

## Kiểm tra model và phân tích kết quả

```bash
python -m pytest tests/test_model_smoke.py -q
python analyze_results.py var/Bilbo/results.jsonl
python analyze_telemetry.py var/Bilbo/telemetrys.jsonl
```

Hai file đầu ra giữ cùng schema với project CNN-LSTM một nhánh, nên có thể dùng
trực tiếp để so sánh accuracy/F1, throughput, inference latency, E2E latency,
CPU và RAM.
