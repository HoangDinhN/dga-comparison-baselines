# DeepDGA single-branch distributed inference

Project triển khai ba artifact tạo bởi `deepDGA_ver3_fixed.ipynb` trong testbed
Controller/Router/Worker. Đầu ra dùng cùng schema với các baseline một nhánh để
so sánh accuracy/F1, throughput, latency, CPU và RAM.

## Luồng dữ liệu

```text
CSV -> Router (passthrough) -> jobs.worker -> DeepDGA Worker
    -> results queue -> Controller -> results.jsonl
                         \-> telemetrys.jsonl
```

DeepDGA chỉ có một nhánh. Router không tải model; Controller cấp checkpoint,
KeyedVectors và TF-IDF cho Worker qua artifact server.

## Artifact và tiền xử lý

Ba tệp trong `artifact/` phải luôn đi cùng nhau:

- `deepDGA_best_bundle.pt`: cấu hình, threshold và trọng số PyTorch;
- `deepDGA_keyedvectors.kv`: 93.118 vector từ 100 chiều, `float32`
  (`gensim 4.4.0`), được trích nguyên vẹn từ Word2Vec;
- `deepDGA_tfidf.joblib`: vocabulary/IDF (`scikit-learn 1.6.1`).

Worker lấy hostname bằng `urlsplit`, lowercase, chuyển IDNA và từ chối IP hoặc
ký tự ngoài `a-z`, `0-9`, `.`, `-`. Nhánh ký tự dùng chuỗi tối đa 75 ký tự và
BiLSTM có packed sequence. Nhánh từ dùng `wordninja`, KeyedVectors và năm phép gộp
`min/mean/max/sum/TF-IDF weighted average` thành vector 500 chiều. Xác suất
sigmoid được so với threshold lưu trong checkpoint.

Kiến trúc giữ nguyên notebook: embedding 128, BiLSTM hidden 128 mỗi hướng,
word FC 500→128, fusion FC 384→64 và đầu ra 64→1; tổng cộng 358.017 tham số.

SHA-256:

```text
deepDGA_best_bundle.pt     73D68A5F48739B9BC3288B772E6A7F134D6029E201258D1653203F76D1EC6CDA
deepDGA_tfidf.joblib       72699F162755A89D9602F1AC0CE761BA3B501C0598EA41500D838C6B8A6DE782
deepDGA_keyedvectors.kv    898A389126DF1D3060EF7EB89C3307BBE27263C9A81B670C91B16243BE8D1D48
```

Mảng vector trong KeyedVectors có SHA-256
`E2A0064EC3A4D8BB16173ED0FA02BA76D7ABF7055EE1BA42959BC48E9BB60A71`
và đã được kiểm tra `array_equal` với vector của Word2Vec gốc. Việc chuyển đổi
không lượng tử hóa, không cắt từ vựng và không thay đổi đầu vào của DeepDGA.

## Cài môi trường

Cài giống nhau trên Controller, Router và từng Worker:

```bash
cd ~/hoang/deepdga_inference
chmod +x setup_env.sh
PYTHON_BIN=python3.11 ./setup_env.sh
source .venv/bin/activate
python -m pytest tests/test_model_smoke.py -q
```

Script mặc định cài PyTorch CPU 2.4.1. Nếu máy đã có đúng PyTorch/CUDA cần dùng,
chạy `SKIP_TORCH=1 PYTHON_BIN=python3.11 ./setup_env.sh`.

## Cấu hình

- Controller: sửa RabbitMQ, `public_base_url`, token, ba đường dẫn artifact và
  `required_model_workers` trong `config/controller.example.yaml`.
- Router: sửa `csv_path`, `max_records` và `loop_forever` trong
  `config/router-client.example.yaml`.
- Mỗi Worker: đặt `client.id` riêng (`worker-01` ... `worker-07`) trong
  `config/worker-client.example.yaml`.
- Tất cả máy phải dùng cùng một giá trị `DGA_QUEUE_PREFIX`.

Với một Worker, đổi `bootstrap.required_model_workers` từ `7` thành `1`.

## Chạy testbed

Chạy Controller trước:

```bash
DGA_QUEUE_PREFIX="dga.deepdga.test03" python run.py controller --config config/controller.example.yaml
```

Chạy lệnh này trên mỗi Worker sau khi đã đặt `client.id` riêng:

```bash
DGA_QUEUE_PREFIX="dga.deepdga.test03" python run.py client --config config/worker-client.example.yaml
```

Chạy Router cuối cùng:

```bash
DGA_QUEUE_PREFIX="dga.deepdga.test03" python run.py client --config config/router-client.example.yaml
```

## Phân tích kết quả

```bash
python analyze_results.py var/DeepDGA/results.jsonl
python analyze_telemetry.py var/DeepDGA/telemetrys.jsonl
```
