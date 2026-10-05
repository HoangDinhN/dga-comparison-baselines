# FastText CNN-LSTM FD32 testbed

Project inference một nhánh cho bundle được xuất bởi
`FastText_CNN_LSTM_FullDomain32_300to128_LastValid.ipynb`. Project dùng cùng giao
thức Controller → Router → Worker và cùng cách thu telemetry với các baseline
testbed trước.

```text
test_genAI.csv -> Router passthrough -> RabbitMQ jobs.worker
               -> 7 Workers -> RabbitMQ results -> Controller
```

Project này độc lập với `cnn_lstm_inference` và bản FastText 75x300 cũ.

## Hành vi model được giữ đúng theo notebook

- Input là toàn bộ domain sau `strip()`, lowercase và bỏ dấu chấm cuối.
- Nếu dài hơn 32 ký tự, giữ **32 ký tự cuối** để ưu tiên SLD/TLD.
- Alphabet là `a-z`, `0-9`, `.`, `-`, `_`; ký tự ngoài alphabet dùng UNK=1.
- Chuỗi ngắn được padding 0 ở bên phải đến 32 ký tự.
- FastText 300 chiều và phép chiếu 300→128 đã được gộp thành embedding 128 chiều
  trong bundle. Inference không thực hiện phép chiếu và không cần file FastText.
- CNN dùng 9 kernel từ 2 đến 10, mỗi kernel 256 filters, global max pooling và
  BatchNorm.
- LSTM hidden 128 dùng `pack_padded_sequence`, nên lấy hidden state tại ký tự
  hợp lệ cuối cùng thay vì vị trí padding cuối.
- Fusion nối 2304 đặc trưng CNN với 128 đặc trưng LSTM; output là một logit.
- Threshold lấy từ `best_threshold` đã chọn trên validation.

Bundle đi kèm tại `artifact/best_bundle.pt`. Chỉ machine-9 cần giữ file này.
Controller cung cấp artifact qua HTTP; mỗi Worker tải, kiểm tra SHA-256, nạp model
vào RAM rồi xóa thư mục tải tạm. Router không cần artifact.

## Cấu hình mặc định

- Controller: machine-9, `192.168.101.99`
- Router: machine-3, đọc `/home/machine-3/hoang/data/test_genAI.csv`
- RabbitMQ: `192.168.101.91:5672`
- Artifact server: `192.168.101.99:8080`
- Queue prefix: `dga.fasttext_cnn_lstm_fd32`
- 7 Workers, batch size 32, prefetch 64
- Controller artifact:
  `/home/machine-9/hoang/fasttext_cnn_lstm_fd32_inference/artifact/best_bundle.pt`

Ba file config tương ứng đúng ba vai trò:

- `config/controller.example.yaml`: machine-9
- `config/router-client.example.yaml`: machine-3
- `config/worker-client.example.yaml`: từng Worker

Trên mỗi Worker, sửa dòng `client.id` trong `worker-client.example.yaml` thành ID
riêng từ `worker-01` đến `worker-07`. Không để hai máy dùng cùng ID.

Prefix đã có sẵn trong cả ba config nên lệnh chạy không bắt buộc ghi
`DGA_QUEUE_PREFIX`. Khi muốn tạo một lượt chạy cô lập, có thể đặt biến này thành
cùng một giá trị trên Controller, Router và tất cả Workers, ví dụ
`dga.fasttext_cnn_lstm_fd32.test02`.

## Cài môi trường trên từng máy

Chép project vào `~/hoang/fasttext_cnn_lstm_fd32_inference` trên machine-9,
machine-3 và tất cả Worker. Chạy trên **mỗi máy**:

```bash
cd ~/hoang/fasttext_cnn_lstm_fd32_inference
python3.11 --version
chmod +x setup_env.sh
PYTHON_BIN=python3.11 ./setup_env.sh
source .venv/bin/activate
which python
python --version
python -c 'import torch, numpy, pika, yaml, psutil; print("torch:", torch.__version__, "| numpy:", numpy.__version__, "| CUDA:", torch.cuda.is_available(), "| threads:", torch.get_num_threads())'
python -m pip check
python -m pytest tests/test_model_smoke.py -q
```

Kết quả đúng:

- `which python` trỏ đến
  `.../fasttext_cnn_lstm_fd32_inference/.venv/bin/python`.
- `python --version` là Python 3.11.x.
- Máy CPU thường in `torch: 2.4.1+cpu`, `CUDA: False`.
- `pip check` in `No broken requirements found.`.
- Machine-9 có bundle: `3 passed`.
- Router/Worker không có bundle: `2 passed, 1 skipped`.

Script cài PyTorch 2.4.1 CPU. Cách cài thủ công tương đương:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy==1.26.4'
python -m pip install 'torch==2.4.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -m pytest tests/test_model_smoke.py -q
```

## Chạy testbed

Khởi động Controller trên machine-9:

```bash
cd ~/hoang/fasttext_cnn_lstm_fd32_inference
source .venv/bin/activate
DGA_QUEUE_PREFIX="dga.fasttext_cnn_lstm_fd32.test01" python run.py controller --config config/controller.example.yaml
```

Khởi động trên từng Worker sau khi đã đặt `client.id` khác nhau:

```bash
cd ~/hoang/fasttext_cnn_lstm_fd32_inference
source .venv/bin/activate
DGA_QUEUE_PREFIX="dga.fasttext_cnn_lstm_fd32.test01" python run.py client --config config/worker-client.example.yaml
```

Chờ Controller nhận đủ `worker-01` đến `worker-07`, sau đó mới chạy Router trên
machine-3:

```bash
cd ~/hoang/fasttext_cnn_lstm_fd32_inference
source .venv/bin/activate
DGA_QUEUE_PREFIX="dga.fasttext_cnn_lstm_fd32.test01" python run.py client --config config/router-client.example.yaml
```

Giá trị `DGA_QUEUE_PREFIX` phải giống hệt trên Controller, cả 7 Workers và Router.
Đổi `test01` thành `test02`, `test03`, ... cho mỗi lượt chạy mới để không nhận
message còn sót từ queue của lần chạy trước.

Controller ghi:

- `var/test_genAI_fd32/results_1M.jsonl`
- `var/test_genAI_fd32/telemetrys_1M.jsonl`

Phân tích sau khi toàn bộ một triệu domain đã hoàn thành:

```bash
python analyze_results.py var/test_genAI_fd32/results_1M.jsonl
python analyze_telemetry.py var/test_genAI_fd32/telemetrys_1M.jsonl
```

## Đo tốc độ model trên từng máy

`benchmark_model.py` đo riêng `predict_many()` và loại RabbitMQ, mạng, queue chờ
ra khỏi phép đo. Trên machine-9 hoặc một Worker có bản sao bundle:

```bash
python benchmark_model.py --bundle artifact/best_bundle.pt --batch-sizes 1,8,32,64 --warmup 10 --repeats 50
python benchmark_model.py --bundle artifact/best_bundle.pt --batch-sizes 32 --threads 1
python benchmark_model.py --bundle artifact/best_bundle.pt --batch-sizes 32 --threads 4
```

Testbed hiện dùng batch 32, vì vậy hàng `batch=32` là số cần so sánh với
`inference_ms` và `rate` trong telemetry. Giữ số thread giống nhau khi so sánh
các baseline.

## Ước tính tốc độ

Bundle triển khai có 1,916,161 tham số và khoảng 47.8 triệu MAC/domain với độ dài
trung bình của dữ liệu train (15.35 ký tự); trường hợp đủ 32 ký tự khoảng 50.0
triệu MAC/domain. So với bản FastText 75x300 khoảng 302.1 triệu MAC/domain, phần
forward giảm khoảng **6.3 lần**. Nó gần bằng CNN-LSTM 32x128 cũ về kích thước
tensor, nhưng thêm BatchNorm và packed LSTM.

Trên Colab T4, notebook đã đo khoảng 13,307 domain/s ở batch 64. Đây là GPU và
không dùng để dự báo trực tiếp máy testbed CPU. Dựa trên lần chạy CNN-LSTM 32x128
cũ (khoảng 2,031 domain/s toàn hệ thống với 7 Workers), mục tiêu hợp lý cho bản
này là khoảng **1,500–2,100 domain/s**, tương đương khoảng **8–11 phút cho một
triệu domain**, nếu CPU, thread, batch và RabbitMQ tương đương. Packed sequence
có thêm chi phí sắp xếp/nén nên tốc độ thực có thể thấp hơn một phần.

Theo dõi `e2e_p95` khi chạy. Nếu `e2e_p95` tăng liên tục lên hàng chục hoặc hàng
trăm giây trong khi `infer_p95` vẫn thấp, Router đang đẩy nhanh hơn tổng năng lực
Workers và queue đang tích backlog. Khi đó giảm `max_total_queue_depth`, giới hạn
`max_publish_rate_per_second`, hoặc tăng năng lực Worker; đó không phải lỗi tải
model. Kết quả chuẩn cuối cùng phải lấy từ `analyze_telemetry.py` vì hiệu năng CPU
thực phụ thuộc từng máy và cấu hình thread.

