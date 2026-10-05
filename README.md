# DGA Comparison Baselines

Repository này lưu trữ đầy đủ năm baseline được sử dụng để đánh giá bài toán phát hiện tên miền sinh bởi Domain Generation Algorithm (DGA). Mỗi project bao gồm bài báo tham chiếu, notebook huấn luyện, artifact triển khai, mã inference, cấu hình testbed, kiểm thử, kết quả thô và báo cáo.


## Baselines

| Project | Baseline | Phiên bản triển khai |
|---|---|---|
| [`bilbo_inference`](baselines/bilbo_inference/) | BILBO | Model BILBO cho phát hiện DGA |
| [`deepdga_inference`](baselines/deepdga_inference/) | DeepDGA | Character BiLSTM kết hợp Word2Vec và TF-IDF |
| [`fanci_150_inference`](baselines/fanci_150_inference/) | FANCI | Official45 Random Forest, 785 cây-> 150 cây |
| [`fasttext_cnn_lstm_fd32_inference`](baselines/fasttext_cnn_lstm_fd32_inference/) | FastText CNN-LSTM | Full-domain length 32, embedding 300→128, Last Valid LSTM |
| [`lstm_mi_inference`](baselines/lstm_mi_inference/) | LSTM-MI | LSTM-MI cho phát hiện DGA |

Hướng dẫn cài đặt và vận hành chi tiết nằm trong `README.md` của từng project.

## Repository structure

```text
dga-comparison-baselines/
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
└── baselines/
    ├── bilbo_inference/
    ├── deepdga_inference/
    ├── fanci_150_inference/
    ├── fasttext_cnn_lstm_fd32_inference/
    └── lstm_mi_inference/
```

Mỗi baseline sử dụng cấu trúc chung:

```text
baseline_inference/
├── artifact/
├── config/
├── domain_inference/
├── paper/
├── reports/
├── tests/
├── training/
├── var/
├── README.md
├── requirements.txt
├── run.py
└── setup_env.sh
```

## Ý nghĩa các thành phần

| Thành phần | Ý nghĩa |
|---|---|
| `artifact/` | Trọng số model và các bộ tiền xử lý cần thiết cho inference. |
| `config/` | Cấu hình Controller, Router, Worker, RabbitMQ và dữ liệu đầu vào. |
| `domain_inference/` | Mã nguồn hệ thống inference phân tán và model adapter. |
| `paper/` | Bài báo được sử dụng làm cơ sở xây dựng baseline. |
| `reports/` | Báo cáo tổng hợp về hiệu suất, tốc độ và tài nguyên. |
| `tests/` | Kiểm thử artifact, tiền xử lý, khôi phục model và dự đoán mẫu. |
| `training/` | Notebook xây dựng, huấn luyện và xuất artifact. |
| `var/` | Kết quả dự đoán và telemetry sinh ra khi chạy testbed. |
| `README.md` | Hướng dẫn cài đặt, cấu hình, kiểm thử và chạy baseline. |
| `requirements.txt` | Danh sách dependency Python của project. |
| `run.py` | Điểm khởi chạy Controller, Router hoặc Worker. |
| `setup_env.sh` | Script tạo môi trường ảo và cài dependency. |

## Cấu hình thông tin xác thực

Các file cấu hình mẫu đọc URL RabbitMQ từ biến môi trường:

```bash
export DGA_RABBITMQ_URL='amqp://USERNAME:PASSWORD@RABBITMQ_HOST:5672/%2F'
```

Không lưu mật khẩu hoặc token thật trong repository. Queue prefix và artifact token có thể được thiết lập theo hướng dẫn trong README của từng baseline.

## Git LFS

Repository sử dụng Git LFS cho artifact, paper PDF và kết quả JSONL có dung lượng lớn. Sau khi clone, chạy:

```bash
git lfs install
git lfs pull
```

Có thể kiểm tra các file được quản lý bởi LFS bằng:

```bash
git lfs ls-files
```

## Môi trường chạy

Mỗi baseline cần một virtual environment riêng. Không sao chép `.venv` giữa các project hoặc giữa các máy. Thực hiện cài đặt theo `setup_env.sh` và `README.md` trong project tương ứng.
