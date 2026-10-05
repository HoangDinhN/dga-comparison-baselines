`best_bundle.pt` ở đây phải được xuất bởi
`FastText_CNN_LSTM_FullDomain32_300to128_LastValid.ipynb`.

Chỉ machine-9 cần giữ file này. Controller phân phối bundle theo SHA-256;
Workers tải vào thư mục tạm, nạp model vào RAM rồi xóa bản tải tạm. Router không cần artifact.

Không thay file này bằng bundle 75x300 hoặc checkpoint dùng để resume training.
