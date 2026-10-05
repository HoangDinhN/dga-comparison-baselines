from __future__ import annotations

from pathlib import Path

import pytest
import torch

from domain_inference.models.fasttext_cnn_lstm import (
    ALPHABET,
    CHAR_TO_ID,
    MAX_LEN,
    FastTextCNNLSTM32Folded,
    FastTextCNNLSTMFD32DGA,
)


def _bundle() -> dict:
    model = FastTextCNNLSTM32Folded().eval()
    return {
        "model_state_dict": model.state_dict(),
        "best_threshold": 0.42,
        "preprocessing": {
            "input": "full_domain_lowercase_strip_trailing_dot_keep_rightmost_32",
            "max_len": MAX_LEN,
            "truncation": "rightmost",
            "alphabet": ALPHABET,
            "pad_id": 0,
            "unk_id": 1,
            "char_to_id": CHAR_TO_ID,
            "length": "min(len(normalized_domain), 32)",
        },
        "model_config": {
            "type": "FastTextCNNLSTM32Folded",
            "embedding_dim": 128,
            "pretrained_embedding_dim_during_training": 300,
            "embedding_projection_folded": True,
            "cnn_kernels": list(range(2, 11)),
            "cnn_filters": 256,
            "lstm_hidden": 128,
            "dropout": 0.5,
            "lstm_sequence_handling": "pack_padded_sequence_last_valid_hidden",
            "fusion": "concat_cnn_pool_lstm_last_valid_hidden",
            "output": "one_logit",
        },
    }


def _load_bundle_model(bundle: dict) -> FastTextCNNLSTM32Folded:
    model = FastTextCNNLSTM32Folded()
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    return model.eval()


def test_bundle_load_and_predict_matches_notebook_forward(tmp_path: Path):
    torch.manual_seed(7)
    bundle = _bundle()
    path = tmp_path / "best_bundle.pt"
    torch.save(bundle, path)
    adapter = FastTextCNNLSTMFD32DGA(str(path))

    domains = ["  Sub.Google.COM.  ", "x_y-Z.net", "qé.com"]
    encoded, length = adapter.encode_domain(domains[0])
    expected_text = "sub.google.com"
    assert encoded[:length] == [CHAR_TO_ID[c] for c in expected_text]
    assert encoded[length:] == [0] * (MAX_LEN - length)
    assert adapter.domain_to_char_ids(domains[2])[1] == 1

    long_domain = "abcdefghijklmnopqrstuvwx" + "0123456789.example.com"
    expected_suffix = adapter.normalize_domain(long_domain)[-MAX_LEN:]
    assert adapter.domain_to_char_ids(long_domain) == [
        CHAR_TO_ID.get(c, 1) for c in expected_suffix
    ]

    pairs = [adapter.encode_domain(domain) for domain in domains]
    token_ids = torch.tensor([item[0] for item in pairs], dtype=torch.long)
    lengths = torch.tensor([item[1] for item in pairs], dtype=torch.long)
    with torch.inference_mode():
        expected = torch.sigmoid(_load_bundle_model(bundle)(token_ids, lengths)).tolist()

    results = adapter.predict_many(domains)
    assert adapter.predict_many([]) == []
    assert [item["score"] for item in results] == pytest.approx(expected, abs=1e-5)
    assert [item["prediction"] for item in results] == [int(p >= 0.42) for p in expected]
    assert all(item["model"] == "FastText_CNN_LSTM_FD32_300to128_LastValid" for item in results)


def test_rejects_empty_domain_and_different_preprocessing(tmp_path: Path):
    bundle = _bundle()
    path = tmp_path / "valid.pt"
    torch.save(bundle, path)
    adapter = FastTextCNNLSTMFD32DGA(str(path))
    with pytest.raises(ValueError, match="empty"):
        adapter.predict_many([" . "])

    bundle["preprocessing"]["truncation"] = "leftmost"
    wrong_path = tmp_path / "wrong.pt"
    torch.save(bundle, wrong_path)
    with pytest.raises(ValueError, match="truncation"):
        FastTextCNNLSTMFD32DGA(str(wrong_path))


def test_exported_bundle_when_present():
    path = Path(__file__).resolve().parents[1] / "artifact" / "best_bundle.pt"
    if not path.is_file():
        pytest.skip("Controller-only best_bundle.pt has not been copied here")
    adapter = FastTextCNNLSTMFD32DGA(str(path))
    results = adapter.predict_many(
        ["google.com", "subdomain.example.co.uk", "abe70b78b996.com"]
    )
    assert len(results) == 3
    assert all(item["prediction"] in (0, 1) for item in results)
    assert all(0.0 <= item["score"] <= 1.0 for item in results)
