from __future__ import annotations

from typing import Any


def load_model(role: str, paths: dict[str, str], *, strict: bool = True) -> Any:
    if role != "worker":
        raise ValueError(f"Unsupported model role: {role}")
    from domain_inference.models.fasttext_cnn_lstm import FastTextCNNLSTMFD32DGA

    return FastTextCNNLSTMFD32DGA(paths.get("bundle"), strict=strict)
