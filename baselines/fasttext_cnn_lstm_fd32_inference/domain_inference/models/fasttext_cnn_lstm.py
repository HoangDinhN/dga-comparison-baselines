"""Inference for the folded deployment bundle exported by the FD32 notebook."""

from __future__ import annotations

import math
import string
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


ALPHABET = string.ascii_lowercase + string.digits + ".-_"
MAX_LEN = 32
PAD_ID = 0
UNK_ID = 1
MODEL_DIM = 128
CHAR_TO_ID = {character: index + 2 for index, character in enumerate(ALPHABET)}
VOCAB_SIZE = len(CHAR_TO_ID) + 2


class FastTextCNNLSTM32Folded(nn.Module):
    """Exact deployment architecture exported by the training notebook."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, MODEL_DIM, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            nn.Conv1d(MODEL_DIM, 256, kernel_size)
            for kernel_size in range(2, 11)
        )
        self.cnn_bn = nn.BatchNorm1d(256 * 9)
        self.cnn_dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(MODEL_DIM, 128, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256 * 9 + 128, 1)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)

        channel_first = embedded.transpose(1, 2)
        cnn_features = torch.cat(
            [
                F.adaptive_max_pool1d(F.relu(conv(channel_first)), 1).squeeze(-1)
                for conv in self.convs
            ],
            dim=1,
        )
        cnn_features = self.cnn_dropout(self.cnn_bn(cnn_features))

        # pack_padded_sequence requires CPU lengths. This selects the LSTM state
        # at the final real character instead of the right-padding position.
        safe_lengths = lengths.to(dtype=torch.long, device="cpu").clamp(1, MAX_LEN)
        packed = pack_padded_sequence(
            embedded,
            safe_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        lstm_features = self.lstm_dropout(hidden[-1])
        return self.classifier(
            torch.cat([cnn_features, lstm_features], dim=1)
        ).squeeze(1)


class FastTextCNNLSTMFD32DGA:
    """Expose the notebook model through the testbed ``predict_many`` contract."""

    def __init__(self, bundle_path: str | None, *, strict: bool = True):
        if not bundle_path:
            raise ValueError("FastText CNN-LSTM FD32 bundle path is required")
        self.bundle_path = Path(bundle_path).expanduser().resolve()
        if not self.bundle_path.is_file():
            raise FileNotFoundError(
                f"FastText CNN-LSTM FD32 bundle not found: {self.bundle_path}"
            )

        try:
            bundle = torch.load(self.bundle_path, map_location="cpu", weights_only=True)
        except TypeError:
            bundle = torch.load(self.bundle_path, map_location="cpu")

        if not isinstance(bundle, dict) or not isinstance(bundle.get("model_state_dict"), dict):
            raise ValueError("Invalid FastText CNN-LSTM FD32 deployment bundle")
        preprocessing = bundle.get("preprocessing")
        model_config = bundle.get("model_config")
        if not isinstance(preprocessing, dict) or not isinstance(model_config, dict):
            raise ValueError("Bundle lacks preprocessing or model configuration")

        expected_preprocessing = {
            "input": "full_domain_lowercase_strip_trailing_dot_keep_rightmost_32",
            "max_len": MAX_LEN,
            "truncation": "rightmost",
            "alphabet": ALPHABET,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "char_to_id": CHAR_TO_ID,
            "length": "min(len(normalized_domain), 32)",
        }
        expected_model = {
            "type": "FastTextCNNLSTM32Folded",
            "embedding_dim": MODEL_DIM,
            "pretrained_embedding_dim_during_training": 300,
            "embedding_projection_folded": True,
            "cnn_kernels": list(range(2, 11)),
            "cnn_filters": 256,
            "lstm_hidden": 128,
            "dropout": 0.5,
            "lstm_sequence_handling": "pack_padded_sequence_last_valid_hidden",
            "fusion": "concat_cnn_pool_lstm_last_valid_hidden",
            "output": "one_logit",
        }
        for key, expected in expected_preprocessing.items():
            if preprocessing.get(key) != expected:
                raise ValueError(
                    f"Unsupported preprocessing {key}: {preprocessing.get(key)!r}"
                )
        for key, expected in expected_model.items():
            if model_config.get(key) != expected:
                raise ValueError(
                    f"Unsupported model configuration {key}: {model_config.get(key)!r}"
                )

        if "best_threshold" not in bundle:
            raise ValueError("Bundle has no validation-selected threshold")
        self.threshold = float(bundle["best_threshold"])
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Invalid threshold: {self.threshold}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FastTextCNNLSTM32Folded()
        self.model.load_state_dict(bundle["model_state_dict"], strict=strict)
        self.model.to(self.device).eval()
        self.model_name = "FastText_CNN_LSTM_FD32_300to128_LastValid"

    @staticmethod
    def normalize_domain(domain: str) -> str:
        if not isinstance(domain, str):
            raise TypeError("domain must be a string")
        return domain.strip().lower().rstrip(".")

    def encode_domain(self, domain: str) -> tuple[list[int], int]:
        normalized = self.normalize_domain(domain)
        if not normalized:
            raise ValueError("domain is empty after normalization")
        clipped = normalized[-MAX_LEN:]
        ids = [CHAR_TO_ID.get(character, UNK_ID) for character in clipped]
        length = len(ids)
        return ids + [PAD_ID] * (MAX_LEN - length), length

    def domain_to_char_ids(self, domain: str) -> list[int]:
        return self.encode_domain(domain)[0]

    def predict_many(self, domains: list[str]) -> list[dict[str, Any]]:
        if not domains:
            return []

        encoded = [self.encode_domain(domain) for domain in domains]
        token_ids = torch.tensor(
            [item[0] for item in encoded],
            dtype=torch.long,
            device=self.device,
        )
        lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(token_ids, lengths)
            probabilities = torch.sigmoid(logits.float()).cpu().tolist()

        results = []
        for probability in probabilities:
            score = float(probability)
            prediction = int(score >= self.threshold)
            results.append(
                {
                    "prediction": prediction,
                    "label": "DGA" if prediction else "Benign",
                    "score": score,
                    "model": self.model_name,
                }
            )
        return results
