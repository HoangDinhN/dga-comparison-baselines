from __future__ import annotations

import re
import string
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import tldextract


_NON_SLD_RE = re.compile(r"[^a-z0-9-]+")
_DEFAULT_ALPHABET = string.ascii_lowercase + string.digits + ".-"


class HybridCNNLSTM(nn.Module):
    """Bilbo v2 architecture reproduced from bilbo_v2.ipynb."""

    def __init__(
        self,
        embedding_dim: int = 128,
        max_string_length: int = 40,
        num_filters: int = 128,
        lstm_embedding_dim: int = 128,
        lstm_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_cnn = nn.Embedding(max_string_length, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size)
            for kernel_size in (2, 3, 4, 5, 6)
        )
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(5 * num_filters, num_filters)

        self.embedding_lstm = nn.Embedding(
            max_string_length,
            lstm_embedding_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=lstm_embedding_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.extra_dense = nn.Linear(num_filters + lstm_hidden, 100)
        self.output_layer = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_embedding = self.embedding_cnn(x).permute(0, 2, 1)
        pooled = [
            F.adaptive_max_pool1d(F.relu(conv(cnn_embedding)), 1).squeeze(-1)
            for conv in self.convs
        ]
        cnn_out = self.dropout(torch.cat(pooled, dim=1))
        cnn_out = self.dropout(F.relu(self.dense(cnn_out)))

        lstm_embedding = self.embedding_lstm(x)
        _, (hidden, _) = self.lstm(lstm_embedding)
        lstm_out = self.dropout(hidden.squeeze(0))

        combined = self.dropout(torch.cat([cnn_out, lstm_out], dim=1))
        combined = self.dropout(F.relu(self.extra_dense(combined)))
        return torch.sigmoid(self.output_layer(combined))


class BilboDGA:
    """Deployment wrapper exposing the testbed's predict_many contract."""

    def __init__(self, checkpoint_path: str | None, *, strict: bool = True) -> None:
        if not checkpoint_path:
            raise ValueError("Bilbo checkpoint path is required")
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if strict and not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Bilbo checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            # The training notebook stores sklearn/NumPy metric scalars beside
            # the state dict. Some PyTorch versions reject those values in
            # weights-only mode, although they are not used by inference.
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # Compatibility with older PyTorch releases without weights_only.
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Invalid Bilbo checkpoint: missing model_state_dict")

        self.model = HybridCNNLSTM()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.model.to(self.device)
        self.model.eval()

        self.max_len = 32
        self.char2idx = {character: index + 1 for index, character in enumerate(_DEFAULT_ALPHABET)}
        self.threshold = 0.5
        self.model_name = "Bilbo_v2"

    @staticmethod
    def normalize_sld(domain: str) -> str:
        if not isinstance(domain, str):
            return ""
        extracted = tldextract.extract(domain)
        return _NON_SLD_RE.sub("", extracted.domain.lower())

    def domain_to_char_ids(self, domain: str) -> list[int]:
        text = str(domain)[: self.max_len]
        values = [self.char2idx.get(character, 0) for character in text]
        return values + [0] * (self.max_len - len(values))

    def predict_many(self, domains: list[str]) -> list[dict[str, Any]]:
        if not domains:
            return []

        encoded = [
            self.domain_to_char_ids(self.normalize_sld(domain))
            for domain in domains
        ]
        inputs = torch.tensor(encoded, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            probabilities = self.model(inputs).squeeze(1).detach().cpu().tolist()

        results: list[dict[str, Any]] = []
        for probability in probabilities:
            score = float(probability)
            prediction = int(score > self.threshold)
            results.append(
                {
                    "prediction": prediction,
                    "label": "DGA" if prediction == 1 else "Benign",
                    "score": score,
                    "model": self.model_name,
                }
            )
        return results
