from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    embed_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.3
    num_classes: int = 2


class LSTMMIClassifier(nn.Module):
    """LSTM_MI architecture used by both original and corrected artifacts."""

    def __init__(self, config: ModelConfig, *, sequence_handling: str) -> None:
        super().__init__()
        self.config = config
        self.sequence_handling = sequence_handling
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(config.embed_dim, config.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = self.embedding(token_ids)

        if self.sequence_handling == "full_padded_sequence_last_timestep":
            lstm_out, _ = self.lstm(embedded)
            features = lstm_out[:, -1, :]
        elif self.sequence_handling == "packed_valid_sequence":
            if lengths is None:
                raise ValueError("lengths is required for packed sequences")
            packed = pack_padded_sequence(
                embedded,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
            features = hidden[-1]
        else:
            raise ValueError(
                f"Unsupported LSTM_MI sequence handling: {self.sequence_handling}"
            )

        features = self.dropout(features)
        return self.fc(features)


class LSTMMIDGA:
    """Deployment wrapper exposing the testbed predict_many contract."""

    ORIGINAL_SEQUENCE = "full_padded_sequence_last_timestep"
    FIXED_SEQUENCE = "packed_valid_sequence"
    ORIGINAL_CLEANING = "original_lower_split_scheme_path_port_query_v1"
    FIXED_CLEANING = "urlsplit-hostname-idna-lower-v1"

    def __init__(self, checkpoint_path: str | None, *, strict: bool = True) -> None:
        if not checkpoint_path:
            raise ValueError("LSTM_MI checkpoint path is required")
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if strict and not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"LSTM_MI checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        self._validate_checkpoint(checkpoint)
        model_config = ModelConfig(**dict(checkpoint["model_config"]))
        preprocessing = dict(checkpoint["preprocessing"])

        self.max_len = int(preprocessing["max_len"])
        self.alphabet = str(preprocessing["alphabet"])
        self.pad_idx = int(preprocessing["pad_idx"])
        self.unk_idx = int(preprocessing["unk_idx"])
        self.char2idx = {
            str(character): int(index)
            for character, index in dict(preprocessing["char2idx"]).items()
        }
        self.cleaning = str(preprocessing["cleaning"])
        self.sequence_handling = self._sequence_handling(preprocessing)
        self._validate_preprocessing(model_config)

        self.model = LSTMMIClassifier(
            model_config,
            sequence_handling=self.sequence_handling,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.model.to(self.device)
        self.model.eval()

        if "decision" in checkpoint:
            decision = dict(checkpoint["decision"])
            self.threshold = float(decision["threshold"])
            self.threshold_operator = str(decision["operator"])
            if int(decision.get("positive_label", 1)) != 1:
                raise ValueError("LSTM_MI positive label must be 1")
        else:
            self.threshold = float(checkpoint["threshold"])
            self.threshold_operator = ">="

        if self.threshold_operator not in {">", ">="}:
            raise ValueError(
                f"Unsupported LSTM_MI threshold operator: {self.threshold_operator}"
            )
        self.model_name = str(checkpoint.get("model_name", "LSTM_MI"))

    @staticmethod
    def _validate_checkpoint(checkpoint: Any) -> None:
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Invalid LSTM_MI checkpoint: missing model_state_dict")
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported LSTM_MI checkpoint format")
        if checkpoint.get("label_mapping") != {"0": "Benign", "1": "DGA"}:
            raise ValueError("Invalid LSTM_MI label mapping")
        for key in ("model_config", "preprocessing"):
            if key not in checkpoint:
                raise ValueError(f"Invalid LSTM_MI checkpoint: missing {key}")
        if "decision" not in checkpoint and "threshold" not in checkpoint:
            raise ValueError("Invalid LSTM_MI checkpoint: missing decision threshold")

    def _sequence_handling(self, preprocessing: dict[str, Any]) -> str:
        value = preprocessing.get("sequence_handling")
        if value is not None:
            return str(value)
        if self.cleaning == self.FIXED_CLEANING:
            return self.FIXED_SEQUENCE
        raise ValueError("LSTM_MI checkpoint is missing sequence handling metadata")

    def _validate_preprocessing(self, model_config: ModelConfig) -> None:
        if self.pad_idx != 0:
            raise ValueError("LSTM_MI pad_idx must be 0")
        if self.max_len <= 0:
            raise ValueError("LSTM_MI max_len must be positive")
        if model_config.num_classes != 2:
            raise ValueError("LSTM_MI must have two output classes")
        if model_config.vocab_size != len(self.char2idx) + 2:
            raise ValueError("LSTM_MI character vocabulary size mismatch")
        if set(self.char2idx) != set(self.alphabet):
            raise ValueError("LSTM_MI alphabet and char2idx do not match")
        if set(self.char2idx.values()) != set(range(1, len(self.char2idx) + 1)):
            raise ValueError("LSTM_MI char2idx values are invalid")
        if self.unk_idx != len(self.char2idx) + 1:
            raise ValueError("LSTM_MI unk_idx is invalid")

        expected = {
            self.ORIGINAL_SEQUENCE: self.ORIGINAL_CLEANING,
            self.FIXED_SEQUENCE: self.FIXED_CLEANING,
        }
        expected_cleaning = expected.get(self.sequence_handling)
        if expected_cleaning is None:
            raise ValueError(
                f"Unsupported LSTM_MI sequence handling: {self.sequence_handling}"
            )
        if self.cleaning != expected_cleaning:
            raise ValueError(
                "LSTM_MI cleaning metadata does not match sequence handling"
            )

    def clean_domain(self, value: str) -> str:
        if self.sequence_handling == self.ORIGINAL_SEQUENCE:
            if not isinstance(value, str):
                raise ValueError("Domain must be a string")
            domain = value.lower()
            domain = domain.split("://")[-1]
            domain = domain.split("/")[0]
            domain = domain.split(":")[0]
            domain = domain.split("?")[0]
            return domain

        if not isinstance(value, str) or not value.strip():
            raise ValueError("Domain must be a non-empty string")

        raw = value.strip()
        parsed = urlsplit(raw if "://" in raw else "//" + raw)
        if parsed.scheme and parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URLs containing username/password are not supported")

        host = (parsed.hostname or "").rstrip(".").lower()
        if not host:
            raise ValueError("No hostname found")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValueError("IP address is not a domain")

        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as error:
            raise ValueError("Invalid Unicode/IDNA domain") from error
        if len(host) > 253:
            raise ValueError("Hostname exceeds 253 characters")
        return host

    def encode_domain(self, domain: str) -> tuple[list[int], int]:
        cleaned = self.clean_domain(domain)
        values = [
            self.char2idx.get(character, self.unk_idx)
            for character in cleaned[: self.max_len]
        ]
        length = len(values)
        values += [self.pad_idx] * (self.max_len - length)
        return values, length

    def _predict_positive(self, score: float) -> bool:
        if self.threshold_operator == ">":
            return score > self.threshold
        return score >= self.threshold

    def predict_many(self, domains: list[str]) -> list[dict[str, Any]]:
        if not domains:
            return []

        encoded = [self.encode_domain(domain) for domain in domains]
        token_ids = torch.tensor(
            [values for values, _length in encoded],
            dtype=torch.long,
            device=self.device,
        )
        lengths = torch.tensor(
            [length for _values, length in encoded],
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            logits = self.model(token_ids, lengths)
            probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()

        results: list[dict[str, Any]] = []
        for probability in probabilities:
            score = float(probability)
            prediction = int(self._predict_positive(score))
            results.append(
                {
                    "prediction": prediction,
                    "label": "DGA" if prediction == 1 else "Benign",
                    "score": score,
                    "model": self.model_name,
                }
            )
        return results

