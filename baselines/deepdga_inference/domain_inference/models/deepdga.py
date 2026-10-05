from __future__ import annotations

import ipaddress
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import joblib
import numpy as np
import torch
import torch.nn as nn
import wordninja
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    char_embedding_dim: int = 128
    char_lstm_hidden: int = 128
    word_feature_dim: int = 500
    word_hidden_dim: int = 128
    fusion_hidden_dim: int = 64
    dropout: float = 0.5


class DeepDGA(nn.Module):
    """DeepDGA architecture shared by the original and corrected artifacts."""

    def __init__(self, config: ModelConfig, *, sequence_handling: str) -> None:
        super().__init__()
        self.config = config
        self.sequence_handling = sequence_handling
        self.char_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.char_embedding_dim,
            padding_idx=0,
        )
        self.char_lstm = nn.LSTM(
            input_size=config.char_embedding_dim,
            hidden_size=config.char_lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.word_fc = nn.Linear(config.word_feature_dim, config.word_hidden_dim)
        self.fc1 = nn.Linear(
            2 * config.char_lstm_hidden + config.word_hidden_dim,
            config.fusion_hidden_dim,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.fusion_hidden_dim, 1)

    def forward(
        self,
        x_char: torch.Tensor,
        x_word: torch.Tensor,
        char_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = self.char_embedding(x_char)
        if self.sequence_handling == "full_padded_sequence":
            _, (hidden, _) = self.char_lstm(embedded)
        elif self.sequence_handling == "packed_valid_sequence":
            if char_lengths is None:
                raise ValueError("char_lengths is required for packed sequences")
            packed = pack_padded_sequence(
                embedded,
                char_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.char_lstm(packed)
        else:
            raise ValueError(
                f"Unsupported DeepDGA sequence handling: {self.sequence_handling}"
            )

        z_char = torch.cat([hidden[0], hidden[1]], dim=1)
        z_word = self.relu(self.word_fc(x_word))
        z = torch.cat([z_char, z_word], dim=1)
        z = self.dropout(z)
        z = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(z)).squeeze(1)


class DeepDGADGA:
    """Deployment wrapper exposing the testbed predict_many contract."""

    EXPECTED_PARAMETER_COUNT = 358_017
    FULL_PADDED_SEQUENCE = "full_padded_sequence"
    PACKED_VALID_SEQUENCE = "packed_valid_sequence"

    def __init__(
        self,
        checkpoint_path: str | None,
        word2vec_path: str | None,
        tfidf_path: str | None,
        *,
        strict: bool = True,
    ) -> None:
        self.checkpoint_path = self._require_file(checkpoint_path, "checkpoint", strict)
        self.word2vec_path = self._require_file(word2vec_path, "KeyedVectors", strict)
        self.tfidf_path = self._require_file(tfidf_path, "TF-IDF", strict)
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
        raw_model_config = dict(checkpoint["model_config"])
        config_keys = {field.name for field in fields(ModelConfig)}
        model_config = ModelConfig(
            **{key: value for key, value in raw_model_config.items() if key in config_keys}
        )

        preprocessing = dict(checkpoint["preprocessing"])
        self.max_len = int(preprocessing["max_len"])
        self.alphabet = str(preprocessing["alphabet"])
        self.allowed_characters = frozenset(self.alphabet)
        self.char2idx = {
            str(character): int(index)
            for character, index in dict(preprocessing["char2idx"]).items()
        }
        self.sequence_handling = self._sequence_handling(preprocessing)
        self._validate_preprocessing(model_config, preprocessing)

        self.model = DeepDGA(
            model_config,
            sequence_handling=self.sequence_handling,
        )
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        declared_parameter_count = raw_model_config.get("parameter_count")
        if (
            declared_parameter_count is not None
            and int(declared_parameter_count) != parameter_count
        ):
            raise ValueError(
                "DeepDGA checkpoint parameter_count does not match the reconstructed model"
            )
        if parameter_count != self.EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                "DeepDGA architecture mismatch: "
                f"expected {self.EXPECTED_PARAMETER_COUNT:,} parameters, "
                f"got {parameter_count:,}"
            )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.model.to(self.device)
        self.model.eval()

        self.word_vectors = KeyedVectors.load(str(self.word2vec_path), mmap=None)
        self.tfidf = joblib.load(self.tfidf_path)
        if int(self.word_vectors.vector_size) * 5 != model_config.word_feature_dim:
            raise ValueError("KeyedVectors dimension is incompatible with the DeepDGA checkpoint")
        if not hasattr(self.tfidf, "vocabulary_") or not hasattr(self.tfidf, "idf_"):
            raise ValueError("Invalid DeepDGA TF-IDF artifact")

        if "decision" in checkpoint:
            decision = dict(checkpoint["decision"])
            self.threshold = float(decision["threshold"])
            self.threshold_operator = str(decision["operator"])
            if int(decision.get("positive_label", 1)) != 1:
                raise ValueError("DeepDGA positive label must be 1")
        else:
            self.threshold = float(checkpoint["threshold"])
            self.threshold_operator = ">="
        if self.threshold_operator not in {">", ">="}:
            raise ValueError(
                f"Unsupported DeepDGA threshold operator: {self.threshold_operator}"
            )
        self.model_name = str(checkpoint.get("model_name", "DeepDGA"))

    @staticmethod
    def _require_file(value: str | None, label: str, strict: bool) -> Path:
        if not value:
            raise ValueError(f"DeepDGA {label} path is required")
        path = Path(value).expanduser().resolve()
        if strict and not path.is_file():
            raise FileNotFoundError(f"DeepDGA {label} not found: {path}")
        return path

    @staticmethod
    def _validate_checkpoint(checkpoint: Any) -> None:
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Invalid DeepDGA checkpoint: missing model_state_dict")
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported DeepDGA checkpoint format")
        if checkpoint.get("label_mapping") != {"0": "Benign", "1": "DGA"}:
            raise ValueError("Invalid DeepDGA label mapping")
        for key in ("model_config", "preprocessing"):
            if key not in checkpoint:
                raise ValueError(f"Invalid DeepDGA checkpoint: missing {key}")
        if "decision" not in checkpoint and "threshold" not in checkpoint:
            raise ValueError("Invalid DeepDGA checkpoint: missing decision threshold")

    def _sequence_handling(self, preprocessing: dict[str, Any]) -> str:
        sequence_handling = preprocessing.get("sequence_handling")
        if sequence_handling is None:
            if preprocessing.get("normalization") == "urlsplit-hostname-idna-lower-v1":
                return self.PACKED_VALID_SEQUENCE
            raise ValueError("DeepDGA checkpoint is missing sequence handling metadata")
        return str(sequence_handling)

    def _validate_preprocessing(
        self,
        model_config: ModelConfig,
        preprocessing: dict[str, Any],
    ) -> None:
        if model_config.vocab_size != len(self.char2idx) + 1:
            raise ValueError("DeepDGA character vocabulary size mismatch")
        if set(self.char2idx) != self.allowed_characters:
            raise ValueError("DeepDGA alphabet and char2idx do not match")
        if set(self.char2idx.values()) != set(range(1, model_config.vocab_size)):
            raise ValueError("DeepDGA char2idx must reserve index 0 for padding")
        if self.max_len <= 0:
            raise ValueError("DeepDGA max_len must be positive")

        if self.sequence_handling == self.FULL_PADDED_SEQUENCE:
            if preprocessing.get("domain_transform") != "lowercase_then_truncate_75":
                raise ValueError("Unsupported original DeepDGA domain transform")
            if int(preprocessing.get("unknown_character_index", -1)) != 0:
                raise ValueError("Original DeepDGA must map unknown characters to index 0")
            if int(preprocessing.get("padding_index", -1)) != 0:
                raise ValueError("Original DeepDGA padding index must be 0")
            if preprocessing.get("word_tokenizer") != "wordninja_split_each_dot_label":
                raise ValueError("Unsupported original DeepDGA word tokenizer")
            if list(preprocessing.get("word_aggregation", [])) != [
                "min",
                "mean",
                "max",
                "sum",
                "tfidf_mean",
            ]:
                raise ValueError("Unsupported original DeepDGA word aggregation")
            if not preprocessing.get("keyed_vectors_file") or not preprocessing.get(
                "tfidf_file"
            ):
                raise ValueError("DeepDGA checkpoint is missing auxiliary artifact metadata")
            return

        if self.sequence_handling == self.PACKED_VALID_SEQUENCE:
            if preprocessing.get("normalization") != "urlsplit-hostname-idna-lower-v1":
                raise ValueError("Unsupported corrected DeepDGA normalization format")
            if not preprocessing.get("word2vec_file") or not preprocessing.get(
                "tfidf_file"
            ):
                raise ValueError("DeepDGA checkpoint is missing auxiliary artifact metadata")
            return

        raise ValueError(
            f"Unsupported DeepDGA sequence handling: {self.sequence_handling}"
        )

    def normalize_domain(self, value: str) -> str:
        if self.sequence_handling == self.FULL_PADDED_SEQUENCE:
            if not isinstance(value, str):
                raise ValueError("Domain must be a string")
            # The original notebook applies only lower() before character
            # truncation and word splitting. Do not strip or parse the value.
            return value.lower()

        if not isinstance(value, str) or not value.strip():
            raise ValueError("Domain must be a non-empty string")
        raw = value.strip()
        parsed = urlsplit(raw if "://" in raw else "//" + raw)
        if parsed.scheme and parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials are not allowed in domain input")
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host:
            raise ValueError("Hostname is empty")
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
        invalid = sorted(set(host) - self.allowed_characters)
        if invalid:
            raise ValueError(f"Unsupported hostname characters: {invalid}")
        return host

    def _encode_characters(self, domain: str) -> tuple[list[int], int]:
        text = domain[: self.max_len]
        if self.sequence_handling == self.FULL_PADDED_SEQUENCE:
            values = [self.char2idx.get(character, 0) for character in text]
        else:
            values = [self.char2idx[character] for character in text]
        length = len(values)
        values += [0] * (self.max_len - length)
        return values, length

    @staticmethod
    def _domain_to_words(domain: str) -> list[str]:
        tokens: list[str] = []
        for part in domain.lower().split("."):
            tokens.extend(wordninja.split(part))
        return tokens

    def _aggregate_domain(self, domain: str) -> np.ndarray:
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        vocabulary = self.tfidf.vocabulary_
        idf = self.tfidf.idf_
        for token in self._domain_to_words(domain):
            if token not in self.word_vectors:
                continue
            vectors.append(np.asarray(self.word_vectors[token], dtype=np.float32))
            weights.append(float(idf[vocabulary[token]]) if token in vocabulary else 1.0)
        if not vectors:
            return np.zeros(self.model.config.word_feature_dim, dtype=np.float32)

        matrix = np.asarray(vectors, dtype=np.float32)
        weight_dtype = (
            np.float64
            if self.sequence_handling == self.FULL_PADDED_SEQUENCE
            else np.float32
        )
        weight_array = np.asarray(weights, dtype=weight_dtype)
        features = [
            np.min(matrix, axis=0),
            np.mean(matrix, axis=0),
            np.max(matrix, axis=0),
            np.sum(matrix, axis=0),
            np.average(matrix, axis=0, weights=weight_array),
        ]
        return np.concatenate(features).astype(np.float32, copy=False)

    def _predict_positive(self, score: float) -> bool:
        if self.threshold_operator == ">":
            return score > self.threshold
        return score >= self.threshold

    def predict_many(self, domains: list[str]) -> list[dict[str, Any]]:
        if not domains:
            return []

        normalized = [self.normalize_domain(domain) for domain in domains]
        encoded = [self._encode_characters(domain) for domain in normalized]
        x_char = torch.tensor(
            [values for values, _length in encoded],
            dtype=torch.long,
            device=self.device,
        )
        lengths = torch.tensor(
            [length for _values, length in encoded],
            dtype=torch.long,
            device=self.device,
        )
        word_features = np.stack(
            [self._aggregate_domain(domain) for domain in normalized],
            axis=0,
        )
        x_word = torch.from_numpy(word_features).to(self.device)

        with torch.inference_mode():
            probabilities = self.model(x_char, x_word, lengths).detach().cpu().tolist()

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

