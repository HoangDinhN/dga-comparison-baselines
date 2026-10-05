from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from .fanci_features import (
    FEATURE_NAMES,
    OfficialFANCI45Extractor,
    extract_features_batch,
    normalize_domain,
)


class FANCIModel:
    MODEL_NAME = "FANCI_RF_Official45_150Trees"
    MODEL_TYPE = "fanci_random_forest_official45_150trees"
    EXTRACTOR_VERSION = "official-fanci-d6c7d08-45d-v1"
    PSL_SHA256 = "ce5c21dd349eb4d93635566d8fd15287ce31e9dc887df3c594a0aef0f4d2ade0"
    TREE_COUNT = 150
    SOURCE_BUNDLE_SHA256 = "952032ee5f1c0fec3190720adb1f34df0b522036de2a99f3d423b3a3d380bae5"

    def __init__(
        self,
        bundle_path: str | None,
        *,
        strict: bool = True,
        feature_n_jobs: int = 1,
        rf_n_jobs: int | None = None,
        threshold_mode: str = "best",
    ):
        if not bundle_path:
            raise ValueError("FANCI bundle path is required")
        self.bundle_path = Path(bundle_path).expanduser().resolve()
        if strict and not self.bundle_path.is_file():
            raise FileNotFoundError(f"FANCI bundle not found: {self.bundle_path}")

        started = time.perf_counter()
        bundle = joblib.load(self.bundle_path)
        self.load_seconds = time.perf_counter() - started
        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError("Invalid FANCI deployment bundle")

        self.bundle = bundle
        self.model = bundle["model"]
        self.metadata = dict(bundle.get("metadata") or {})
        self.feature_n_jobs = int(feature_n_jobs)

        self._validate_bundle(strict=strict)
        self.extractor = OfficialFANCI45Extractor(
            bundle["valid_tlds"], bundle["valid_public_suffixes"]
        )

        if rf_n_jobs is not None and hasattr(self.model, "n_jobs"):
            self.model.n_jobs = int(rf_n_jobs)

        if threshold_mode == "best":
            self.threshold = float(bundle["best_threshold"])
        elif threshold_mode == "paper":
            self.threshold = float(bundle.get("paper_threshold", 0.5))
        else:
            raise ValueError("threshold_mode must be 'best' or 'paper'")

        classes = np.asarray(self.model.classes_)
        positions = np.where(classes == 1)[0]
        if len(positions) != 1:
            raise RuntimeError(f"Positive class 1 is not unique in classes={classes}")
        self._positive_index = int(positions[0])

    def _validate_bundle(self, *, strict: bool) -> None:
        checks = {
            "model_type": (self.bundle.get("model_type"), self.MODEL_TYPE),
            "feature_count": (self.bundle.get("feature_count"), 45),
            "extractor_version": (self.bundle.get("extractor_version"), self.EXTRACTOR_VERSION),
            "psl_sha256": (self.bundle.get("psl_sha256"), self.PSL_SHA256),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                raise RuntimeError(f"FANCI {label} mismatch: expected={expected!r}, actual={actual!r}")

        if list(self.bundle.get("feature_names") or []) != FEATURE_NAMES:
            raise RuntimeError("Feature name/order mismatch between runtime extractor and bundle")
        if not self.bundle.get("valid_tlds") or not self.bundle.get("valid_public_suffixes"):
            raise RuntimeError("FANCI bundle is missing Public Suffix List data")
        if int(getattr(self.model, "n_features_in_", -1)) != 45:
            raise RuntimeError(
                f"Expected a 45-feature model, got {getattr(self.model, 'n_features_in_', None)}"
            )
        tree_count = len(getattr(self.model, "estimators_", []))
        if tree_count != self.TREE_COUNT or int(getattr(self.model, "n_estimators", -1)) != self.TREE_COUNT:
            raise RuntimeError(f"Expected {self.TREE_COUNT} trees, got {tree_count}")
        derivation = dict(self.metadata.get("derivation") or {})
        if derivation.get("retrained") is not False:
            raise RuntimeError("Bundle must identify the 150-tree variant as derived without retraining")
        if derivation.get("source_bundle_sha256") != self.SOURCE_BUNDLE_SHA256:
            raise RuntimeError("Unexpected source bundle for the 150-tree variant")

        saved_version = str(self.metadata.get("sklearn_version", ""))
        if strict and saved_version and saved_version != sklearn.__version__:
            raise RuntimeError(
                f"scikit-learn version mismatch: bundle={saved_version}, runtime={sklearn.__version__}. "
                "Install scikit-learn==1.6.1."
            )

    def predict_many(self, domains: list[str]) -> list[dict[str, Any]]:
        if not domains:
            return []
        normalized = [normalize_domain(domain) for domain in domains]
        features = extract_features_batch(
            normalized, self.extractor, n_jobs=self.feature_n_jobs
        )
        probabilities = self.model.predict_proba(features)
        scores = np.asarray(probabilities[:, self._positive_index], dtype=float).reshape(-1)
        predictions = (scores >= self.threshold).astype(np.int8)
        return [
            {
                "prediction": int(prediction),
                "label": "DGA" if prediction == 1 else "Benign",
                "score": float(score),
                "threshold": self.threshold,
                "model": self.MODEL_NAME,
            }
            for prediction, score in zip(predictions, scores)
        ]

