from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib

from domain_inference.models.fanci import FANCIModel
from domain_inference.models.fanci_features import FEATURE_NAMES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a FANCI RF Official45 150-tree deployment bundle")
    parser.add_argument(
        "bundle",
        nargs="?",
        default="artifact/fanci_rf_official45_150trees_bundle.joblib",
    )
    args = parser.parse_args()
    path = Path(args.bundle).expanduser().resolve()
    bundle = joblib.load(path)
    model = bundle["model"]

    print("path:", path)
    print("size_bytes:", path.stat().st_size)
    print("sha256:", sha256_file(path))
    print("format_version:", bundle.get("format_version"))
    print("model_type:", bundle.get("model_type"))
    print("extractor_version:", bundle.get("extractor_version"))
    print("feature_count:", bundle.get("feature_count"))
    print("n_features_in_:", getattr(model, "n_features_in_", None))
    print("trees:", len(getattr(model, "estimators_", [])))
    print("best_threshold:", bundle.get("best_threshold"))
    print("paper_threshold:", bundle.get("paper_threshold"))
    print("sklearn_version:", (bundle.get("metadata") or {}).get("sklearn_version"))
    print("psl_sha256:", bundle.get("psl_sha256"))

    assert bundle.get("model_type") == FANCIModel.MODEL_TYPE
    assert bundle.get("feature_count") == 45
    assert list(bundle.get("feature_names") or []) == FEATURE_NAMES
    assert getattr(model, "n_features_in_", None) == 45
    print("PASS: bundle metadata matches the Official45 runtime.")


if __name__ == "__main__":
    main()
