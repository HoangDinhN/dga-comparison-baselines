import hashlib
from pathlib import Path

from domain_inference.models.fanci import FANCIModel


EXPECTED_SHA256 = "a0251fd62d21c6fe39adc1b3ae9e51715c7398c48931561a6560c12ccfc5e6c9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bundle_load_and_predict():
    bundle = Path(__file__).resolve().parents[1] / "artifact" / "fanci_rf_official45_150trees_bundle.joblib"
    assert bundle.is_file(), f"Missing FANCI artifact: {bundle}"
    assert sha256_file(bundle) == EXPECTED_SHA256

    model = FANCIModel(str(bundle), strict=True, feature_n_jobs=1, rf_n_jobs=1)
    assert model.model.n_features_in_ == 45
    assert len(model.model.estimators_) == 150
    assert model.model.n_estimators == 150
    assert model.threshold == 0.500924704924705

    output = model.predict_many([
        "google.com",
        "itsec.rwth-aachen.de",
        "6301un092rsh.org",
    ])
    assert len(output) == 3
    assert all(item["prediction"] in (0, 1) for item in output)
    assert all(0.0 <= item["score"] <= 1.0 for item in output)
    assert all(item["model"] == "FANCI_RF_Official45_150Trees" for item in output)
