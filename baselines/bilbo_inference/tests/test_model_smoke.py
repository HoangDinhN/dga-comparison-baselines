from pathlib import Path

from domain_inference.models.bilbo import BilboDGA


def test_checkpoint_load_and_predict():
    checkpoint = Path(__file__).resolve().parents[1] / "artifact" / "bilbo.pt_ver2"
    model = BilboDGA(str(checkpoint))
    out = model.predict_many(["google.com", "ajd82ksla9q.net"])
    assert len(out) == 2
    assert all(item["prediction"] in (0, 1) for item in out)
