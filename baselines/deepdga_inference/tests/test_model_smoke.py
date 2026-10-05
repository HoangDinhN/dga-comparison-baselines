from pathlib import Path

import numpy as np
import torch

from domain_inference.models.deepdga import DeepDGADGA


def _origin_artifact_paths() -> tuple[Path, Path, Path]:
    artifact_dir = Path(__file__).resolve().parents[1] / "artifact"
    original = (
        artifact_dir / "deepDGA_original_bundle.pt",
        artifact_dir / "deepDGA_original_keyedvectors.kv",
        artifact_dir / "deepDGA_original_tfidf.joblib",
    )
    if all(path.is_file() for path in original):
        return original
    return (
        artifact_dir / "deepDGA_best_bundle.pt",
        artifact_dir / "deepDGA_keyedvectors.kv",
        artifact_dir / "deepDGA_tfidf.joblib",
    )


def test_origin_checkpoint_load_and_predict():
    checkpoint_path, vectors_path, tfidf_path = _origin_artifact_paths()
    model = DeepDGADGA(
        str(checkpoint_path),
        str(vectors_path),
        str(tfidf_path),
    )

    assert model.model_name == "DeepDGA-origin-test-target-stopped"
    assert model.sequence_handling == "full_padded_sequence"
    assert model.threshold == 0.5
    assert model.threshold_operator == ">"
    assert sum(parameter.numel() for parameter in model.model.parameters()) == 358_017

    # The original notebook lowercases only. It neither parses URLs nor rejects
    # unknown characters; unknown characters share padding index 0.
    prepared = model.normalize_domain("HTTP://A_B.COM/Path")
    assert prepared == "http://a_b.com/path"
    encoded, length = model._encode_characters(prepared)
    assert length == len(prepared)
    assert encoded[prepared.index("_")] == 0

    domains = ["google.com", "ajd82ksla9q.net", "A_B.example"]
    out = model.predict_many(domains)
    assert len(out) == len(domains)
    assert all(item["prediction"] in (0, 1) for item in out)
    assert all(0.0 <= item["score"] <= 1.0 for item in out)
    assert all(item["model"] == model.model_name for item in out)

    normalized = [model.normalize_domain(domain) for domain in domains]
    character_rows = [model._encode_characters(domain)[0] for domain in normalized]
    word_rows = [model._aggregate_domain(domain) for domain in normalized]
    x_char = torch.tensor(character_rows, dtype=torch.long, device=model.device)
    x_word = torch.from_numpy(np.stack(word_rows)).to(model.device)
    with torch.inference_mode():
        direct_scores = model.model(x_char, x_word).cpu().numpy()

    np.testing.assert_allclose(
        np.asarray([item["score"] for item in out]),
        direct_scores,
        rtol=1e-6,
        atol=1e-7,
    )
    assert [item["prediction"] for item in out] == [
        int(score > model.threshold) for score in direct_scores
    ]

