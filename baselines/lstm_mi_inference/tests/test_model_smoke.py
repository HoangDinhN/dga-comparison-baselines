from pathlib import Path

import numpy as np
import torch

from domain_inference.models.lstm_mi import LSTMMIDGA


def _original_checkpoint_path() -> Path:
    artifact_dir = Path(__file__).resolve().parents[1] / "artifact"
    original = artifact_dir / "best_model_LSTM_MI_original.pt"
    if original.is_file():
        return original
    return artifact_dir / "best_model_LSTM_MI.pt"


def test_original_checkpoint_load_and_predict():
    model = LSTMMIDGA(str(_original_checkpoint_path()))

    assert model.model_name == "LSTM_MI-origin-test-target-stopped"
    assert model.sequence_handling == "full_padded_sequence_last_timestep"
    assert model.cleaning == "original_lower_split_scheme_path_port_query_v1"
    assert model.threshold == 0.5
    assert model.threshold_operator == ">"

    # Preserve the original string-splitting behavior, including a trailing dot.
    cleaned = model.clean_domain("HTTP://A_B.COM.:443/path?q=1")
    assert cleaned == "a_b.com."
    encoded, length = model.encode_domain("Ábc.com")
    assert length == len("ábc.com")
    assert encoded[0] == model.unk_idx

    domains = ["google.com", "ajd82ksla9q.net", "A_B.example"]
    out = model.predict_many(domains)
    assert len(out) == len(domains)
    assert all(item["prediction"] in (0, 1) for item in out)
    assert all(0.0 <= item["score"] <= 1.0 for item in out)
    assert all(item["model"] == model.model_name for item in out)

    rows = [model.encode_domain(domain)[0] for domain in domains]
    token_ids = torch.tensor(rows, dtype=torch.long, device=model.device)
    with torch.inference_mode():
        direct_scores = (
            torch.softmax(model.model(token_ids), dim=1)[:, 1].cpu().numpy()
        )

    np.testing.assert_allclose(
        np.asarray([item["score"] for item in out]),
        direct_scores,
        rtol=1e-6,
        atol=1e-7,
    )
    assert [item["prediction"] for item in out] == [
        int(score > model.threshold) for score in direct_scores
    ]

