from __future__ import annotations

from typing import Any


def load_model(role: str, paths: dict[str, str], *, strict: bool = True) -> Any:
    if role != "worker":
        raise ValueError(f"Unsupported model role: {role}")
    from domain_inference.models.deepdga import DeepDGADGA

    return DeepDGADGA(
        paths.get("checkpoint"),
        paths.get("word2vec"),
        paths.get("tfidf"),
        strict=strict,
    )
