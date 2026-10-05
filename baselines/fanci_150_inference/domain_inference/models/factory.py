from __future__ import annotations

from typing import Any

from domain_inference.config import nested


def load_model(
    role: str,
    paths: dict[str, str],
    *,
    strict: bool = True,
    config: dict[str, Any] | None = None,
) -> Any:
    if role != "worker":
        raise ValueError(f"Unsupported model role: {role}")
    from domain_inference.models.fanci import FANCIModel

    config = config or {}
    raw_rf_jobs = nested(config, "workers.worker.rf_n_jobs")
    rf_n_jobs = int(raw_rf_jobs) if raw_rf_jobs is not None else None
    return FANCIModel(
        paths.get("bundle"),
        strict=strict,
        feature_n_jobs=int(nested(config, "workers.worker.feature_n_jobs", 1)),
        rf_n_jobs=rf_n_jobs,
        threshold_mode=str(nested(config, "workers.worker.threshold_mode", "best")),
    )
