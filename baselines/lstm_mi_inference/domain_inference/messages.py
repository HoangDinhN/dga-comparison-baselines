from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueueNames:
    prefix: str = "dga.lstm_mi"

    @property
    def jobs_exchange(self) -> str:
        return f"{self.prefix}.jobs"

    @property
    def worker_jobs(self) -> str:
        return f"{self.prefix}.jobs.worker"

    @property
    def register(self) -> str:
        return f"{self.prefix}.register"

    @property
    def telemetry(self) -> str:
        return f"{self.prefix}.telemetry"

    @property
    def results(self) -> str:
        return f"{self.prefix}.results"

    def control(self, client_id: str) -> str:
        return f"{self.prefix}.control.{client_id}"

    def queue_for_role(self, role: str) -> str:
        if role == "worker":
            return self.worker_jobs
        raise ValueError(f"Role {role!r} has no inference queue")


def encode_message(message: dict[str, Any]) -> bytes:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_message(body: bytes) -> dict[str, Any]:
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RabbitMQ payload must be a JSON object")
    return value


def utc_timestamp() -> float:
    return time.time()
