from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[position])


@dataclass
class MetricsBuffer:
    client_id: str
    role: str = "waiting"
    accuracy_enabled: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _count: int = 0
    _dga: int = 0
    _correct: int = 0
    _labeled: int = 0
    _errors: int = 0
    _inference_ms: list[float] = field(default_factory=list, init=False, repr=False)
    _e2e_ms: list[float] = field(default_factory=list, init=False, repr=False)
    _started_at: float = field(default_factory=time.time, init=False)

    def set_role(self, role: str, accuracy_enabled: bool) -> None:
        with self._lock:
            self.role = role
            self.accuracy_enabled = accuracy_enabled

    def record(
        self,
        *,
        prediction: int,
        truth: int | None,
        inference_ms: float,
        e2e_ms: float,
    ) -> None:
        with self._lock:
            self._count += 1
            self._dga += int(prediction == 1)
            self._inference_ms.append(float(inference_ms))
            self._e2e_ms.append(float(e2e_ms))
            if self.accuracy_enabled and truth in (0, 1):
                self._labeled += 1
                self._correct += int(prediction == truth)

    def error(self) -> None:
        with self._lock:
            self._errors += 1

    def drain(self) -> dict[str, Any] | None:
        with self._lock:
            now = time.time()
            if self._count == 0 and self._errors == 0:
                # Close idle windows too, so throughput isn't divided by idle time.
                self._started_at = now
                return None
            elapsed = max(now - self._started_at, 1e-6)
            report = {
                "type": "metrics",
                "client_id": self.client_id,
                "role": self.role,
                "window_started_at": self._started_at,
                "window_ended_at": now,
                "processed": self._count,
                "errors": self._errors,
                "throughput_per_second": round(self._count / elapsed, 3),
                "dga_predictions": self._dga,
                "benign_predictions": self._count - self._dga,
                "inference_ms_avg": round(sum(self._inference_ms) / max(len(self._inference_ms), 1), 3),
                "inference_ms_p95": round(percentile(self._inference_ms, 0.95), 3),
                "end_to_end_ms_avg": round(sum(self._e2e_ms) / max(len(self._e2e_ms), 1), 3),
                "end_to_end_ms_p95": round(percentile(self._e2e_ms, 0.95), 3),
                "accuracy_enabled": self.accuracy_enabled,
                "labeled": self._labeled,
                "correct": self._correct,
                "accuracy": round(self._correct / self._labeled, 6) if self._labeled else None,
            }
            self._count = self._dga = self._correct = self._labeled = self._errors = 0
            self._inference_ms.clear()
            self._e2e_ms.clear()
            self._started_at = now
            return report