from __future__ import annotations

import csv
import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from domain_inference.config import nested, require
from domain_inference.messages import decode_message, utc_timestamp
from domain_inference.metrics import MetricsBuffer, percentile
from domain_inference.rabbit import connect, declare_topology, publish_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainRecord:
    source_id: str
    domain: str
    truth: int | None
    subclass: str | None


class RandomCsvSource:
    """Shuffle without replacement inside each CSV epoch."""

    def __init__(self, path: str, seed: int | None = None):
        self.path = Path(path).expanduser().resolve()
        self.random = random.Random(seed)
        self.records = self._load()

    def _load(self) -> list[DomainRecord]:
        records: list[DomainRecord] = []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "domain_name" not in reader.fieldnames:
                raise ValueError(f"CSV must contain column 'domain_name': {self.path}")
            for position, row in enumerate(reader):
                domain = str(row.get("domain_name", "")).strip()
                if not domain:
                    continue
                raw_truth = row.get("label")
                truth = int(raw_truth) if raw_truth in {"0", "1", 0, 1} else None
                records.append(DomainRecord(
                    source_id=str(row.get("") or position),
                    domain=domain,
                    truth=truth,
                    subclass=row.get("subclass"),
                ))
        if not records:
            raise ValueError(f"No domains found in {self.path}")
        LOGGER.info("Loaded %d domains from %s", len(records), self.path)
        return records

    def batches(self, batch_size: int, stop_event: threading.Event, *, loop_forever: bool,
                max_records: int | None) -> Iterator[tuple[int, list[DomainRecord]]]:
        emitted = 0
        epoch = 0
        while not stop_event.is_set():
            indexes = list(range(len(self.records)))
            self.random.shuffle(indexes)
            for offset in range(0, len(indexes), batch_size):
                if stop_event.is_set() or (max_records is not None and emitted >= max_records):
                    return
                selection = indexes[offset:offset + batch_size]
                if max_records is not None:
                    selection = selection[:max_records - emitted]
                batch = [self.records[index] for index in selection]
                emitted += len(batch)
                yield epoch, batch
            epoch += 1
            if not loop_forever:
                return


class RouterMetricsBuffer:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.role = "router"
        self.accuracy_enabled = False
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._routed = 0
        self._errors = 0
        self._latencies: list[float] = []

    def set_role(self, role: str, accuracy_enabled: bool) -> None:
        self.accuracy_enabled = accuracy_enabled

    def record_routes(self, count: int, elapsed_ms: float) -> None:
        with self._lock:
            self._routed += int(count)
            each = elapsed_ms / max(count, 1)
            self._latencies.extend([each] * count)

    def error(self) -> None:
        with self._lock:
            self._errors += 1

    def drain(self) -> dict[str, Any] | None:
        with self._lock:
            now = time.time()
            if self._routed == 0 and self._errors == 0:
                self._started_at = now
                return None
            elapsed = max(now - self._started_at, 1e-6)
            report = {
                "type": "metrics",
                "client_id": self.client_id,
                "role": "router",
                "window_started_at": self._started_at,
                "window_ended_at": now,
                "routed": self._routed,
                "processed": self._routed,
                "errors": self._errors,
                "routed_worker": self._routed,
                "throughput_per_second": round(self._routed / elapsed, 3),
                "routing_ms_avg": round(sum(self._latencies) / max(len(self._latencies), 1), 3),
                "routing_ms_p95": round(percentile(self._latencies, 0.95), 3),
            }
            self._routed = self._errors = 0
            self._latencies.clear()
            self._started_at = now
            return report


class RouterRunner:
    """Single-branch router: every domain is published to jobs.worker."""

    def __init__(self, config: dict[str, Any], client_id: str, metrics: RouterMetricsBuffer,
                 stop_event: threading.Event, accuracy_enabled: bool):
        self.config = config
        self.client_id = client_id
        self.metrics = metrics
        self.stop_event = stop_event
        self.accuracy_enabled = accuracy_enabled

    def run(self) -> None:
        source = RandomCsvSource(
            str(require(self.config, "router_input.csv_path")),
            nested(self.config, "router_input.random_seed"),
        )
        batch_size = int(nested(self.config, "router_input.batch_size", 256))
        loop_forever = bool(nested(self.config, "router_input.loop_forever", False))
        max_records_raw = nested(self.config, "router_input.max_records")
        max_records = int(max_records_raw) if max_records_raw is not None else None
        target_rate = float(nested(self.config, "router_input.max_publish_rate_per_second", 0))
        max_backlog = int(nested(self.config, "router_input.max_total_queue_depth", 50_000))

        connection = connect(self.config, self.stop_event)
        try:
            channel = connection.channel()
            names = declare_topology(channel, self.config, include_results=True)
            if bool(nested(self.config, "router_input.publisher_confirms", False)):
                channel.confirm_delivery()
            published = 0
            started_at = time.monotonic()

            for epoch, batch in source.batches(
                batch_size, self.stop_event, loop_forever=loop_forever, max_records=max_records,
            ):
                self._wait_for_capacity(channel, names, max_backlog)
                if self.stop_event.is_set():
                    break

                route_started = time.perf_counter()
                branches = ["worker"] * len(batch)
                routing_ms = (time.perf_counter() - route_started) * 1000
                self.metrics.record_routes(len(batch), routing_ms)

                created_at = utc_timestamp()
                for record, branch in zip(batch, branches):
                    job = {
                        "type": "inference_job",
                        "job_id": f"{self.client_id}:{epoch}:{record.source_id}:{uuid.uuid4().hex[:8]}",
                        "domain": record.domain,
                        "source_id": record.source_id,
                        "subclass": record.subclass,
                        "router_client_id": self.client_id,
                        "route": branch,
                        "created_at": created_at,
                    }
                    if self.accuracy_enabled and record.truth in (0, 1):
                        job["truth"] = record.truth
                    publish_json(channel, self.config, names.jobs_exchange, "worker", job)
                    published += 1

                if target_rate > 0:
                    expected = published / target_rate
                    remaining = expected - (time.monotonic() - started_at)
                    if remaining > 0:
                        self.stop_event.wait(remaining)

            LOGGER.info("Router source completed after publishing %d domains", published)
        finally:
            if connection.is_open:
                connection.close()

    def _wait_for_capacity(self, channel, names, max_backlog: int) -> None:
        if max_backlog <= 0:
            return
        while not self.stop_event.is_set():
            backlog = channel.queue_declare(queue=names.worker_jobs, passive=True).method.message_count
            if backlog < max_backlog:
                return
            self.stop_event.wait(0.05)


class WorkerRunner:
    def __init__(self, config: dict[str, Any], client_id: str, model,
                 metrics: MetricsBuffer, stop_event: threading.Event,
                 accuracy_enabled: bool, results_enabled: bool):
        self.config = config
        self.client_id = client_id
        self.role = "worker"
        self.model = model
        self.metrics = metrics
        self.stop_event = stop_event
        self.accuracy_enabled = accuracy_enabled
        self.results_enabled = results_enabled

    def run(self) -> None:
        connection = connect(self.config, self.stop_event)
        pending: list[tuple[Any, dict[str, Any]]] = []
        try:
            channel = connection.channel()
            names = declare_topology(channel, self.config, include_results=True)
            role_config = nested(self.config, "workers.worker", {})
            batch_size = int(role_config.get("batch_size", 32))
            batch_wait = float(role_config.get("batch_wait_ms", 5)) / 1000.0
            prefetch = int(role_config.get("prefetch", batch_size * 2))
            channel.basic_qos(prefetch_count=max(prefetch, batch_size))
            consumer = channel.consume(
                names.worker_jobs,
                inactivity_timeout=batch_wait,
                auto_ack=False,
            )
            for method, _properties, body in consumer:
                if method is not None:
                    try:
                        job = decode_message(body)
                        if not str(job.get("domain", "")).strip():
                            raise ValueError("Inference job is missing domain")
                        pending.append((method, job))
                    except Exception:
                        LOGGER.exception("Dropping malformed inference job")
                        channel.basic_reject(method.delivery_tag, requeue=False)
                        self.metrics.error()
                if pending and (len(pending) >= batch_size or method is None or self.stop_event.is_set()):
                    self._process_batch(channel, names, pending)
                    pending.clear()
                if self.stop_event.is_set():
                    break
            if pending:
                self._process_batch(channel, names, pending)
                pending.clear()
            channel.cancel()
        finally:
            if connection.is_open:
                connection.close()

    def _process_batch(self, channel, names, pending: list[tuple[Any, dict[str, Any]]]) -> None:
        jobs = [job for _method, job in pending]
        started = time.perf_counter()
        try:
            predictions = self.model.predict_many([str(job["domain"]) for job in jobs])
            if len(predictions) != len(jobs):
                raise RuntimeError("Model returned a different number of predictions than inputs")
            batch_ms = (time.perf_counter() - started) * 1000
        except Exception:
            self.metrics.error()
            for method, _job in pending:
                channel.basic_nack(method.delivery_tag, requeue=True)
            raise

        per_item_ms = batch_ms / max(len(jobs), 1)
        now = utc_timestamp()
        for index, ((method, job), prediction) in enumerate(zip(pending, predictions)):
            try:
                truth = job.get("truth") if self.accuracy_enabled else None
                result = {
                    "type": "inference_result",
                    "job_id": job.get("job_id"),
                    "source_id": job.get("source_id"),
                    "domain": job.get("domain"),
                    "subclass": job.get("subclass"),
                    "route": "worker",
                    "worker_client_id": self.client_id,
                    "prediction": prediction["prediction"],
                    "label": prediction["label"],
                    "score": prediction["score"],
                    "model": prediction["model"],
                    "created_at": job.get("created_at"),
                    "completed_at": now,
                }
                if truth in (0, 1):
                    result["truth"] = truth
                    result["correct"] = int(prediction["prediction"] == truth)
                if self.results_enabled:
                    publish_json(channel, self.config, "", names.results, result)
                self.metrics.record(
                    prediction=int(prediction["prediction"]),
                    truth=truth if truth in (0, 1) else None,
                    inference_ms=per_item_ms,
                    e2e_ms=max(0.0, (now - float(job.get("created_at", now))) * 1000),
                )
                channel.basic_ack(method.delivery_tag)
            except Exception:
                self.metrics.error()
                for remaining_method, _job in pending[index:]:
                    channel.basic_nack(remaining_method.delivery_tag, requeue=True)
                raise
