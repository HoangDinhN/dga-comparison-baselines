from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from domain_inference.artifacts import ArtifactRegistry, ArtifactServer
from domain_inference.config import nested
from domain_inference.messages import decode_message, utc_timestamp
from domain_inference.rabbit import connect, declare_topology, publish_json, queue_names

LOGGER = logging.getLogger(__name__)


@dataclass
class ClientState:
    client_id: str
    preferred_role: str
    capabilities: set[str]
    generation: int = 0
    role: str | None = None
    status: str = "waiting"
    last_seen: float = field(default_factory=utc_timestamp)
    last_switch: float = 0.0


class Controller:
    """Single-branch controller: router + one or more identical worker clients."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.clients: dict[str, ClientState] = {}
        self.started = False
        self.outbound: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()

        self.registry = ArtifactRegistry(config)
        self.artifact_server = ArtifactServer(
            self.registry,
            str(nested(config, "artifact_server.bind_host", "0.0.0.0")),
            int(nested(config, "artifact_server.port", 8080)),
        )
        self.threads: list[threading.Thread] = []
        self._results_handle = None
        self._telemetry_handle = None

    def run(self) -> None:
        self.registry.validate_role("worker")
        self._open_results()
        self._open_telemetry()
        self.artifact_server.start()
        self.threads = [
            threading.Thread(target=self._broker_loop, name="controller-broker", daemon=True),
            threading.Thread(target=self._result_loop, name="controller-results", daemon=True),
        ]
        for thread in self.threads:
            thread.start()
        LOGGER.info("Controller is waiting for router + required worker clients")
        try:
            while not self.stop_event.wait(1):
                pass
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()
        self.artifact_server.stop()
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=10)
        if self._results_handle is not None:
            self._results_handle.close()
            self._results_handle = None
        if self._telemetry_handle is not None:
            self._telemetry_handle.close()
            self._telemetry_handle = None

    def _open_results(self) -> None:
        path = nested(self.config, "results.output_jsonl")
        if not path:
            return
        output = Path(str(path)).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self._results_handle = output.open("a", encoding="utf-8", buffering=1)
        LOGGER.info("Writing inference results to %s", output)

    def _open_telemetry(self) -> None:
        path = nested(self.config, "metrics.output_jsonl")
        if not path:
            return
        output = Path(str(path)).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self._telemetry_handle = output.open("a", encoding="utf-8", buffering=1)
        LOGGER.info("Writing telemetry metrics to %s", output)

    def _broker_loop(self) -> None:
        while not self.stop_event.is_set():
            connection = None
            try:
                connection = connect(self.config, self.stop_event)
                channel = connection.channel()
                names = declare_topology(channel, self.config, include_results=True)
                channel.basic_qos(prefetch_count=200)
                channel.basic_consume(names.register, self._on_registration, auto_ack=False)
                channel.basic_consume(names.telemetry, self._on_telemetry, auto_ack=False)
                while not self.stop_event.is_set() and connection.is_open:
                    connection.process_data_events(time_limit=0.5)
                    while True:
                        try:
                            routing_key, message = self.outbound.get_nowait()
                        except queue.Empty:
                            break
                        publish_json(channel, self.config, "", routing_key, message)
            except Exception:
                if not self.stop_event.is_set():
                    LOGGER.exception("Controller broker loop failed; reconnecting")
                    self.stop_event.wait(2)
            finally:
                if connection is not None and connection.is_open:
                    connection.close()

    def _on_registration(self, channel, method, _properties, body: bytes) -> None:
        try:
            message = decode_message(body)
            client_id = str(message["client_id"])
            preferred_role = str(message["preferred_role"])
            capabilities = {str(item) for item in message.get("capabilities", [preferred_role])}
            if preferred_role not in {"router", "worker"}:
                raise ValueError(f"Invalid preferred_role {preferred_role!r}")
            capabilities.add(preferred_role)
            reported_generation = int(message.get("generation", 0))

            with self.lock:
                state = self.clients.get(client_id)
                if state is None:
                    state = ClientState(client_id, preferred_role, capabilities)
                    state.generation = reported_generation
                    self.clients[client_id] = state
                    LOGGER.info(
                        "Registered %s preferred=%s capabilities=%s",
                        client_id, preferred_role, sorted(capabilities),
                    )
                else:
                    state.preferred_role = preferred_role
                    state.capabilities = capabilities
                    state.generation = max(state.generation, reported_generation)
                state.last_seen = utc_timestamp()

                if self.started and state.role is None:
                    self._assign_locked(state, preferred_role, "late client joined")
                elif self.started and reported_generation < state.generation and state.role is not None:
                    self._queue_assignment_locked(state, "client requested assignment replay")
                self._maybe_start_locked()
        except Exception:
            LOGGER.exception("Invalid client registration")
        finally:
            channel.basic_ack(method.delivery_tag)

    def _on_telemetry(self, channel, method, _properties, body: bytes) -> None:
        try:
            message = decode_message(body)
            client_id = str(message["client_id"])
            if message.get("type") == "heartbeat":
                with self.lock:
                    state = self.clients.get(client_id)
                    if state is not None:
                        state.last_seen = utc_timestamp()
                        if int(message.get("generation", -1)) == state.generation:
                            state.status = str(message.get("status", state.status))
                            retry_after = float(nested(self.config, "client.assignment_retry_seconds", 15))
                            if (
                                state.status == "error"
                                and state.role
                                and utc_timestamp() - state.last_switch >= retry_after
                            ):
                                self._assign_locked(state, state.role, "retry after client error")
            elif message.get("type") == "metrics":
                accuracy = message.get("accuracy")
                accuracy_text = f" accuracy={accuracy:.4f}" if isinstance(accuracy, (int, float)) else ""
                LOGGER.info(
                    "METRIC client=%s role=%s n=%s rate=%s/s infer_p95=%sms e2e_p95=%sms%s errors=%s",
                    client_id,
                    message.get("role"),
                    message.get("processed", message.get("routed", 0)),
                    message.get("throughput_per_second"),
                    message.get("inference_ms_p95", message.get("routing_ms_p95")),
                    message.get("end_to_end_ms_p95", "-"),
                    accuracy_text,
                    message.get("errors", 0),
                )
                if self._telemetry_handle is not None:
                    record = dict(message)
                    record["received_at"] = utc_timestamp()
                    self._telemetry_handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
        except Exception:
            LOGGER.exception("Invalid telemetry message")
        finally:
            channel.basic_ack(method.delivery_tag)

    def _result_loop(self) -> None:
        while not self.stop_event.is_set():
            connection = None
            try:
                connection = connect(self.config, self.stop_event)
                channel = connection.channel()
                names = declare_topology(channel, self.config, include_results=True)
                channel.basic_qos(prefetch_count=int(nested(self.config, "results.prefetch", 1000)))
                channel.basic_consume(names.results, self._on_result, auto_ack=False)
                while not self.stop_event.is_set() and connection.is_open:
                    connection.process_data_events(time_limit=0.5)
            except Exception:
                if not self.stop_event.is_set():
                    LOGGER.exception("Result consumer failed; reconnecting")
                    self.stop_event.wait(2)
            finally:
                if connection is not None and connection.is_open:
                    connection.close()

    def _on_result(self, channel, method, _properties, body: bytes) -> None:
        try:
            if self._results_handle is not None:
                message = decode_message(body)
                self._results_handle.write(
                    json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        except Exception:
            LOGGER.exception("Could not persist inference result")
        finally:
            channel.basic_ack(method.delivery_tag)

    def _bootstrap_satisfied_locked(self) -> bool:
        now = utc_timestamp()
        stale_after = float(nested(self.config, "bootstrap.registration_stale_seconds", 30))
        required_workers = int(nested(self.config, "bootstrap.required_model_workers", 1))
        required_routers = int(nested(self.config, "bootstrap.required_router_clients", 1))

        available_workers = sum(
            state.preferred_role == "worker" and now - state.last_seen <= stale_after
            for state in self.clients.values()
        )
        available_routers = sum(
            state.preferred_role == "router" and now - state.last_seen <= stale_after
            for state in self.clients.values()
        )
        return available_workers >= required_workers and available_routers >= required_routers

    def _maybe_start_locked(self) -> None:
        if self.started or not self._bootstrap_satisfied_locked():
            return
        self.started = True
        LOGGER.info("Bootstrap requirement satisfied; releasing all registered clients")
        for state in self.clients.values():
            self._assign_locked(state, state.preferred_role, "bootstrap gate opened")

    def _assign_locked(self, state: ClientState, role: str, reason: str) -> None:
        if role not in state.capabilities:
            raise ValueError(f"Client {state.client_id} cannot run role {role}")
        state.generation += 1
        state.role = role
        state.status = "switching"
        state.last_switch = utc_timestamp()
        self._queue_assignment_locked(state, reason)

    def _queue_assignment_locked(self, state: ClientState, reason: str) -> None:
        router_bypass = state.role == "router"
        artifacts = [] if router_bypass else self.registry.manifest("worker")
        message = {
            "type": "assignment",
            "client_id": state.client_id,
            "generation": state.generation,
            "role": state.role,
            "reason": reason,
            "accuracy_enabled": bool(nested(self.config, "metrics.accuracy_enabled", False)),
            "results_enabled": bool(nested(self.config, "results.enabled", True)),
            "artifacts": artifacts,
            "router_bypass": router_bypass,
            "artifact_token": self.registry.token,
            "issued_at": utc_timestamp(),
        }
        self.outbound.put((queue_names(self.config).control(state.client_id), message))
        LOGGER.info(
            "Assigned client=%s role=%s generation=%d reason=%s",
            state.client_id, state.role, state.generation, reason,
        )
