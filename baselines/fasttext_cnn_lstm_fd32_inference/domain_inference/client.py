from __future__ import annotations

import gc
import logging
import os
import queue
import socket
import threading
import time
from typing import Any

import psutil

from domain_inference.artifacts import download_artifacts
from domain_inference.config import nested, require
from domain_inference.messages import decode_message, utc_timestamp
from domain_inference.metrics import MetricsBuffer
from domain_inference.models import load_model
from domain_inference.rabbit import connect, declare_control_queue, declare_topology, publish_json, queue_names
from domain_inference.runners import RouterMetricsBuffer, RouterRunner, WorkerRunner

LOGGER = logging.getLogger(__name__)


class InferenceClient:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.preferred_role = str(require(config, "client.preferred_role"))
        if self.preferred_role not in {"router", "worker"}:
            raise ValueError("client.preferred_role must be router or worker")

        default_id = f"{socket.gethostname()}-{self.preferred_role}-{os.getpid()}"
        self.client_id = str(nested(config, "client.id", default_id))
        self.capabilities = {str(role) for role in nested(config, "client.capabilities", [self.preferred_role])}
        self.capabilities.add(self.preferred_role)

        self.stop_event = threading.Event()
        self.runner_stop = threading.Event()
        self.command_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.runner_thread: threading.Thread | None = None
        self.control_thread: threading.Thread | None = None
        self.telemetry_thread: threading.Thread | None = None
        self.model = None
        self.generation = 0
        self.role = "waiting"
        self.status = "waiting"
        self.state_lock = threading.RLock()
        self.metrics: MetricsBuffer | RouterMetricsBuffer = MetricsBuffer(self.client_id)

        self.process = psutil.Process(os.getpid())
        try:
            self.cpu_count = len(self.process.cpu_affinity())
        except (AttributeError, NotImplementedError, psutil.Error):
            self.cpu_count = psutil.cpu_count(logical=True) or 1
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    def run(self) -> None:
        LOGGER.info(
            "Starting client=%s preferred=%s capabilities=%s",
            self.client_id, self.preferred_role, sorted(self.capabilities),
        )
        self.control_thread = threading.Thread(target=self._control_loop, name="client-control", daemon=True)
        self.telemetry_thread = threading.Thread(target=self._telemetry_loop, name="client-telemetry", daemon=True)
        self.control_thread.start()
        self.telemetry_thread.start()
        try:
            while not self.stop_event.is_set():
                try:
                    command = self.command_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if command.get("type") == "assignment":
                    self._apply_assignment(command)
                elif command.get("type") == "stop":
                    self.stop_event.set()
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()
        self._stop_runner()
        for thread in (self.control_thread, self.telemetry_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=10)

    def _registration(self) -> dict[str, Any]:
        with self.state_lock:
            generation = self.generation
            role = self.role
            status = self.status
        return {
            "type": "register",
            "client_id": self.client_id,
            "preferred_role": self.preferred_role,
            "capabilities": sorted(self.capabilities),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "control_queue": queue_names(self.config).control(self.client_id),
            "generation": generation,
            "role": role,
            "status": status,
            "sent_at": utc_timestamp(),
        }

    def _control_loop(self) -> None:
        registration_interval = float(nested(self.config, "client.registration_interval_seconds", 5))
        while not self.stop_event.is_set():
            connection = None
            try:
                connection = connect(self.config, self.stop_event)
                channel = connection.channel()
                names = declare_topology(channel, self.config, include_results=True)
                control_queue = declare_control_queue(channel, self.config, self.client_id)

                def on_command(callback_channel, method, _properties, body):
                    try:
                        message = decode_message(body)
                        if message.get("client_id") in (None, self.client_id):
                            self.command_queue.put(message)
                    except Exception:
                        LOGGER.exception("Invalid controller command")
                    finally:
                        callback_channel.basic_ack(method.delivery_tag)

                channel.basic_consume(control_queue, on_command, auto_ack=False)
                last_registration = 0.0
                while not self.stop_event.is_set() and connection.is_open:
                    now = time.monotonic()
                    if now - last_registration >= registration_interval:
                        publish_json(channel, self.config, "", names.register, self._registration())
                        last_registration = now
                    connection.process_data_events(time_limit=0.5)
            except Exception:
                if not self.stop_event.is_set():
                    LOGGER.exception("Control connection failed; reconnecting")
                    self.stop_event.wait(2)
            finally:
                if connection is not None and connection.is_open:
                    connection.close()

    def _telemetry_loop(self) -> None:
        interval = float(nested(self.config, "metrics.report_interval_seconds", 10))
        while not self.stop_event.is_set():
            connection = None
            try:
                connection = connect(self.config, self.stop_event)
                channel = connection.channel()
                names = declare_topology(channel, self.config, include_results=True)

                while not self.stop_event.wait(interval):
                    process_cpu_percent = self.process.cpu_percent(interval=None)
                    system_cpu_percent = psutil.cpu_percent(interval=None)
                    with self.state_lock:
                        heartbeat = {
                            "type": "heartbeat",
                            "client_id": self.client_id,
                            "role": self.role,
                            "status": self.status,
                            "generation": self.generation,
                            "sent_at": utc_timestamp(),
                        }
                        metrics = self.metrics.drain()

                    publish_json(channel, self.config, "", names.telemetry, heartbeat)

                    if metrics is not None:
                        memory = self.process.memory_info()
                        system_memory = psutil.virtual_memory()
                        metrics["sent_at"] = utc_timestamp()
                        metrics["cpu_count"] = self.cpu_count
                        metrics["process_cpu_percent"] = process_cpu_percent
                        metrics["process_cpu_cores_used"] = round(process_cpu_percent / 100.0, 4)
                        metrics["process_cpu_normalized_percent"] = round(
                            process_cpu_percent / self.cpu_count, 4
                        )
                        metrics["system_cpu_percent"] = system_cpu_percent

                        throughput = metrics.get("throughput_per_second")
                        if isinstance(throughput, (int, float)):
                            metrics["throughput_per_vcpu"] = round(throughput / self.cpu_count, 4)
                            used_cores = process_cpu_percent / 100.0
                            metrics["cpu_efficiency_jobs_per_core"] = (
                                round(throughput / used_cores, 4) if used_cores > 0 else None
                            )

                        metrics["process_rss_mb"] = memory.rss / (1024 * 1024)
                        metrics["system_ram_percent"] = system_memory.percent
                        metrics["system_ram_used_mb"] = system_memory.used / (1024 * 1024)
                        metrics["system_ram_available_mb"] = system_memory.available / (1024 * 1024)
                        publish_json(channel, self.config, "", names.telemetry, metrics)
            except Exception:
                if not self.stop_event.is_set():
                    LOGGER.exception("Telemetry connection failed; reconnecting")
                    self.stop_event.wait(2)
            finally:
                if connection is not None and connection.is_open:
                    connection.close()

    def _apply_assignment(self, command: dict[str, Any]) -> None:
        generation = int(command["generation"])
        if generation <= self.generation:
            return
        role = str(command["role"])
        router_bypass = role == "router" and bool(command.get("router_bypass", False))
        if role not in self.capabilities:
            LOGGER.error("Ignoring unsupported assignment role=%s", role)
            return
        if not self._stop_runner():
            with self.state_lock:
                self.status = "error"
            return

        with self.state_lock:
            self.status = "configuring" if router_bypass else "downloading_model"
            self.role = role
            self.generation = generation
        LOGGER.info("Applying assignment generation=%d role=%s reason=%s", generation, role, command.get("reason"))

        try:
            if router_bypass:
                self.model = None
                LOGGER.info("Single-branch router passthrough enabled; no router model is needed")
            else:
                downloaded = download_artifacts(
                    command["artifacts"],
                    str(command["artifact_token"]),
                    float(nested(self.config, "client.artifact_download_timeout_seconds", 300)),
                )
                try:
                    with self.state_lock:
                        self.status = "loading_model"
                    self.model = load_model(
                        role,
                        downloaded.paths,
                        strict=bool(nested(self.config, "client.strict_artifacts", True)),
                    )
                finally:
                    downloaded.cleanup()

            accuracy_enabled = bool(command.get("accuracy_enabled", False))
            self.metrics = RouterMetricsBuffer(self.client_id) if role == "router" else MetricsBuffer(self.client_id)
            self.metrics.set_role(role, accuracy_enabled)
            self.runner_stop = threading.Event()

            if role == "router":
                runner = RouterRunner(
                    self.config, self.client_id, self.metrics,
                    self.runner_stop, accuracy_enabled,
                )
            else:
                runner = WorkerRunner(
                    self.config, self.client_id, self.model, self.metrics,
                    self.runner_stop, accuracy_enabled, bool(command.get("results_enabled", True)),
                )

            self.runner_thread = threading.Thread(
                target=self._run_service,
                args=(runner,),
                name=f"runner-{role}",
                daemon=True,
            )
            with self.state_lock:
                self.status = "running"
            self.runner_thread.start()
        except Exception:
            with self.state_lock:
                self.status = "error"
            LOGGER.exception("Could not apply role assignment %s", role)

    def _run_service(self, runner) -> None:
        try:
            runner.run()
            with self.state_lock:
                if not self.runner_stop.is_set():
                    self.status = "completed"
        except Exception:
            with self.state_lock:
                self.status = "error"
            LOGGER.exception("Runner for role %s failed", self.role)

    def _stop_runner(self) -> bool:
        self.runner_stop.set()
        if self.runner_thread is not None and self.runner_thread is not threading.current_thread():
            self.runner_thread.join(timeout=float(nested(self.config, "client.switch_timeout_seconds", 30)))
            if self.runner_thread.is_alive():
                LOGGER.warning("Previous runner did not stop before switch timeout")
                return False
        self.runner_thread = None
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return True
