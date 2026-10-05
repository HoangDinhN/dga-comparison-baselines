from __future__ import annotations

import logging
import random
import time
from typing import Any

from domain_inference.config import nested, require
from domain_inference.messages import QueueNames, encode_message

LOGGER = logging.getLogger(__name__)


def pika_module():
    try:
        import pika
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'pika'; install requirements.txt") from exc
    return pika


def connection_parameters(config: dict[str, Any]):
    pika = pika_module()
    params = pika.URLParameters(str(require(config, "rabbitmq.url")))
    params.heartbeat = int(nested(config, "rabbitmq.heartbeat_seconds", 30))
    params.blocked_connection_timeout = int(nested(config, "rabbitmq.blocked_timeout_seconds", 60))
    params.connection_attempts = int(nested(config, "rabbitmq.connection_attempts", 5))
    params.retry_delay = float(nested(config, "rabbitmq.retry_delay_seconds", 2))
    params.socket_timeout = float(nested(config, "rabbitmq.socket_timeout_seconds", 10))
    return params


def connect(config: dict[str, Any], stop_event=None):
    pika = pika_module()
    delay = 1.0
    while stop_event is None or not stop_event.is_set():
        try:
            return pika.BlockingConnection(connection_parameters(config))
        except Exception as exc:
            LOGGER.warning("RabbitMQ connection failed: %s; retrying in %.1fs", exc, delay)
            if stop_event is not None and stop_event.wait(delay):
                break
            time.sleep(0 if stop_event is not None else delay)
            delay = min(30.0, delay * 1.7 + random.random())
    raise RuntimeError("RabbitMQ connection stopped before it was established")


def queue_names(config: dict[str, Any]) -> QueueNames:
    return QueueNames(str(nested(config, "rabbitmq.queue_prefix", "dga.deepdga")))


def declare_topology(channel, config: dict[str, Any], include_results: bool = True) -> QueueNames:
    names = queue_names(config)
    durable = bool(nested(config, "rabbitmq.durable", True))
    queue_args = {"x-queue-type": str(nested(config, "rabbitmq.queue_type", "classic"))}

    channel.exchange_declare(exchange=names.jobs_exchange, exchange_type="direct", durable=durable)
    channel.queue_declare(queue=names.worker_jobs, durable=durable, arguments=queue_args)
    channel.queue_bind(queue=names.worker_jobs, exchange=names.jobs_exchange, routing_key="worker")
    channel.queue_declare(queue=names.register, durable=durable, arguments=queue_args)
    channel.queue_declare(queue=names.telemetry, durable=durable, arguments=queue_args)
    if include_results:
        channel.queue_declare(queue=names.results, durable=durable, arguments=queue_args)
    return names


def declare_control_queue(channel, config: dict[str, Any], client_id: str) -> str:
    names = queue_names(config)
    durable = bool(nested(config, "rabbitmq.durable", True))
    arguments = {
        "x-expires": int(nested(config, "client.control_queue_expiry_ms", 86_400_000)),
        "x-max-length": int(nested(config, "client.control_queue_max_messages", 20)),
        "x-overflow": "drop-head",
    }
    channel.queue_declare(queue=names.control(client_id), durable=durable, arguments=arguments)
    return names.control(client_id)


def properties(config: dict[str, Any], *, correlation_id: str | None = None):
    pika = pika_module()
    delivery_mode = 2 if bool(nested(config, "rabbitmq.persistent_messages", False)) else 1
    return pika.BasicProperties(
        content_type="application/json",
        delivery_mode=delivery_mode,
        correlation_id=correlation_id,
        timestamp=int(time.time()),
    )


def publish_json(channel, config: dict[str, Any], exchange: str, routing_key: str, message: dict[str, Any]) -> None:
    channel.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=encode_message(message),
        properties=properties(config, correlation_id=str(message.get("job_id", "")) or None),
    )
