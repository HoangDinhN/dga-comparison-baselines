from __future__ import annotations

import argparse
import logging
import signal
from typing import Any

from domain_inference.client import InferenceClient
from domain_inference.config import load_config
from domain_inference.controller import Controller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-branch LSTM_MI distributed DGA inference")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("controller", "client"):
        item = commands.add_parser(command)
        item.add_argument("--config", required=True, help="YAML configuration file")
    return parser


def install_signal_handlers(service: Any) -> None:
    def stop(_signum, _frame):
        service.stop_event.set()
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s - %(message)s",
    )
    config = load_config(args.config)
    service = Controller(config) if args.command == "controller" else InferenceClient(config)
    install_signal_handlers(service)
    service.run()


if __name__ == "__main__":
    main()
