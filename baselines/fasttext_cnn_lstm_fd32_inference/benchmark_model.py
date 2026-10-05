#!/usr/bin/env python3
"""Measure local model throughput without RabbitMQ or network overhead."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from domain_inference.models.fasttext_cnn_lstm import FastTextCNNLSTMFD32DGA


SAMPLE_DOMAINS = [
    "google.com",
    "mail.subdomain.example.co.uk",
    "abe70b78b996.com",
    "xj3k9qp2m7z.info",
    "cdn.assets.example.net",
    "api-v2.service.example.org",
    "qpwharrres.info",
    "www.university.edu",
]


def parse_batch_sizes(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FastText CNN-LSTM FD32 predict_many()"
    )
    parser.add_argument(
        "--bundle",
        default="artifact/best_bundle.pt",
        help="Path to the exported deployment bundle",
    )
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=[1, 8, 32, 64])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="PyTorch intra-op threads; omit to retain the environment default",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be >= 0 and repeats must be > 0")
    if args.threads is not None:
        if args.threads <= 0:
            parser.error("threads must be > 0")
        torch.set_num_threads(args.threads)

    bundle = Path(args.bundle).expanduser().resolve()
    adapter = FastTextCNNLSTMFD32DGA(str(bundle))
    parameter_count = sum(parameter.numel() for parameter in adapter.model.parameters())

    print("bundle:", bundle)
    print("device:", adapter.device)
    print("torch:", torch.__version__)
    print("threads:", torch.get_num_threads())
    print("parameters:", f"{parameter_count:,}")
    print("threshold:", adapter.threshold)
    print()
    print(f"{'batch':>8} {'median ms/batch':>18} {'ms/domain':>14} {'domains/s':>14}")
    print("-" * 58)

    for batch_size in args.batch_sizes:
        domains = [SAMPLE_DOMAINS[index % len(SAMPLE_DOMAINS)] for index in range(batch_size)]
        for _ in range(args.warmup):
            adapter.predict_many(domains)
        synchronize(adapter.device)

        samples = []
        for _ in range(args.repeats):
            synchronize(adapter.device)
            started = time.perf_counter()
            adapter.predict_many(domains)
            synchronize(adapter.device)
            samples.append((time.perf_counter() - started) * 1000.0)

        median_ms = statistics.median(samples)
        per_domain_ms = median_ms / batch_size
        throughput = 1000.0 * batch_size / median_ms
        print(
            f"{batch_size:>8d} {median_ms:>18.3f} "
            f"{per_domain_ms:>14.4f} {throughput:>14.2f}"
        )


if __name__ == "__main__":
    main()
