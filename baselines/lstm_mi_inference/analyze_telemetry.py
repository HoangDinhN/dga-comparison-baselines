#!/usr/bin/env python3
"""Analyze telemetrys.jsonl produced by lstm_mi_inference.

This analyzer is designed for the single-branch architecture:
    Router (passthrough) -> jobs.worker -> one or more LSTM_MI workers

It intentionally removes Heavy/Light baseline detection, adaptive role switching,
routed_heavy/routed_light statistics, and hard-coded VM names/vCPU counts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


# ============================================================
# HELPERS
# ============================================================


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def average(values) -> float | None:
    clean = [float(v) for v in values if is_number(v)]
    return sum(clean) / len(clean) if clean else None


def weighted_average(rows: list[dict[str, Any]], field: str, weight_field: str = "processed") -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(field)
        weight = row.get(weight_field)
        if is_number(value) and is_number(weight) and float(weight) > 0:
            numerator += float(value) * float(weight)
            denominator += float(weight)
    return numerator / denominator if denominator > 0 else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_int(value: int | float | None) -> str:
    return "-" if value is None else f"{int(value):,}"


def get_start(row: dict[str, Any]) -> float | None:
    for field in ("window_started_at", "sent_at", "received_at"):
        value = row.get(field)
        if is_number(value):
            return float(value)
    return None


def get_end(row: dict[str, Any]) -> float | None:
    for field in ("window_ended_at", "sent_at", "received_at"):
        value = row.get(field)
        if is_number(value):
            return float(value)
    return None


def get_window_duration(row: dict[str, Any]) -> float | None:
    start = row.get("window_started_at")
    end = row.get("window_ended_at")
    if is_number(start) and is_number(end) and float(end) >= float(start):
        return float(end) - float(start)
    return None


def get_cpu_count(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        value = row.get("cpu_count")
        if is_number(value) and float(value) > 0:
            return int(value)
    return None


def normalized_cpu(row: dict[str, Any], cpu_count: int | None) -> float | None:
    # lstm_mi_inference telemetry already stores this field.
    value = row.get("process_cpu_normalized_percent")
    if is_number(value):
        return float(value)

    # Backward-compatible fallback.
    raw = row.get("process_cpu_percent")
    if is_number(raw) and cpu_count and cpu_count > 0:
        return float(raw) / cpu_count
    return None


def throughput_per_vcpu(row: dict[str, Any], cpu_count: int | None) -> float | None:
    value = row.get("throughput_per_vcpu")
    if is_number(value):
        return float(value)

    raw = row.get("throughput_per_second")
    if is_number(raw) and cpu_count and cpu_count > 0:
        return float(raw) / cpu_count
    return None


def load_telemetry(path: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_lines = 0
    empty_lines = 0
    json_errors = 0
    non_metrics_records = 0

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            total_lines += 1
            line = raw_line.strip()
            if not line:
                empty_lines += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            if record.get("type") not in (None, "metrics"):
                non_metrics_records += 1
                continue
            rows.append(record)

    return {
        "rows": rows,
        "total_lines": total_lines,
        "empty_lines": empty_lines,
        "json_errors": json_errors,
        "non_metrics_records": non_metrics_records,
    }


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, text: str = "") -> None:
        self.lines.append(str(text))

    def section(self, number: int, title: str) -> None:
        self.add()
        self.add("=" * 100)
        self.add(f"{number}. {title}")
        self.add("=" * 100)

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"


# ============================================================
# ANALYSIS
# ============================================================


def analyze_telemetry(path: str) -> str:
    data = load_telemetry(path)
    rows: list[dict[str, Any]] = data["rows"]
    if not rows:
        raise ValueError("No valid telemetry metric records were found.")

    by_client: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_client[str(row.get("client_id", "unknown"))].append(row)

    router_rows = [row for row in rows if row.get("role") == "router"]
    worker_rows = [row for row in rows if row.get("role") == "worker"]
    unknown_role_rows = [row for row in rows if row.get("role") not in {"router", "worker"}]

    router_clients = sorted({str(row.get("client_id", "unknown")) for row in router_rows})
    worker_clients = sorted({str(row.get("client_id", "unknown")) for row in worker_rows})
    all_clients = sorted(by_client)

    starts = [get_start(row) for row in rows if get_start(row) is not None]
    ends = [get_end(row) for row in rows if get_end(row) is not None]
    experiment_duration = (max(ends) - min(starts)) if starts and ends else None
    if experiment_duration is not None and experiment_duration < 0:
        experiment_duration = None

    processed_by_client: dict[str, int] = {}
    for client in worker_clients:
        processed_by_client[client] = sum(
            int(row.get("processed", 0))
            for row in by_client[client]
            if row.get("role") == "worker" and is_number(row.get("processed", 0))
        )
    total_processed = sum(processed_by_client.values())

    worker_starts = [get_start(row) for row in worker_rows if get_start(row) is not None]
    worker_ends = [get_end(row) for row in worker_rows if get_end(row) is not None]
    worker_wall_span = None
    if worker_starts and worker_ends:
        span = max(worker_ends) - min(worker_starts)
        if span > 0:
            worker_wall_span = span

    system_throughput = (
        total_processed / worker_wall_span
        if worker_wall_span is not None and worker_wall_span > 0
        else None
    )

    weighted_inference_avg = weighted_average(worker_rows, "inference_ms_avg")
    weighted_inference_p95 = weighted_average(worker_rows, "inference_ms_p95")
    weighted_e2e_avg = weighted_average(worker_rows, "end_to_end_ms_avg")
    weighted_e2e_p95 = weighted_average(worker_rows, "end_to_end_ms_p95")

    total_labeled = sum(
        int(row.get("labeled", 0)) for row in worker_rows if is_number(row.get("labeled", 0))
    )
    total_correct = sum(
        int(row.get("correct", 0)) for row in worker_rows if is_number(row.get("correct", 0))
    )
    telemetry_accuracy = safe_div(total_correct, total_labeled) if total_labeled else None

    total_dga_predictions = sum(
        int(row.get("dga_predictions", 0))
        for row in worker_rows
        if is_number(row.get("dga_predictions", 0))
    )
    total_benign_predictions = sum(
        int(row.get("benign_predictions", 0))
        for row in worker_rows
        if is_number(row.get("benign_predictions", 0))
    )

    report = Report()
    report.add("=" * 100)
    report.add("LSTM_MI TELEMETRY ANALYSIS REPORT")
    report.add("=" * 100)
    report.add(f"File                     : {path}")
    report.add("Architecture             : Single branch (Router passthrough -> jobs.worker -> LSTM_MI workers)")

    # 1
    report.section(1, "DATA INTEGRITY")
    report.add(f"Total lines              : {data['total_lines']:,}")
    report.add(f"Valid metric records     : {len(rows):,}")
    report.add(f"JSON errors              : {data['json_errors']:,}")
    report.add(f"Non-metric records       : {data['non_metrics_records']:,}")
    report.add(f"Client count             : {len(all_clients)}")
    report.add(f"Router clients           : {', '.join(router_clients) if router_clients else '-'}")
    report.add(f"Worker clients           : {', '.join(worker_clients) if worker_clients else '-'}")
    report.add(f"Unknown-role records     : {len(unknown_role_rows):,}")
    report.add(f"Experiment duration      : {fmt(experiment_duration)} s")
    report.add()
    report.add("Detected vCPU count from telemetry:")
    for client in all_clients:
        cpu_count = get_cpu_count(by_client[client])
        report.add(f"  {client:<24}: {cpu_count if cpu_count is not None else '-'}")

    # 2
    report.section(2, "WORKLOAD DISTRIBUTION")
    report.add(f"Worker count             : {len(worker_clients)}")
    report.add(f"Total processed          : {total_processed:,} domains")
    report.add(f"Predicted Benign         : {total_benign_predictions:,}")
    report.add(f"Predicted DGA            : {total_dga_predictions:,}")
    if telemetry_accuracy is not None:
        report.add(f"Telemetry accuracy       : {telemetry_accuracy * 100:.4f}% ({total_correct:,}/{total_labeled:,})")
    else:
        report.add("Telemetry accuracy       : -")

    report.add()
    report.add(f"{'Worker':<24}{'Processed':>15}{'Load share':>15}")
    report.add("-" * 54)
    for client, processed in sorted(processed_by_client.items(), key=lambda item: item[1], reverse=True):
        share = safe_div(processed, total_processed) * 100
        report.add(f"{client:<24}{processed:>15,}{share:>14.2f}%")

    # 3
    report.section(3, "THROUGHPUT")
    worker_window_throughput = average(row.get("throughput_per_second") for row in worker_rows)

    normalized_values = []
    for row in worker_rows:
        client = str(row.get("client_id", "unknown"))
        normalized_values.append(throughput_per_vcpu(row, get_cpu_count(by_client[client])))
    avg_throughput_per_vcpu = average(normalized_values)

    report.add(f"Average worker throughput: {fmt(worker_window_throughput)} domains/s")
    report.add(f"Average throughput/vCPU  : {fmt(avg_throughput_per_vcpu)} domains/s/vCPU")
    report.add(f"System throughput        : {fmt(system_throughput)} domains/s")
    report.add(f"Worker wall-clock span   : {fmt(worker_wall_span)} s")

    report.add()
    report.add(
        f"{'Worker':<22}{'vCPU':>7}{'Processed':>14}{'Active(s)':>14}"
        f"{'domains/s':>15}{'domains/s/vCPU':>18}"
    )
    report.add("-" * 90)
    for client in worker_clients:
        items = [row for row in by_client[client] if row.get("role") == "worker"]
        processed = processed_by_client[client]
        active_time = sum(
            duration
            for duration in (get_window_duration(row) for row in items)
            if duration is not None
        )
        rate = processed / active_time if active_time > 0 else None
        cpu_count = get_cpu_count(items)
        rate_per_vcpu = rate / cpu_count if rate is not None and cpu_count else None
        report.add(
            f"{client:<22}{str(cpu_count) if cpu_count is not None else '-':>7}"
            f"{processed:>14,}{active_time:>14.3f}{fmt(rate):>15}{fmt(rate_per_vcpu):>18}"
        )

    # 4
    report.section(4, "INFERENCE LATENCY")
    report.add(f"Average inference latency: {fmt(weighted_inference_avg)} ms")
    report.add(f"P95 inference latency*   : {fmt(weighted_inference_p95)} ms")
    report.add()
    report.add("* P95 here is the processed-weighted mean of each telemetry window's reported P95;")
    report.add("  it is not the exact global P95 across all domains.")
    report.add("  In lstm_mi_inference, inference_ms is batch predict_many() time divided by batch size,")
    report.add("  so it includes hostname cleaning/encoding + tensor creation + packed LSTM forward + output handling.")

    # 5
    report.section(5, "END-TO-END LATENCY")
    report.add(f"Average E2E latency      : {fmt(weighted_e2e_avg)} ms")
    report.add(f"P95 E2E latency*         : {fmt(weighted_e2e_p95)} ms")
    report.add()
    report.add("* P95 here is the processed-weighted mean of window P95 values, not an exact global P95.")
    report.add("  E2E starts from Router-created created_at immediately before job publishing and ends after")
    report.add("  worker prediction completion; it therefore reflects queueing + batching + model pipeline time.")

    # 6
    report.section(6, "CPU UTILIZATION")
    report.add(f"{'Client':<24}{'Role':<10}{'vCPU':>7}{'Normalized CPU %':>20}{'System CPU %':>16}")
    report.add("-" * 77)

    cpu_summary: dict[str, dict[str, float | None]] = {}
    for client in all_clients:
        items = by_client[client]
        cpu_count = get_cpu_count(items)
        norm_avg = average(normalized_cpu(row, cpu_count) for row in items)
        system_avg = average(row.get("system_cpu_percent") for row in items)
        roles = sorted({str(row.get("role", "unknown")) for row in items})
        role_text = "/".join(roles)
        cpu_summary[client] = {"normalized": norm_avg, "system": system_avg}
        report.add(
            f"{client:<24}{role_text:<10}{str(cpu_count) if cpu_count is not None else '-':>7}"
            f"{fmt(norm_avg, 2):>20}{fmt(system_avg, 2):>16}"
        )

    # 7
    report.section(7, "MEMORY UTILIZATION")
    report.add(f"{'Client':<24}{'Role':<10}{'Process RAM MB':>18}{'System RAM %':>16}")
    report.add("-" * 68)

    worker_rss_total = 0.0
    worker_rss_seen = 0
    memory_pressure_clients: list[str] = []
    for client in all_clients:
        items = by_client[client]
        roles = sorted({str(row.get("role", "unknown")) for row in items})
        role_text = "/".join(roles)
        rss_avg = average(row.get("process_rss_mb") for row in items)
        ram_avg = average(row.get("system_ram_percent") for row in items)
        if client in worker_clients and rss_avg is not None:
            worker_rss_total += rss_avg
            worker_rss_seen += 1
        if ram_avg is not None and ram_avg >= 85:
            memory_pressure_clients.append(client)
        report.add(f"{client:<24}{role_text:<10}{fmt(rss_avg, 2):>18}{fmt(ram_avg, 2):>16}")

    report.add()
    report.add(
        f"Total mean worker RSS    : {fmt(worker_rss_total, 2) if worker_rss_seen else '-'} MB "
        f"(sum of per-worker mean RSS)"
    )

    # 8
    report.section(8, "ERRORS & STABILITY")
    total_errors = sum(
        int(row.get("errors", 0)) for row in rows if is_number(row.get("errors", 0))
    )
    windows_with_errors = sum(
        1 for row in rows if is_number(row.get("errors")) and float(row.get("errors")) > 0
    )
    report.add(f"Total errors             : {total_errors:,}")
    report.add(f"Windows with errors      : {windows_with_errors:,}")
    report.add(f"Status                   : {'Stable - no telemetry errors detected' if total_errors == 0 else 'Errors detected'}")

    if total_errors:
        report.add()
        report.add(f"{'Client':<24}{'Errors':>12}{'Error windows':>18}")
        report.add("-" * 54)
        for client in all_clients:
            items = by_client[client]
            client_errors = sum(
                int(row.get("errors", 0)) for row in items if is_number(row.get("errors", 0))
            )
            error_windows = sum(
                1 for row in items if is_number(row.get("errors")) and float(row.get("errors")) > 0
            )
            if client_errors or error_windows:
                report.add(f"{client:<24}{client_errors:>12,}{error_windows:>18,}")

    # 9
    report.section(9, "ROUTER / SINGLE-BRANCH PIPELINE")
    total_routed = sum(
        int(row.get("routed", 0)) for row in router_rows if is_number(row.get("routed", 0))
    )
    routed_worker = sum(
        int(row.get("routed_worker", 0)) for row in router_rows if is_number(row.get("routed_worker", 0))
    )
    router_throughput = weighted_average(router_rows, "throughput_per_second", "routed")
    routing_avg = weighted_average(router_rows, "routing_ms_avg", "routed")
    routing_p95 = weighted_average(router_rows, "routing_ms_p95", "routed")
    completion_ratio = safe_div(total_processed, total_routed) if total_routed else None

    report.add(f"Router count             : {len(router_clients)}")
    report.add(f"Worker count             : {len(worker_clients)}")
    report.add(f"Total routed             : {fmt_int(total_routed)} domains")
    report.add(f"Routed to jobs.worker    : {fmt_int(routed_worker)} domains")
    report.add(f"Total worker processed   : {fmt_int(total_processed)} domains")
    report.add(
        f"Processed / routed       : {completion_ratio * 100:.2f}%"
        if completion_ratio is not None
        else "Processed / routed       : -"
    )
    report.add(f"Router throughput        : {fmt(router_throughput)} domains/s")
    report.add(f"Router decision latency  : {fmt(routing_avg)} ms avg")
    report.add(f"Router decision P95*     : {fmt(routing_p95)} ms")
    report.add()
    report.add("The router is passthrough: all jobs should be routed to jobs.worker. routing_ms measures only")
    report.add("the trivial single-branch routing decision, not RabbitMQ publish/network latency.")

    # 10
    report.section(10, "BOTTLENECK INDICATORS")
    latency_ratio = None
    if weighted_e2e_avg is not None and weighted_inference_avg is not None and weighted_inference_avg > 0:
        latency_ratio = weighted_e2e_avg / weighted_inference_avg
    report.add(f"E2E / inference ratio    : {fmt(latency_ratio, 2)}x")
    report.add()

    findings: list[str] = []
    if latency_ratio is not None:
        if latency_ratio >= 10:
            findings.append("Queue/backlog dominates: E2E latency is much larger than LSTM_MI inference time.")
        elif latency_ratio >= 3:
            findings.append("Queue/messaging overhead is significant relative to LSTM_MI inference time.")
        else:
            findings.append("E2E latency is relatively close to model-pipeline inference time.")

    saturated = sorted(
        client
        for client, values in cpu_summary.items()
        if client in worker_clients
        and (
            (values["normalized"] is not None and values["normalized"] >= 85)
            or (values["system"] is not None and values["system"] >= 85)
        )
    )
    if saturated:
        findings.append("Worker CPU saturation: " + ", ".join(saturated) + ".")
    else:
        findings.append("CPU: no worker shows sustained >=85% CPU saturation in the averaged telemetry.")

    shares = [safe_div(v, total_processed) * 100 for v in processed_by_client.values() if total_processed > 0]
    if shares:
        min_share, max_share = min(shares), max(shares)
        if max_share - min_share <= 5.0:
            findings.append(f"Workload is approximately balanced across workers ({min_share:.2f}% - {max_share:.2f}%).")
        else:
            findings.append(f"Workload distribution is uneven across workers ({min_share:.2f}% - {max_share:.2f}%).")

    if memory_pressure_clients:
        findings.append("Memory pressure (average system RAM >=85%): " + ", ".join(sorted(memory_pressure_clients)) + ".")
    else:
        findings.append("Memory: no client has average system RAM usage >=85%.")

    if total_routed:
        if total_processed == total_routed:
            findings.append("Pipeline completion: worker processed count matches router routed count.")
        elif total_processed < total_routed:
            findings.append(
                f"Pipeline completion: {total_routed - total_processed:,} routed jobs are not represented in worker processed telemetry."
            )
        else:
            findings.append(
                "Pipeline count warning: worker processed count exceeds routed count; check stale/shared queues or mixed experiment prefixes."
            )

    if total_errors == 0:
        findings.append("Stability: no telemetry errors were recorded.")
    else:
        findings.append(f"Stability: telemetry recorded {total_errors:,} errors.")

    for index, finding in enumerate(findings, start=1):
        report.add(f"[{index}] {finding}")

    report.add()
    report.add("=" * 100)
    report.add("ANALYSIS COMPLETED")
    report.add("=" * 100)
    return report.render()


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze lstm_mi_inference telemetry JSONL"
    )
    parser.add_argument("telemetry_file", help="Path to var/LSTM_MI/telemetrys.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional TXT report path. Default: <telemetry_stem>_report.txt next to input.",
    )
    args = parser.parse_args()

    input_path = Path(args.telemetry_file).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    report = analyze_telemetry(str(input_path))
    print(report, end="")

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else input_path.with_name(input_path.stem + "_report.txt")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
