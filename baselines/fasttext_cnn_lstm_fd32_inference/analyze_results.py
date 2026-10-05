#!/usr/bin/env python3
"""Analyze results.jsonl produced by the FD32 FastText CNN-LSTM project.

Architecture assumed by this analyzer:
    CSV -> Router (passthrough) -> jobs.worker -> FastText_CNN_LSTM worker
        -> results -> Controller

The project has one model branch only. Therefore this analyzer intentionally does
NOT contain Heavy/Light routing statistics or adaptive-role analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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


def average(values: list[float]) -> float | None:
    clean = [float(v) for v in values if is_number(v)]
    return sum(clean) / len(clean) if clean else None


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile for per-result latency values."""
    clean = sorted(float(v) for v in values if is_number(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]

    k = (len(clean) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - k) + clean[hi] * (k - lo)


def fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def confusion_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    evaluated = tp + tn + fp + fn
    accuracy = safe_div(tp + tn, evaluated)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
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


def analyze_results(path: str) -> str:
    total_lines = 0
    empty_lines = 0
    valid_json = 0
    json_errors = 0
    inference_results = 0
    unexpected_types = 0

    missing_truth = 0
    invalid_truth_or_prediction = 0
    missing_score = 0
    invalid_score = 0

    job_ids: set[str] = set()
    source_ids: set[str] = set()
    domains_seen: set[str] = set()
    duplicate_job_id = 0
    duplicate_source_id = 0
    duplicate_domains = 0

    truth_counter: Counter[int] = Counter()
    prediction_counter: Counter[int] = Counter()
    label_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    route_counter: Counter[str] = Counter()
    worker_counter: Counter[str] = Counter()

    label_mismatches = 0

    tp = tn = fp = fn = 0
    e2e_latencies_ms: list[float] = []
    scores: list[float] = []
    scores_truth_benign: list[float] = []
    scores_truth_dga: list[float] = []

    family_stats = defaultdict(lambda: {"samples": 0, "tp": 0, "fn": 0})
    worker_stats = defaultdict(
        lambda: {
            "samples": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "latencies": [],
            "scores": [],
        }
    )

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

            valid_json += 1
            if record.get("type") not in (None, "inference_result"):
                unexpected_types += 1
                continue
            inference_results += 1

            job_id = record.get("job_id")
            if job_id is not None:
                key = str(job_id)
                if key in job_ids:
                    duplicate_job_id += 1
                else:
                    job_ids.add(key)

            source_id = record.get("source_id")
            if source_id is not None:
                key = str(source_id)
                if key in source_ids:
                    duplicate_source_id += 1
                else:
                    source_ids.add(key)

            domain = record.get("domain")
            if domain is not None:
                key = str(domain)
                if key in domains_seen:
                    duplicate_domains += 1
                else:
                    domains_seen.add(key)

            route_counter[str(record.get("route", "unknown"))] += 1
            worker = str(record.get("worker_client_id", "unknown"))
            worker_counter[worker] += 1
            model_counter[str(record.get("model", "unknown"))] += 1

            prediction = record.get("prediction")
            label = record.get("label")
            if prediction in (0, 1):
                prediction = int(prediction)
                prediction_counter[prediction] += 1
                expected_label = "DGA" if prediction == 1 else "Benign"
                if label is not None and str(label) != expected_label:
                    label_mismatches += 1
            if label is not None:
                label_counter[str(label)] += 1

            score = record.get("score")
            if score is None:
                missing_score += 1
            elif is_number(score):
                score_f = float(score)
                scores.append(score_f)
                worker_stats[worker]["scores"].append(score_f)
            else:
                invalid_score += 1

            created_at = record.get("created_at")
            completed_at = record.get("completed_at")
            latency_ms = None
            if (
                is_number(created_at)
                and is_number(completed_at)
                and float(completed_at) >= float(created_at)
            ):
                latency_ms = (float(completed_at) - float(created_at)) * 1000.0
                e2e_latencies_ms.append(latency_ms)
                worker_stats[worker]["latencies"].append(latency_ms)

            truth = record.get("truth")
            if truth is None:
                missing_truth += 1
                continue

            if truth not in (0, 1) or prediction not in (0, 1):
                invalid_truth_or_prediction += 1
                continue

            truth = int(truth)
            truth_counter[truth] += 1
            worker_stats[worker]["samples"] += 1

            if is_number(score):
                if truth == 0:
                    scores_truth_benign.append(float(score))
                else:
                    scores_truth_dga.append(float(score))

            if truth == 1 and prediction == 1:
                tp += 1
                worker_stats[worker]["tp"] += 1
            elif truth == 0 and prediction == 0:
                tn += 1
                worker_stats[worker]["tn"] += 1
            elif truth == 0 and prediction == 1:
                fp += 1
                worker_stats[worker]["fp"] += 1
            elif truth == 1 and prediction == 0:
                fn += 1
                worker_stats[worker]["fn"] += 1

            if truth == 1:
                family = record.get("subclass")
                family_name = str(family) if family not in (None, "") else "unknown"
                family_stats[family_name]["samples"] += 1
                if prediction == 1:
                    family_stats[family_name]["tp"] += 1
                else:
                    family_stats[family_name]["fn"] += 1

    evaluated = tp + tn + fp + fn
    overall = confusion_metrics(tp, tn, fp, fn)

    report = Report()
    report.add("=" * 100)
    report.add("FastText_CNN_LSTM_FD32 RESULTS ANALYSIS REPORT")
    report.add("=" * 100)
    report.add(f"File                     : {path}")
    report.add("Architecture             : Single branch (Router -> jobs.worker -> FastText_CNN_LSTM_FD32 workers)")

    # 1
    report.section(1, "DATA INTEGRITY")
    report.add(f"Total lines              : {total_lines:,}")
    report.add(f"Empty lines              : {empty_lines:,}")
    report.add(f"Valid JSON records       : {valid_json:,}")
    report.add(f"Inference results        : {inference_results:,}")
    report.add(f"JSON errors              : {json_errors:,}")
    report.add(f"Unexpected record types  : {unexpected_types:,}")
    report.add(f"Unique job_id            : {len(job_ids):,}")
    report.add(f"Duplicate job_id         : {duplicate_job_id:,}")
    report.add(f"Unique source_id         : {len(source_ids):,}")
    report.add(f"Duplicate source_id      : {duplicate_source_id:,}")
    report.add(f"Unique domains           : {len(domains_seen):,}")
    report.add(f"Duplicate domains        : {duplicate_domains:,}")
    report.add(f"Missing truth            : {missing_truth:,}")
    report.add(f"Invalid truth/prediction : {invalid_truth_or_prediction:,}")
    report.add(f"Missing/invalid score    : {missing_score:,} / {invalid_score:,}")
    report.add(f"Label/prediction mismatch: {label_mismatches:,}")

    # 2
    report.section(2, "DATASET AND PREDICTION DISTRIBUTION")
    benign = truth_counter[0]
    dga = truth_counter[1]
    pred_benign = prediction_counter[0]
    pred_dga = prediction_counter[1]
    report.add(f"Evaluated records        : {evaluated:,}")
    report.add(f"Truth Benign             : {benign:,} ({safe_div(benign, evaluated) * 100:.2f}%)")
    report.add(f"Truth DGA                : {dga:,} ({safe_div(dga, evaluated) * 100:.2f}%)")
    report.add(f"Predicted Benign         : {pred_benign:,} ({safe_div(pred_benign, pred_benign + pred_dga) * 100:.2f}%)")
    report.add(f"Predicted DGA            : {pred_dga:,} ({safe_div(pred_dga, pred_benign + pred_dga) * 100:.2f}%)")

    # 3
    report.section(3, "CONFUSION MATRIX")
    report.add("                    Predicted")
    report.add("                Benign       DGA")
    report.add(f"Truth Benign    {tn:<12} {fp}")
    report.add(f"Truth DGA       {fn:<12} {tp}")
    report.add()
    report.add(f"TN                       : {tn:,}")
    report.add(f"FP                       : {fp:,}")
    report.add(f"FN                       : {fn:,}")
    report.add(f"TP                       : {tp:,}")

    # 4
    report.section(4, "CLASSIFICATION METRICS")
    report.add(f"Accuracy                 : {pct(overall['accuracy'])}")
    report.add(f"Precision (DGA)          : {pct(overall['precision'])}")
    report.add(f"Recall (DGA)             : {pct(overall['recall'])}")
    report.add(f"F1-score (DGA)           : {pct(overall['f1'])}")
    report.add(f"Specificity              : {pct(overall['specificity'])}")
    report.add(f"False positive rate      : {pct(safe_div(fp, fp + tn))}")
    report.add(f"False negative rate      : {pct(safe_div(fn, fn + tp))}")

    # 5
    report.section(5, "MODEL SCORE SUMMARY")
    report.add(f"Score samples            : {len(scores):,}")
    report.add(f"Overall mean score       : {fmt(average(scores), 6)}")
    report.add(f"Mean score | Truth Benign: {fmt(average(scores_truth_benign), 6)}")
    report.add(f"Mean score | Truth DGA   : {fmt(average(scores_truth_dga), 6)}")
    report.add(f"Minimum score            : {fmt(min(scores) if scores else None, 6)}")
    report.add(f"Maximum score            : {fmt(max(scores) if scores else None, 6)}")
    report.add()
    report.add("Note: score is the FastText_CNN_LSTM_FD32 sigmoid output. The decision threshold is stored in the model bundle,")
    report.add("not in each result record, so this analyzer does not assume or reconstruct a threshold.")

    # 6
    report.section(6, "WORKER DISTRIBUTION AND PERFORMANCE")
    report.add(
        f"{'Worker':<22}{'Results':>12}{'Share':>10}{'Accuracy':>13}"
        f"{'FP':>9}{'FN':>9}{'Avg E2E ms':>15}"
    )
    report.add("-" * 90)

    total_worker_results = sum(worker_counter.values())
    for worker, count in worker_counter.most_common():
        ws = worker_stats[worker]
        wm = confusion_metrics(ws["tp"], ws["tn"], ws["fp"], ws["fn"])
        avg_e2e = average(ws["latencies"])
        report.add(
            f"{worker:<22}{count:>12,}{safe_div(count, total_worker_results) * 100:>9.2f}%"
            f"{wm['accuracy'] * 100:>12.4f}%{ws['fp']:>9,}{ws['fn']:>9,}{fmt(avg_e2e):>15}"
        )

    # 7
    report.section(7, "SINGLE-BRANCH PIPELINE CONSISTENCY")
    report.add("Route distribution:")
    for route, count in route_counter.most_common():
        report.add(f"  {route:<22}: {count:>12,} ({safe_div(count, inference_results) * 100:.2f}%)")

    report.add()
    report.add("Model distribution:")
    for model, count in model_counter.most_common():
        report.add(f"  {model:<22}: {count:>12,} ({safe_div(count, inference_results) * 100:.2f}%)")

    unexpected_route = sum(count for route, count in route_counter.items() if route != "worker")
    report.add()
    report.add(f"Unexpected non-worker route: {unexpected_route:,}")
    if unexpected_route == 0:
        report.add("Pipeline status           : Consistent with single-branch architecture")
    else:
        report.add("Pipeline status           : WARNING - non-worker route values were found")

    # 8
    report.section(8, "END-TO-END LATENCY")
    if e2e_latencies_ms:
        report.add(f"Samples                  : {len(e2e_latencies_ms):,}")
        report.add(f"Average                  : {fmt(average(e2e_latencies_ms))} ms")
        report.add(f"P50                      : {fmt(percentile(e2e_latencies_ms, 50))} ms")
        report.add(f"P95                      : {fmt(percentile(e2e_latencies_ms, 95))} ms")
        report.add(f"P99                      : {fmt(percentile(e2e_latencies_ms, 99))} ms")
        report.add(f"Maximum                  : {fmt(max(e2e_latencies_ms))} ms")
        report.add()
        report.add("E2E is computed from result.created_at to result.completed_at. In this project created_at is set")
        report.add("by the Router immediately before publishing jobs, so E2E includes queue wait + worker batching +")
        report.add("CNN-LSTM predict_many() processing, but not CSV loading or the router's earlier source work.")
    else:
        report.add("No valid created_at/completed_at latency samples were found.")

    # 9
    report.section(9, "DGA FAMILY ANALYSIS")
    report.add(f"{'Family':<30}{'Samples':>12}{'TP':>12}{'FN':>12}{'Recall':>15}")
    report.add("-" * 81)
    for family, fs in sorted(family_stats.items(), key=lambda item: item[1]["samples"], reverse=True):
        recall = safe_div(fs["tp"], fs["tp"] + fs["fn"])
        report.add(
            f"{family:<30}{fs['samples']:>12,}{fs['tp']:>12,}{fs['fn']:>12,}{recall * 100:>14.2f}%"
        )

    # 10
    report.section(10, "WORST DGA FAMILIES BY RECALL")
    report.add(f"{'Family':<30}{'Samples':>12}{'FN':>12}{'Recall':>15}")
    report.add("-" * 69)
    worst = sorted(
        family_stats.items(),
        key=lambda item: (
            safe_div(item[1]["tp"], item[1]["tp"] + item[1]["fn"]),
            -item[1]["samples"],
        ),
    )
    for family, fs in worst[:10]:
        recall = safe_div(fs["tp"], fs["tp"] + fs["fn"])
        report.add(f"{family:<30}{fs['samples']:>12,}{fs['fn']:>12,}{recall * 100:>14.2f}%")

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
        description="Analyze FastText CNN-LSTM FD32 results JSONL"
    )
    parser.add_argument("results_file", help="Path to var/test_genAI/results_1M.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional TXT report path. Default: <results_stem>_report.txt next to input.",
    )
    args = parser.parse_args()

    input_path = Path(args.results_file).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    report = analyze_results(str(input_path))
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
