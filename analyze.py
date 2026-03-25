#!/usr/bin/env python3
"""Analyze stepwise metric JSON and plot batch size vs. loop-end timestamp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def iter_points(plans: Iterable[dict]) -> Iterable[tuple[float, int]]:
    """Yield (timestamp_s, batch_size) pairs from the metric plans."""
    for plan in plans:
        batch_size = plan.get("batch_size")
        ts_ms = plan.get("loop_end_timestamp_ms", plan.get("timestamp_ms"))
        if batch_size is None or ts_ms is None:
            continue
        yield float(ts_ms) / 1000.0, int(batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot batch size vs. loop-end timestamp (s) from stepwise_metric.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/stepwise_worker_gpu_integration/32_requests/stepwise_metric.json"),
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/stepwise_batch_size_vs_timestamp_s.png"),
        help="Path to save the output plot.",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        metric = json.load(f)

    points = sorted(iter_points(metric.get("plans", [])), key=lambda item: item[0])
    if not points:
        raise SystemExit(f"No timestamp/batch points found in {args.input}")

    timestamps_s = [timestamp_s for timestamp_s, _ in points]
    batch_sizes = [batch for _, batch in points]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(timestamps_s, batch_sizes, marker="o", linewidth=1.8)
    plt.xlabel("Loop-end timestamp (s)")
    plt.ylabel("Batch size")
    plt.title("Batch size vs. Loop-end timestamp (s)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
