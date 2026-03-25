#!/usr/bin/env python3
"""Analyze stepwise metric JSON and plot batch size vs. step id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def iter_points(plans: Iterable[dict]) -> Iterable[tuple[int, int]]:
    """Yield (step_id, batch_size) pairs from the metric plans."""
    for plan in plans:
        batch_size = plan.get("batch_size")
        step_indices = plan.get("step_indices") or []
        if batch_size is None:
            continue
        for step_id in step_indices:
            yield int(step_id), int(batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot batch size vs. step id from stepwise_metric.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/stepwise_worker_gpu_integration/stepwise_metric.json"),
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/stepwise_batch_size_vs_step_id.png"),
        help="Path to save the output plot.",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        metric = json.load(f)

    points = sorted(iter_points(metric.get("plans", [])), key=lambda item: item[0])
    if not points:
        raise SystemExit(f"No step/batch points found in {args.input}")

    steps = [step for step, _ in points]
    batch_sizes = [batch for _, batch in points]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, batch_sizes, marker="o", linewidth=1.8)
    plt.xlabel("Step id")
    plt.ylabel("Batch size")
    plt.title("Batch size vs. Step id")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
