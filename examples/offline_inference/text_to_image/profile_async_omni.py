# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Profile AsyncOmni text-to-image latency across multiple image sizes.

This script uses the standard AsyncOmni startup flow (no stepwise mode, no
custom pipeline) and measures end-to-end latency for concurrent requests at
four fixed image sizes:

  - 256x256
  - 512x512
  - 1024x1024
  - 2048x2048

The benchmark performs one warmup batch before measuring each image size. The
measured request latencies are written to a CSV file, one row per run and one
column per image size. Each CSV cell stores the list of per-request e2e
latencies for that batch.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from vllm import SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

IMAGE_SIZES: list[tuple[int, int]] = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile AsyncOmni image generation latency.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image",
        help="Diffusion model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="a cup of coffee on the table",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for requests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent requests per image size.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="True CFG scale for Qwen-Image-style models.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale for guidance-distilled models.",
    )
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=None,
        help="Secondary guidance scale for models that support it.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="output/async_omni_profile.csv",
        help="CSV file to append results to.",
    )
    return parser.parse_args()


def _build_prompt(args: argparse.Namespace) -> dict[str, Any]:
    prompt: dict[str, Any] = {"prompt": args.prompt}
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt
    return prompt


def _get_stage_types(omni: AsyncOmni) -> list[str]:
    stage_types: list[str] = []
    for stage in omni.stage_list:
        if isinstance(stage, dict):
            stage_types.append(stage.get("stage_type", "diffusion"))
        else:
            stage_types.append(getattr(stage, "stage_type", "diffusion"))
    return stage_types


def _build_sampling_params_list(omni: AsyncOmni, diffusion_params: OmniDiffusionSamplingParams) -> list[Any]:
    """Build a per-stage sampling list using the engine's default stage layout."""
    stage_types = _get_stage_types(omni)
    default_params_list = list(getattr(omni, "default_sampling_params_list", []) or [])
    if len(default_params_list) != len(stage_types):
        fallback = [OmniDiffusionSamplingParams() if st == "diffusion" else SamplingParams() for st in stage_types]
        default_params_list = (default_params_list + fallback)[: len(stage_types)]

    sampling_params_list: list[Any] = []
    for idx, stage_type in enumerate(stage_types):
        if stage_type == "diffusion":
            sampling_params_list.append(diffusion_params.clone())
        else:
            sampling_params_list.append(default_params_list[idx])
    return sampling_params_list


def _format_latency_list(latencies_ms: list[float]) -> str:
    return json.dumps([round(latency, 3) for latency in latencies_ms], ensure_ascii=False)


def _output_has_image(output: Any) -> bool:
    if getattr(output, "images", None):
        return True
    request_output = getattr(output, "request_output", None)
    if request_output is None:
        return False
    return bool(getattr(request_output, "images", None))


async def _run_single_request(
    omni: AsyncOmni,
    prompt: dict[str, Any],
    sampling_params_list: list[Any],
    request_id: str,
) -> float:
    start = time.perf_counter()
    final_output = None
    async for output in omni.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=sampling_params_list,
    ):
        final_output = output

    if final_output is None:
        raise RuntimeError(f"No output returned for request_id={request_id}")
    if not _output_has_image(final_output):
        raise RuntimeError(f"No image returned for request_id={request_id}")

    return (time.perf_counter() - start) * 1000.0


async def _run_batch(
    omni: AsyncOmni,
    prompt: dict[str, Any],
    size: tuple[int, int],
    *,
    batch_size: int,
    base_seed: int,
    num_inference_steps: int,
    cfg_scale: float,
    guidance_scale: float,
    guidance_scale_2: float | None,
    record: bool,
) -> list[float]:
    height, width = size
    size_label = f"{height}x{width}"
    tasks = []

    for req_idx in range(batch_size):
        diffusion_params = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=cfg_scale,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            seed=base_seed + req_idx,
        )
        sampling_params_list = _build_sampling_params_list(omni, diffusion_params)
        request_id = f"{size_label}-{base_seed}-{req_idx}"
        tasks.append(asyncio.create_task(_run_single_request(omni, prompt, sampling_params_list, request_id)))

    latencies_ms = await asyncio.gather(*tasks)

    if record:
        for req_idx, latency_ms in enumerate(latencies_ms):
            print(f"[{size_label}] request {req_idx} e2e latency: {latency_ms:.2f} ms")
        print(f"[{size_label}] batch mean e2e latency: {mean(latencies_ms):.2f} ms")
    else:
        print(f"[{size_label}] warmup batch complete")

    return list(latencies_ms)


def _append_csv_row(csv_path: Path, row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["batch_size", "256x256", "512x512", "1024x1024", "2048x2048"]
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


async def main_async(args: argparse.Namespace) -> None:
    batch_size = max(1, args.batch_size)
    prompt = _build_prompt(args)
    csv_path = Path(args.output_csv)

    omni = AsyncOmni(model=args.model, max_batch_size=batch_size)
    try:
        print(f"Model: {args.model}")
        print(f"Batch size: {batch_size}")
        print(f"Num inference steps: {args.num_inference_steps}")
        print(f"CSV output: {csv_path}")

        measured_row: dict[str, str] = {"batch_size": str(batch_size)}

        for size_idx, size in enumerate(IMAGE_SIZES):
            height, width = size
            size_label = f"{height}x{width}"
            print(f"\n=== {size_label} ===")

            warmup_seed = args.seed + size_idx * 10_000
            await _run_batch(
                omni,
                prompt,
                size,
                batch_size=batch_size,
                base_seed=warmup_seed,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                guidance_scale=args.guidance_scale,
                guidance_scale_2=args.guidance_scale_2,
                record=False,
            )

            measure_seed = args.seed + size_idx * 10_000 + 1_000_000
            latencies_ms = await _run_batch(
                omni,
                prompt,
                size,
                batch_size=batch_size,
                base_seed=measure_seed,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                guidance_scale=args.guidance_scale,
                guidance_scale_2=args.guidance_scale_2,
                record=True,
            )
            measured_row[size_label] = _format_latency_list(latencies_ms)

        _append_csv_row(csv_path, measured_row)
        print(f"\nAppended results to {csv_path}")
    finally:
        omni.close()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
