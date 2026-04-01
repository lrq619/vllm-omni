# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Profile Omni text-to-image latency across multiple image sizes.

This script follows the standard Omni flow from the docs and measures how
prompt length affects end-to-end latency. The prompt is repeated
``prompt_length_scale`` times to approximate a longer prompt, and the CSV row
starts with the estimated prompt token length computed from whitespace.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

IMAGE_SIZES: list[tuple[int, int]] = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Omni image generation latency.")
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
        help="Seed for sampling parameters.",
    )
    parser.add_argument(
        "--prompt-length-scale",
        type=int,
        default=1,
        help="Repeat the prompt this many times to approximate a longer prompt.",
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
        default="output/omni_profile.csv",
        help="CSV file to append results to.",
    )
    return parser.parse_args()


def _build_prompt(args: argparse.Namespace) -> dict[str, Any]:
    prompt_scale = max(1, args.prompt_length_scale)
    scaled_prompt = " ".join([args.prompt] * prompt_scale)
    prompt: dict[str, Any] = {"prompt": scaled_prompt}
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt
    return prompt


def _estimate_prompt_token_length(prompt_text: str) -> int:
    return len(prompt_text.split())


def _get_stage_types(omni: Omni) -> list[str]:
    stage_types: list[str] = []
    for stage in omni.stage_list:
        if isinstance(stage, dict):
            stage_types.append(stage.get("stage_type", "diffusion"))
        else:
            stage_types.append(getattr(stage, "stage_type", "diffusion"))
    return stage_types


def _build_sampling_params(args: argparse.Namespace) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.cfg_scale,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        seed=args.seed,
    )


def _build_sampling_params_list(omni: Omni, diffusion_params: OmniDiffusionSamplingParams) -> list[Any]:
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


def _append_csv_row(csv_path: Path, row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["prompt_token_length", "256x256", "512x512", "1024x1024", "2048x2048"]
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _run_batch(
    omni: Omni,
    prompt: dict[str, Any],
    *,
    sampling_params_list: list[Any],
    record: bool,
) -> float:
    batch_start = time.perf_counter()

    output_count = 0
    final_latency_ms = 0.0
    for output in omni._run_generation(  # noqa: SLF001 - keep the instance alive across multiple runs
        prompt,
        sampling_params_list,
        use_tqdm=False,
    ):
        output_count += 1
        final_latency_ms = (time.perf_counter() - batch_start) * 1000.0
        if record:
            print(f"[request_id={output.request_id}] e2e latency: {final_latency_ms:.2f} ms")

    if output_count != 1:
        raise RuntimeError(f"Expected 1 output, got {output_count}")

    if not record:
        print("warmup complete")

    return final_latency_ms


def main() -> None:
    args = parse_args()
    csv_path = Path(args.output_csv)

    omni = Omni(model=args.model)
    try:
        print(f"Model: {args.model}")
        print(f"Prompt length scale: {max(1, args.prompt_length_scale)}")
        print(f"Num inference steps: {args.num_inference_steps}")
        print(f"CSV output: {csv_path}")

        prompt = _build_prompt(args)
        prompt_token_length = _estimate_prompt_token_length(prompt["prompt"])
        measured_row: dict[str, str] = {"prompt_token_length": str(prompt_token_length)}

        for size_idx, size in enumerate(IMAGE_SIZES):
            height, width = size
            size_label = f"{height}x{width}"
            print(f"\n=== {size_label} ===")

            warmup_sampling_params = _build_sampling_params(args)
            warmup_sampling_params.height = height
            warmup_sampling_params.width = width
            warmup_sampling_params_list = _build_sampling_params_list(omni, warmup_sampling_params)
            _run_batch(omni, prompt, sampling_params_list=warmup_sampling_params_list, record=False)

            measured_sampling_params = _build_sampling_params(args)
            measured_sampling_params.height = height
            measured_sampling_params.width = width
            # Keep the row deterministic across sizes while still using a fixed seed.
            measured_sampling_params.seed = args.seed + size_idx * 10_000
            measured_sampling_params_list = _build_sampling_params_list(omni, measured_sampling_params)

            latency_ms = _run_batch(
                omni,
                prompt,
                sampling_params_list=measured_sampling_params_list,
                record=True,
            )
            measured_row[size_label] = f"{latency_ms:.3f}"

        _append_csv_row(csv_path, measured_row)
        print(f"\nAppended results to {csv_path}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
