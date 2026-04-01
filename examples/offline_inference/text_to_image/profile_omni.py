from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from vllm_omni.entrypoints.omni import Omni


DEFAULT_MODEL = "/tmp/models/Qwen/Qwen-Image"
DEFAULT_PROMPT = "a cup of coffee on a table"
DEFAULT_NEGATIVE_PROMPT = "blurry low quality distorted deformed watermark"
DEFAULT_OUTPUT_CSV = "output/omni_prompt_profile.csv"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_SIZES = (
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Omni text-to-image e2e latency by prompt length."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path.")
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="CSV file to append results to.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Base prompt text to repeat for prompt length profiling.",
    )
    parser.add_argument(
        "--prompt-length-scale",
        type=int,
        default=1,
        help="Repeat the prompt text this many times before profiling.",
    )
    parser.add_argument(
        "--enable-negative",
        action="store_true",
        help="Enable negative prompt profiling and write results to a separate CSV.",
    )
    return parser.parse_args()


def _repeat_text(text: str, scale: int) -> str:
    repeat_count = max(1, scale)
    return " ".join([text] * repeat_count)


def _estimate_prompt_token_length(prompt_text: str) -> int:
    return len(prompt_text.split())


def _derive_output_csv_path(output_csv: str, enable_negative: bool) -> Path:
    output_path = Path(output_csv)
    if not enable_negative:
        return output_path
    suffix = output_path.suffix or ".csv"
    return output_path.with_name(f"{output_path.stem}_negative{suffix}")


def _build_request(
    prompt_text: str,
    negative_prompt_text: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "prompt": prompt_text,
        "width": width,
        "height": height,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
    }
    if negative_prompt_text:
        request["negative_prompt"] = negative_prompt_text
    return request


def _run_request(omni: Omni, request: dict[str, Any]) -> float:
    start = time.perf_counter()
    outputs = list(omni.generate([request]))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if len(outputs) != 1:
        raise RuntimeError(f"Expected exactly one output, got {len(outputs)}")
    return elapsed_ms


def main() -> None:
    args = parse_args()
    if args.prompt_length_scale < 1:
        raise ValueError("--prompt-length-scale must be a positive integer")

    omni = Omni(model=args.model)

    prompt_text = _repeat_text(args.prompt, args.prompt_length_scale)
    prompt_token_length = _estimate_prompt_token_length(prompt_text)

    negative_prompt_text = ""
    negative_prompt_token_length = 0
    if args.enable_negative:
        negative_prompt_text = _repeat_text(
            DEFAULT_NEGATIVE_PROMPT, args.prompt_length_scale
        )
        negative_prompt_token_length = _estimate_prompt_token_length(
            negative_prompt_text
        )

    output_csv = _derive_output_csv_path(args.output_csv, args.enable_negative)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Output CSV: {output_csv}")
    print(f"Prompt length scale: {args.prompt_length_scale}")
    print(f"Prompt token length: {prompt_token_length}")
    print(f"Negative prompt enabled: {args.enable_negative}")
    if args.enable_negative:
        print(f"Negative prompt token length: {negative_prompt_token_length}")

    fieldnames = [
        "prompt_token_length",
        "negative_prompt_token_length",
        "256x256",
        "512x512",
        "1024x1024",
        "2048x2048",
    ]

    csv_exists = output_csv.exists()
    with output_csv.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        # Prewarm the instance once per size before recording latency.
        for width, height in IMAGE_SIZES:
            warmup_request = _build_request(
                prompt_text, negative_prompt_text, width, height
            )
            _run_request(omni, warmup_request)

        row: dict[str, Any] = {
            "prompt_token_length": prompt_token_length,
            "negative_prompt_token_length": negative_prompt_token_length,
        }

        for width, height in IMAGE_SIZES:
            request = _build_request(prompt_text, negative_prompt_text, width, height)
            latency_ms = _run_request(omni, request)
            size_key = f"{width}x{height}"
            row[size_key] = f"{latency_ms:.3f}"
            print(f"{size_key}: {latency_ms:.3f} ms")

        writer.writerow(row)


if __name__ == "__main__":
    main()
