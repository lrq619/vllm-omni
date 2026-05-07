# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare AsyncOmni Qwen-Image BF16 and FP8 latency.

This is a manual benchmark script, not a pytest test.  It creates AsyncOmni
instances with a generated single-stage diffusion YAML so that
runtime.max_batch_size can be set to 1, 2, 4, or 8.

Important batching detail:
    AsyncOmni.generate() accepts one prompt per call.  Real batching happens in
    the OmniStage worker when multiple generate() calls arrive close together
    and share equal sampling params.  The generated YAML sets max_batch_size,
    and this script submits each group concurrently to trigger that batching.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


PROMPTS = [
    "A red ceramic teapot on a wooden table, soft morning light",
    "A small robot reading a book in a quiet library",
    "A glass greenhouse full of tropical plants after rain",
    "A blue bicycle leaning against a yellow brick wall",
    "A bowl of ramen with steam rising, cinematic food photo",
    "A lighthouse on a rocky coast during a clear sunset",
    "A cozy cabin interior with a fireplace and wool blankets",
    "A futuristic city tram passing under cherry blossom trees",
]

SEED = 1101


@dataclass
class RequestRecord:
    engine: str
    batch_size: int
    group_index: int
    request_index: int
    request_id: str
    prompt: str
    seed: int
    latency_s: float
    image_path: str | None


@dataclass
class GroupRecord:
    engine: str
    batch_size: int
    group_index: int
    request_ids: list[str]
    seed: int
    latency_s: float
    throughput_img_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/tmp/models/Qwen-Image")
    parser.add_argument("--output-dir", default="benchmarks/diffusion/qwen_image_fp8_results")
    parser.add_argument("--gpu", default="0", help="GPU id used by both BF16 and FP8 instances.")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    return parser.parse_args()


def write_stage_config(path: Path, *, devices: str, max_batch_size: int) -> None:
    path.write_text(
        f"""stage_args:
  - stage_id: 0
    stage_type: diffusion
    runtime:
      process: true
      devices: "{devices}"
      max_batch_size: {max_batch_size}
    engine_args:
      model_stage: diffusion
    final_output: true
    final_output_type: image
    default_sampling_params:
      height: 512
      width: 512
      num_inference_steps: 50

runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 1
""",
        encoding="utf-8",
    )


def fp8_quantization_config(ignored_layers_csv: str) -> dict[str, Any]:
    ignored_layers = [x.strip() for x in ignored_layers_csv.split(",") if x.strip()]
    config: dict[str, Any] = {"method": "fp8"}
    if ignored_layers:
        config["ignored_layers"] = ignored_layers
    return config


def make_sampling_params(*, args: argparse.Namespace) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=0.0,
        true_cfg_scale=4.0,
        num_outputs_per_prompt=1,
        seed=SEED,
    )


def make_async_omni(
    *,
    args: argparse.Namespace,
    stage_config_path: Path,
    engine_name: str,
    batch_size: int,
) -> AsyncOmni:
    kwargs: dict[str, Any] = {
        "stage_configs_path": str(stage_config_path),
        "batch_timeout": 2.0,
        "stage_init_timeout": 300,
        "init_timeout": 600,
        "master_port": 29500 + batch_size * 10 + (1 if engine_name == "fp8" else 0),
    }
    if engine_name == "fp8":
        kwargs["quantization_config"] = fp8_quantization_config("img_mlp")
    return AsyncOmni(model=args.model, **kwargs)


def save_first_image(output: OmniRequestOutput, path: Path) -> str | None:
    images = output.images
    if not images and isinstance(output.request_output, OmniRequestOutput):
        images = output.request_output.images
    if not images:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path)
    return str(path)


async def run_one_request(
    omni: AsyncOmni,
    *,
    prompt: str,
    request_id: str,
    sampling_params: OmniDiffusionSamplingParams,
) -> OmniRequestOutput:
    last_output: OmniRequestOutput | None = None
    async for output in omni.generate(
        {"prompt": prompt, "negative_prompt": None},
        request_id=request_id,
        sampling_params_list=[sampling_params],
    ):
        last_output = output
    if last_output is None:
        raise RuntimeError(f"No output for request {request_id}")
    return last_output


async def run_group(
    omni: AsyncOmni,
    *,
    args: argparse.Namespace,
    engine_name: str,
    batch_size: int,
    group_index: int,
    prompts: list[str],
    output_dir: Path,
) -> tuple[GroupRecord, list[RequestRecord]]:
    sampling_params = make_sampling_params(args=args)
    request_ids = [
        f"{engine_name}-b{batch_size}-g{group_index}-i{i}-seed{SEED}"
        for i in range(len(prompts))
    ]

    start = time.perf_counter()
    outputs = await asyncio.gather(
        *[
            run_one_request(
                omni,
                prompt=prompt,
                request_id=request_id,
                sampling_params=sampling_params,
            )
            for prompt, request_id in zip(prompts, request_ids, strict=True)
        ]
    )
    latency_s = time.perf_counter() - start

    group = GroupRecord(
        engine=engine_name,
        batch_size=batch_size,
        group_index=group_index,
        request_ids=request_ids,
        seed=SEED,
        latency_s=latency_s,
        throughput_img_s=len(prompts) / latency_s if latency_s > 0 else 0.0,
    )

    should_save_images = batch_size == 1
    records: list[RequestRecord] = []
    for i, (prompt, request_id, output) in enumerate(zip(prompts, request_ids, outputs, strict=True)):
        image_path = None
        if should_save_images:
            image_path = save_first_image(
                output,
                output_dir / "images" / engine_name / f"{i:02d}_seed{SEED}.png",
            )
        records.append(
            RequestRecord(
                engine=engine_name,
                batch_size=batch_size,
                group_index=group_index,
                request_index=i,
                request_id=request_id,
                prompt=prompt,
                seed=SEED,
                latency_s=latency_s,
                image_path=image_path,
            )
        )

    return group, records


async def run_engine(
    omni: AsyncOmni,
    *,
    args: argparse.Namespace,
    engine_name: str,
    batch_size: int,
    output_dir: Path,
) -> tuple[list[GroupRecord], list[RequestRecord]]:
    groups: list[GroupRecord] = []
    requests: list[RequestRecord] = []
    for group_index, start in enumerate(range(0, len(PROMPTS), batch_size)):
        group_prompts = PROMPTS[start : start + batch_size]
        group, records = await run_group(
            omni,
            args=args,
            engine_name=engine_name,
            batch_size=batch_size,
            group_index=group_index,
            prompts=group_prompts,
            output_dir=output_dir,
        )
        groups.append(group)
        requests.extend(records)
        print(
            f"{engine_name} batch={batch_size} group={group_index} "
            f"latency={group.latency_s:.3f}s throughput={group.throughput_img_s:.3f} img/s",
            flush=True,
        )
    return groups, requests


def close_omni(omni: AsyncOmni | None) -> None:
    if omni is None:
        return
    omni.shutdown()


async def run_batch_size(args: argparse.Namespace, batch_size: int, output_dir: Path) -> tuple[list[GroupRecord], list[RequestRecord]]:
    config_dir = output_dir / "stage_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    bf16_config = config_dir / f"qwen_image_bf16_b{batch_size}.yaml"
    fp8_config = config_dir / f"qwen_image_fp8_b{batch_size}.yaml"
    write_stage_config(bf16_config, devices=args.gpu, max_batch_size=batch_size)
    write_stage_config(fp8_config, devices=args.gpu, max_batch_size=batch_size)

    all_groups: list[GroupRecord] = []
    all_requests: list[RequestRecord] = []

    bf16_omni: AsyncOmni | None = None
    fp8_omni: AsyncOmni | None = None
    try:
        bf16_omni = make_async_omni(args=args, stage_config_path=bf16_config, engine_name="bf16", batch_size=batch_size)
        fp8_omni = make_async_omni(args=args, stage_config_path=fp8_config, engine_name="fp8", batch_size=batch_size)

        groups, requests = await run_engine(
            bf16_omni,
            args=args,
            engine_name="bf16",
            batch_size=batch_size,
            output_dir=output_dir,
        )
        all_groups.extend(groups)
        all_requests.extend(requests)

        groups, requests = await run_engine(
            fp8_omni,
            args=args,
            engine_name="fp8",
            batch_size=batch_size,
            output_dir=output_dir,
        )
        all_groups.extend(groups)
        all_requests.extend(requests)
    finally:
        close_omni(bf16_omni)
        close_omni(fp8_omni)

    return all_groups, all_requests


def write_results(output_dir: Path, groups: list[GroupRecord], requests: list[RequestRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "groups": [asdict(x) for x in groups],
                "requests": [asdict(x) for x in requests],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (output_dir / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(groups[0]).keys()) if groups else [])
        if groups:
            writer.writeheader()
            writer.writerows(asdict(x) for x in groups)

    with (output_dir / "requests.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(requests[0]).keys()) if requests else [])
        if requests:
            writer.writeheader()
            writer.writerows(asdict(x) for x in requests)


async def main_async() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    all_groups: list[GroupRecord] = []
    all_requests: list[RequestRecord] = []
    for batch_size in batch_sizes:
        if len(PROMPTS) % batch_size != 0:
            raise ValueError(f"batch_size={batch_size} must divide {len(PROMPTS)} prompts")
        groups, requests = await run_batch_size(args, batch_size, output_dir)
        all_groups.extend(groups)
        all_requests.extend(requests)
        write_results(output_dir, all_groups, all_requests)

    print(f"Wrote results to {output_dir.resolve()}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
