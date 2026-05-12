# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AsyncOmni SP switching smoke and stress tests.

This file contains:
1. A minimal smoke test that exercises SP=2 -> SP=1 -> SP=2 with one image
   per phase.
2. A lightweight stress test that exercises the same SP transitions while
   generating 8 images per phase at 1280x1280, writing per-phase outputs and
   latency summaries under ./output.
"""

import asyncio
import gc
import json
import os
import sys
import socket
import time
from pathlib import Path
from statistics import mean

import pytest
import torch
from PIL import Image
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.utils import hardware_test
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.entrypoints.async_omni import AsyncOmni

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_PATH = "/tmp/models/Qwen/Qwen-Image"
PROMPT = "a photo of a cat sitting on a laptop keyboard"
OUTPUT_DIR = Path("./output")
BENCH_PROMPTS = [
    "a red apple on a wooden table",
    "a blue bicycle parked by a wall",
    "a small robot holding a flower",
    "a mountain lake at sunrise",
    "a paper airplane flying indoors",
    "a yellow bird on a branch",
    "a cup of coffee beside a notebook",
    "a cozy chair near a window",
]


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _extract_single_image(output: OmniRequestOutput) -> Image.Image:
    if output.final_output_type != "image":
        raise ValueError(f"Expected final output type 'image', got {output.final_output_type!r}")
    if not output.images:
        raise ValueError("No images returned by AsyncOmni.generate()")

    image = output.images[0]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")
    return image


def _make_sampling_params(
    *,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 20,
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
        num_outputs_per_prompt=1,
    )


def _slugify_prompt(prompt: str) -> str:
    chars: list[str] = []
    last_dash = False
    for ch in prompt.lower():
        if ch.isalnum():
            chars.append(ch)
            last_dash = False
        elif not last_dash:
            chars.append("-")
            last_dash = True
    slug = "".join(chars).strip("-")
    return slug or "prompt"


async def _run_generate_and_save(
    engine: AsyncOmni,
    *,
    prompt: str,
    request_id: str,
    output_path: Path,
    label: str,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 20,
) -> float:
    print(f"[{label}] starting generate request_id={request_id}")
    start = time.perf_counter()
    last_output: OmniRequestOutput | None = None
    omni_output: OmniRequestOutput | None = None
    async for omni_output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=[_make_sampling_params(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )],
    ):
        last_output = omni_output
    elapsed_s = time.perf_counter() - start

    if last_output is None:
        raise ValueError("AsyncOmni.generate() produced no outputs")

    print(f"[{label}] generate finished request_id={request_id}, extracting image")
    image = _extract_single_image(last_output)
    image.save(output_path)
    print(f"[{label}] saved image to {output_path}")

    # Release the output object promptly so the next request can reuse IPC/shm
    # resources before we start the next generation phase.
    del omni_output
    del last_output
    del image
    gc.collect()
    current_omni_platform.empty_cache()
    current_omni_platform.synchronize()
    print(f"[{label}] released output references and completed cleanup")
    return elapsed_s


async def _run_generate_batch_and_collect(
    engine: AsyncOmni,
    *,
    round_name: str,
    prompts: list[str],
    output_root: Path,
    height: int,
    width: int,
    num_inference_steps: int = 20,
) -> dict[str, object]:
    round_dir = output_root / round_name
    round_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    latencies_s: list[float] = []

    print(f"[{round_name}] begin batch generation num_prompts={len(prompts)} output_dir={round_dir}")
    for idx, prompt in enumerate(prompts):
        request_id = f"{round_name}-request-{idx}"
        output_path = round_dir / f"{idx:02d}-{_slugify_prompt(prompt)[:64]}.png"
        latency_s = await _run_generate_and_save(
            engine,
            prompt=prompt,
            request_id=request_id,
            output_path=output_path,
            label=f"{round_name}[{idx}]",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )
        latencies_s.append(latency_s)
        records.append(
            {
                "index": idx,
                "prompt": prompt,
                "request_id": request_id,
                "output_path": str(output_path),
                "e2e_latency_s": latency_s,
            }
        )
        print(f"[{round_name}] image {idx} e2e latency {latency_s:.3f}s -> {output_path}")

    round_total_s = sum(latencies_s)
    round_avg_s = mean(latencies_s) if latencies_s else 0.0
    round_result = {
        "round_name": round_name,
        "output_dir": str(round_dir),
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "num_prompts": len(prompts),
        "images": records,
        "round_total_e2e_latency_s": round_total_s,
        "avg_e2e_latency_s": round_avg_s,
    }
    print(
        f"[{round_name}] summary total_e2e_latency_s={round_total_s:.3f} "
        f"avg_e2e_latency_s={round_avg_s:.3f}"
    )
    return round_result


async def _run_test_flow(tmp_output_dir: Path) -> None:
    engine = AsyncOmni(
        model=MODEL_PATH,
        parallel_config=DiffusionParallelConfig(ulysses_degree=2),
        master_port=_get_free_port(),
        shm_threshold_bytes=sys.maxsize,
    )
    try:
        sp2_path = tmp_output_dir / "sp2.png"
        sp1_path = tmp_output_dir / "sp1.png"
        sp2_roundtrip_path = tmp_output_dir / "sp2_roundtrip.png"

        print("[SP smoke] phase=sp2 begin")
        sp2_time_s = await _run_generate_and_save(
            engine,
            prompt=PROMPT,
            request_id="sp2-request",
            output_path=sp2_path,
            label="SP=2",
        )
        print(f"[SP=2] generate took {sp2_time_s:.3f}s, saved to {sp2_path}")

        print("[SP smoke] phase=shrink begin")
        await engine.shrink_sp_one()
        print("[SP smoke] phase=shrink done")

        print("[SP smoke] phase=sp1 begin")
        sp1_time_s = await _run_generate_and_save(
            engine,
            prompt=PROMPT,
            request_id="sp1-request",
            output_path=sp1_path,
            label="SP=1",
        )
        print(f"[SP=1] generate took {sp1_time_s:.3f}s, saved to {sp1_path}")

        print("[SP smoke] phase=extend begin")
        await engine.extend_sp_two()
        print("[SP smoke] phase=extend done")

        print("[SP smoke] phase=sp2_roundtrip begin")
        sp2_roundtrip_time_s = await _run_generate_and_save(
            engine,
            prompt=PROMPT,
            request_id="sp2-roundtrip-request",
            output_path=sp2_roundtrip_path,
            label="SP=2-roundtrip",
        )
        print(
            f"[SP=2-roundtrip] generate took {sp2_roundtrip_time_s:.3f}s, "
            f"saved to {sp2_roundtrip_path}"
        )
    finally:
        engine.shutdown()
        cleanup_dist_env_and_memory()


async def _run_benchmark_flow(tmp_output_dir: Path) -> None:
    engine = AsyncOmni(
        model=MODEL_PATH,
        parallel_config=DiffusionParallelConfig(ulysses_degree=2),
        master_port=_get_free_port(),
        shm_threshold_bytes=sys.maxsize,
    )
    try:
        benchmark_result: dict[str, object] = {
            "model": MODEL_PATH,
            "prompts": BENCH_PROMPTS,
            "rounds": [],
        }

        print("[SP bench] phase=sp2 begin")
        round_sp2 = await _run_generate_batch_and_collect(
            engine,
            round_name="sp2",
            prompts=BENCH_PROMPTS,
            output_root=tmp_output_dir,
            height=1280,
            width=1280,
        )
        benchmark_result["rounds"].append(round_sp2)
        print("[SP bench] phase=shrink begin")
        await engine.shrink_sp_one()
        print("[SP bench] phase=shrink done")

        print("[SP bench] phase=sp1 begin")
        round_sp1 = await _run_generate_batch_and_collect(
            engine,
            round_name="sp1",
            prompts=BENCH_PROMPTS,
            output_root=tmp_output_dir,
            height=1280,
            width=1280,
        )
        benchmark_result["rounds"].append(round_sp1)

        print("[SP bench] phase=extend begin")
        await engine.extend_sp_two()
        print("[SP bench] phase=extend done")

        print("[SP bench] phase=sp2_roundtrip begin")
        round_sp2_roundtrip = await _run_generate_batch_and_collect(
            engine,
            round_name="sp2_roundtrip",
            prompts=BENCH_PROMPTS,
            output_root=tmp_output_dir,
            height=1280,
            width=1280,
        )
        benchmark_result["rounds"].append(round_sp2_roundtrip)

        benchmark_json_path = tmp_output_dir / "sp_switch_benchmark.json"
        benchmark_json_path.write_text(json.dumps(benchmark_result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SP bench] wrote benchmark summary json to {benchmark_json_path}")

        print("[SP bench] summary")
        for round_result in benchmark_result["rounds"]:
            rr = round_result  # help type checkers keep the intent clear
            assert isinstance(rr, dict)
            print(
                f"[SP bench] {rr['round_name']}: total_e2e_latency_s={rr['round_total_e2e_latency_s']:.3f}, "
                f"avg_e2e_latency_s={rr['avg_e2e_latency_s']:.3f}, output_dir={rr['output_dir']}"
            )
    finally:
        engine.shutdown()
        cleanup_dist_env_and_memory()


async def _run_transition_wait_flow(tmp_output_dir: Path) -> None:
    engine = AsyncOmni(
        model=MODEL_PATH,
        parallel_config=DiffusionParallelConfig(ulysses_degree=2),
        master_port=_get_free_port(),
        shm_threshold_bytes=sys.maxsize,
    )
    try:
        wait_dir = tmp_output_dir / "transition_wait"
        wait_dir.mkdir(parents=True, exist_ok=True)
        sp2_path = wait_dir / "sp2_before_shrink.png"
        sp1_path = wait_dir / "sp1_after_shrink.png"
        sp2_roundtrip_path = wait_dir / "sp2_after_extend.png"

        print("[SP wait] phase=sp2 begin")
        gen_task = asyncio.create_task(
            _run_generate_and_save(
                engine,
                prompt=PROMPT,
                request_id="sp2-wait-request",
                output_path=sp2_path,
                label="SP=2-wait",
                height=1280,
                width=1280,
                num_inference_steps=40,
            )
        )
        await asyncio.sleep(0.05)

        print("[SP wait] phase=shrink begin while sp2 request is still running")
        shrink_task = asyncio.create_task(engine.shrink_sp_one())
        await asyncio.sleep(0.1)
        assert not shrink_task.done(), "shrink_sp_one() finished before the in-flight request drained."
        print("[SP wait] shrink is waiting for the in-flight request to finish")

        sp2_time_s = await gen_task
        print(f"[SP=2-wait] generate took {sp2_time_s:.3f}s, saved to {sp2_path}")

        await shrink_task
        print("[SP wait] phase=shrink done")

        print("[SP wait] phase=sp1 begin")
        sp1_time_s = await _run_generate_and_save(
            engine,
            prompt=PROMPT,
            request_id="sp1-wait-request",
            output_path=sp1_path,
            label="SP=1-wait",
            height=1280,
            width=1280,
            num_inference_steps=40,
        )
        print(f"[SP=1-wait] generate took {sp1_time_s:.3f}s, saved to {sp1_path}")

        print("[SP wait] phase=extend begin")
        await engine.extend_sp_two()
        print("[SP wait] phase=extend done")

        print("[SP wait] phase=sp2_roundtrip begin")
        sp2_roundtrip_time_s = await _run_generate_and_save(
            engine,
            prompt=PROMPT,
            request_id="sp2-wait-roundtrip-request",
            output_path=sp2_roundtrip_path,
            label="SP=2-wait-roundtrip",
            height=1280,
            width=1280,
            num_inference_steps=40,
        )
        print(
            f"[SP=2-wait-roundtrip] generate took {sp2_roundtrip_time_s:.3f}s, "
            f"saved to {sp2_roundtrip_path}"
        )
    finally:
        engine.shutdown()
        cleanup_dist_env_and_memory()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 2, "rocm": 2})
def test_async_omni_sp_switch_smoke() -> None:
    if current_omni_platform.is_npu():
        pytest.skip("This smoke test is currently intended for CUDA/ROCm diffusion workers.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run_test_flow(OUTPUT_DIR))


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 2, "rocm": 2})
def test_async_omni_sp_switch_benchmark() -> None:
    if current_omni_platform.is_npu():
        pytest.skip("This benchmark test is currently intended for CUDA/ROCm diffusion workers.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run_benchmark_flow(OUTPUT_DIR))


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 2, "rocm": 2})
def test_async_omni_sp_transition_waits_for_inflight_request() -> None:
    if current_omni_platform.is_npu():
        pytest.skip("This integration test is currently intended for CUDA/ROCm diffusion workers.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run_transition_wait_flow(OUTPUT_DIR))
