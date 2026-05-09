# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal smoke test for AsyncOmni SP switching.

This test starts AsyncOmni with SP=2, runs one generation, shrinks to SP=1,
runs a second generation, and saves both images under ./output.
"""

import asyncio
import gc
import os
import sys
import socket
import time
from pathlib import Path

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


def _make_sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=20,
        guidance_scale=0.0,
        generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
        num_outputs_per_prompt=1,
    )


async def _run_generate_and_save(
    engine: AsyncOmni,
    *,
    prompt: str,
    request_id: str,
    output_path: Path,
    label: str,
) -> float:
    print(f"[{label}] starting generate request_id={request_id}")
    start = time.perf_counter()
    last_output: OmniRequestOutput | None = None
    omni_output: OmniRequestOutput | None = None
    async for omni_output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=[_make_sampling_params()],
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
