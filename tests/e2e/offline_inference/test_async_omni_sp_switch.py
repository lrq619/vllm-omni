# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal smoke test for AsyncOmni SP switching.

This test starts AsyncOmni with SP=2, runs one generation, shrinks to SP=1,
runs a second generation, and saves both images under ./output.
"""

import asyncio
import os
import sys
import time
import socket
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


def _extract_single_image(outputs: list[OmniRequestOutput]) -> Image.Image:
    if not outputs:
        raise ValueError("Empty outputs from AsyncOmni.generate()")

    first_output = outputs[-1]
    if first_output.final_output_type != "image":
        raise ValueError(f"Expected final output type 'image', got {first_output.final_output_type!r}")
    if not first_output.images:
        raise ValueError("No images returned by AsyncOmni.generate()")

    image = first_output.images[0]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")
    return image


def _make_sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=2,
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
) -> float:
    start = time.perf_counter()
    outputs = []
    async for omni_output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=[_make_sampling_params()],
    ):
        outputs.append(omni_output)
    elapsed_s = time.perf_counter() - start

    image = _extract_single_image(outputs)
    image.save(output_path)
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

        sp2_time_s = await _run_generate_and_save(
            engine,
            prompt=PROMPT,
            request_id="sp2-request",
            output_path=sp2_path,
        )
        print(f"[SP=2] generate took {sp2_time_s:.3f}s, saved to {sp2_path}")

        await engine.shrink_sp_one()

        sp1_time_s = await _run_generate_and_save(
            engine,
            prompt=PROMPT,
            request_id="sp1-request",
            output_path=sp1_path,
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
