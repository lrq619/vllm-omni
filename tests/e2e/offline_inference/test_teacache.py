# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for TeaCache backend.

This test verifies that TeaCache acceleration works correctly with diffusion models.
It uses minimal settings to keep test time short for CI.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import AsyncOmni, Omni
from vllm_omni.outputs import OmniRequestOutput

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# Use random weights model for testing
models = ["riverclouds/qwen_image_random"]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("model_name", models)
def test_teacache(model_name: str):
    """Test TeaCache backend with diffusion model."""
    # Configure TeaCache with default settings for fast testing
    cache_config = {
        "rel_l1_thresh": 0.2,  # Default threshold
    }
    m = None
    try:
        m = Omni(
            model=model_name,
            cache_backend="tea_cache",
            cache_config=cache_config,
        )

        # Use minimal settings for fast testing
        height = 256
        width = 256
        num_inference_steps = 4  # Minimal steps for fast test

        outputs = m.generate(
            "a photo of a cat sitting on a laptop keyboard",
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                num_outputs_per_prompt=1,  # Single output for speed
            ),
        )
        # Extract images from request_output[0]['images']
        first_output = outputs[0]
        assert first_output.final_output_type == "image"
        if not hasattr(first_output, "request_output") or not first_output.request_output:
            raise ValueError("No request_output found in OmniRequestOutput")

        req_out = first_output.request_output[0]
        if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
            raise ValueError("Invalid request_output structure or missing 'images' key")

        images = req_out.images

        # Verify generation succeeded
        assert images is not None
        assert len(images) == 1
        # Check image size
        assert images[0].width == width
        assert images[0].height == height
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


async def _run_async_teacache_custom_pipeline(model_name: str) -> OmniRequestOutput:
    prompt_len = 80
    prompt_ids = list(range(1, prompt_len + 1))
    negative_prompt_ids = [2] * prompt_len
    prompt = {
        "prompt_ids": prompt_ids,
        "prompt_mask": [1] * prompt_len,
        "negative_prompt_ids": negative_prompt_ids,
        "negative_prompt_mask": [1] * prompt_len,
    }

    # Use a very large threshold to force cache reuse after the first step.
    # This makes skipped-step behavior deterministic for the logging assertion.
    cache_config = {
        "rel_l1_thresh": 1e6,
    }

    sampling_params = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=6,
        guidance_scale=0.0,
        true_cfg_scale=4.0,
        generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
        num_outputs_per_prompt=1,
    )

    omni = None
    final_output = None
    try:
        omni = AsyncOmni(
            model=model_name,
            num_gpus=1,
            diffusion_load_format="custom_pipeline",
            custom_pipeline_args={
                "pipeline_class": "vllm_omni.diffusion.models.qwen_image.non_step.pipeline_qwenimage."
                "QwenImagePipelineWithLogProb"
            },
            cache_backend="tea_cache",
            cache_config=cache_config,
        )

        async for out in omni.generate(
            prompt=prompt,
            request_id="async-teacache-custom-pipeline",
            sampling_params_list=[sampling_params],
        ):
            final_output = out
    finally:
        if omni is not None and hasattr(omni, "close"):
            omni.close()

    if final_output is None:
        raise RuntimeError(
            "AsyncOmni.generate() returned no final output for custom pipeline TeaCache test. "
            "Please check diffusion worker logs for request_id=async-teacache-custom-pipeline."
        )
    return final_output


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("model_name", models)
def test_async_teacache_custom_pipeline_logs_skipped_steps(
    model_name: str,
    capfd: pytest.CaptureFixture[str],
):
    """Test TeaCache skip-step logging with AsyncOmni + custom non-step Qwen pipeline."""
    final_output = asyncio.run(_run_async_teacache_custom_pipeline(model_name))

    if not hasattr(final_output, "final_output_type"):
        raise TypeError(
            "AsyncOmni final output is missing attribute 'final_output_type'. "
            f"Got type={type(final_output)!r}."
        )
    assert final_output.final_output_type == "image", (
        "Expected final_output_type='image' in AsyncOmni custom pipeline TeaCache test, "
        f"got {final_output.final_output_type!r}."
    )

    if not hasattr(final_output, "request_output") or not final_output.request_output:
        raise ValueError(
            "AsyncOmni final output does not contain request_output entries in custom pipeline TeaCache test."
        )

    req_out = final_output.request_output[0]
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise TypeError(
            "Unexpected request_output structure in AsyncOmni custom pipeline TeaCache test. "
            f"Type={type(req_out)!r}, has_images={hasattr(req_out, 'images')}."
        )
    if req_out.images is None or len(req_out.images) == 0:
        raise AssertionError(
            "AsyncOmni custom pipeline TeaCache test produced no images. "
            "TeaCache should still return image outputs."
        )

    captured = capfd.readouterr()
    combined_logs = f"{captured.out}\n{captured.err}"
    assert "TEACACHE_STEP_SKIPPED" in combined_logs, (
        "TeaCache skip-step log not found. Expected log marker 'TEACACHE_STEP_SKIPPED' "
        "from custom pipeline AsyncOmni generation."
    )
