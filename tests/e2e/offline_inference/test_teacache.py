# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for TeaCache backend.

This test verifies that TeaCache acceleration works correctly with diffusion models.
It uses minimal settings to keep test time short for CI.
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

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

_MODEL_ENV_VAR = "VLLM_OMNI_STEPWISE_TEST_MODEL"
_TOKENIZER_MAX_LENGTH = 1024
_PROMPT_TEMPLATE_ENCODE_START_IDX = 34
_APPLY_CHAT_TEMPLATE_KWARGS = {
    "max_length": _TOKENIZER_MAX_LENGTH + _PROMPT_TEMPLATE_ENCODE_START_IDX,
    "padding": True,
    "truncation": True,
}
_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
_PROMPT_TEXT = "A green apple on a wooden table."
_NEG_PROMPT_TEXT = " "
_TEACACHE_OUTPUT_DIR = Path("output/teacache_custom_pipeline")


def _get_model_name_from_env() -> str:
    model_name = os.environ.get(_MODEL_ENV_VAR)
    if not model_name:
        raise RuntimeError(
            f"Missing required environment variable '{_MODEL_ENV_VAR}'. "
            f"Please export {_MODEL_ENV_VAR}=/tmp/models/Qwen/Qwen-Image before running this test."
        )
    return model_name


def _load_tokenizer_from_model(model: str):
    tokenizer_dir = Path(model) / "tokenizer"
    if not tokenizer_dir.exists():
        pytest.skip(f"Expected tokenizer at '{tokenizer_dir}', but it does not exist.")
    return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=True)


def _normalize_token_ids(tokenized_output: Any) -> list[int]:
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict) and "input_ids" in tokenized_output:
        token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])
    return [int(x.item() if hasattr(x, "item") else x) for x in token_ids]


def _chat_template_to_ids(tokenizer, messages):
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **_APPLY_CHAT_TEMPLATE_KWARGS,
    )
    return _normalize_token_ids(token_ids), None


def _make_sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_inference_steps=50,
        true_cfg_scale=4.0,
        width=512,
        height=512,
        output_type="pil",
        seed=42,
        extra_args={"noise_level": 1, "sde_type": "sde", "logprobs": True, "sde_window_size": 2, "sde_window_range": [0, 5]},
    )


def _to_image_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        raise RuntimeError(f"Unexpected output tensor shape: {tuple(t.shape)}")
    if t.shape[0] == 3:
        chw = t
    elif t.shape[-1] == 3:
        chw = t.permute(2, 0, 1)
    else:
        raise RuntimeError(f"Cannot infer channel dim from shape: {tuple(t.shape)}")
    chw = chw.detach().cpu().float()
    vmin = float(chw.min())
    vmax = float(chw.max())
    if vmax > vmin:
        chw = (chw - vmin) / (vmax - vmin)
    else:
        chw = torch.zeros_like(chw)
    return chw


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    img_tensor = _to_image_tensor(t)
    image_uint8 = (img_tensor * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(image_uint8)


def _extract_output_image(output: OmniRequestOutput) -> Image.Image:
    candidates = list(getattr(output, "images", []) or [])
    inner = getattr(output, "request_output", None)
    if isinstance(inner, OmniRequestOutput):
        candidates.extend(getattr(inner, "images", []) or [])

    for item in candidates:
        if isinstance(item, torch.Tensor):
            return _tensor_to_pil(item)
        if isinstance(item, Image.Image):
            return item.convert("RGB")

    raise RuntimeError("No image/tensor found in AsyncOmni output.images.")


def _save_teacache_image(final_output: OmniRequestOutput) -> Path:
    if _TEACACHE_OUTPUT_DIR.exists():
        shutil.rmtree(_TEACACHE_OUTPUT_DIR)
    _TEACACHE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = _extract_output_image(final_output)
    image_path = _TEACACHE_OUTPUT_DIR / "output.png"
    image.save(image_path)
    return image_path


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
def test_teacache():
    """Test TeaCache backend with diffusion model."""
    model_name = _get_model_name_from_env()
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

        # Keep core sampling settings aligned with other AsyncOmni diffusion workloads.
        height = 512
        width = 512
        num_inference_steps = 50

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
    tokenizer = _load_tokenizer_from_model(model_name)
    prompt_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _PROMPT_TEXT},
    ]
    negative_prompt_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _NEG_PROMPT_TEXT},
    ]
    prompt_ids, prompt_mask = _chat_template_to_ids(tokenizer, prompt_messages)
    negative_prompt_ids, negative_prompt_mask = _chat_template_to_ids(tokenizer, negative_prompt_messages)
    prompt = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "negative_prompt_ids": negative_prompt_ids,
        "negative_prompt_mask": negative_prompt_mask,
    }

    # Use a very large threshold to force cache reuse after the first step.
    # This makes skipped-step behavior deterministic for the logging assertion.
    cache_config = {
        "rel_l1_thresh": 1e6,
    }

    sampling_params = _make_sampling_params()

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
def test_async_teacache_custom_pipeline_logs_skipped_steps(
    capfd: pytest.CaptureFixture[str],
):
    """Test TeaCache skip-step logging with AsyncOmni + custom non-step Qwen pipeline."""
    model_name = _get_model_name_from_env()
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

    try:
        _extract_output_image(final_output)
    except Exception as e:
        raise AssertionError(
            "AsyncOmni custom pipeline TeaCache test produced no extractable image output."
        ) from e

    captured = capfd.readouterr()
    combined_logs = f"{captured.out}\n{captured.err}"
    assert "TEACACHE_STEP_SKIPPED" in combined_logs, (
        "TeaCache skip-step log not found. Expected log marker 'TEACACHE_STEP_SKIPPED' "
        "from custom pipeline AsyncOmni generation."
    )

    image_path = _save_teacache_image(final_output)
    assert image_path.exists(), (
        f"TeaCache output image was not saved to expected path: {image_path}."
    )
