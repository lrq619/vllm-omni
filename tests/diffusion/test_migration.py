# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.diffusion]

logger = init_logger(__name__)

_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
_PROMPT_TEXT = "A green apple on a wooden table."
_NEG_PROMPT_TEXT = " "
_TOKENIZER_MAX_LENGTH = 1024
_PROMPT_TEMPLATE_ENCODE_START_IDX = 34
_APPLY_CHAT_TEMPLATE_KWARGS = {
    "max_length": _TOKENIZER_MAX_LENGTH + _PROMPT_TEMPLATE_ENCODE_START_IDX,
    "padding": True,
    "truncation": True,
}


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


def _serialize_tensor(t: torch.Tensor) -> dict[str, Any]:
    t_cpu = t.detach().cpu()
    return {
        "__type__": "tensor",
        "dtype": str(t_cpu.dtype),
        "shape": list(t_cpu.shape),
        "data": t_cpu.tolist(),
    }


def _serialize_obj(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return _serialize_tensor(obj)
    if isinstance(obj, dict):
        return {str(k): _serialize_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_obj(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    img_tensor = _to_image_tensor(t)
    image_uint8 = (img_tensor * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(image_uint8)


def _extract_output_tensor_and_image(output: OmniRequestOutput) -> tuple[torch.Tensor, Image.Image]:
    candidates = list(getattr(output, "images", []) or [])
    inner = getattr(output, "request_output", None)
    if isinstance(inner, OmniRequestOutput):
        candidates.extend(getattr(inner, "images", []) or [])

    for item in candidates:
        if isinstance(item, torch.Tensor):
            return item, _tensor_to_pil(item)
        if isinstance(item, Image.Image):
            img = item.convert("RGB")
            img_t = torch.from_numpy(np.array(img))
            return img_t, img

    raise RuntimeError("No image/tensor found in AsyncOmni output.images.")


def _write_request_result(
    out_root: Path,
    req_index: int,
    prompt_text: str,
    prompt_ids: list[int],
    negative_prompt_ids: list[int],
    output: OmniRequestOutput,
) -> dict[str, Any]:
    req_dir = out_root / f"req_{req_index}"
    req_dir.mkdir(parents=True, exist_ok=True)

    output_tensor, output_image = _extract_output_tensor_and_image(output)
    image_path = req_dir / "output.png"
    output_image.save(image_path)

    custom_output = {}
    if isinstance(output, OmniRequestOutput):
        custom_output = output.custom_output or {}

    payload = {
        "request_id": output.request_id,
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids,
        "negative_prompt_ids": negative_prompt_ids,
        "image_path": str(image_path),
        "output_tensor": _serialize_tensor(output_tensor),
        "custom_output": _serialize_obj(custom_output),
    }
    (req_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return payload


def _write_single_stage_config(config_path: Path, device_idx: int) -> None:
    config_path.write_text(
        "\n".join(
            [
                "stage_args:",
                "  - stage_id: 0",
                "    stage_type: diffusion",
                "    runtime:",
                "      process: true",
                f"      devices: \"{device_idx}\"",
                "      max_batch_size: 1",
                "    engine_args:",
                "      model_stage: diffusion",
                "      enable_stepwise: true",
                "      num_gpus: 1",
                "      diffusion_load_format: custom_pipeline",
                "      custom_pipeline_args:",
                "        pipeline_class: vllm_omni.diffusion.models.qwen_image.pipeline_qwenimage_step.QwenImagePipelineWithLogProbStep",
                "    final_output: true",
                "    final_output_type: image",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _create_async_omni(model: str, config_path: Path) -> AsyncOmni:
    return AsyncOmni(
        model=model,
        stage_configs_path=str(config_path),
        num_gpus=1,
    )


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


async def _run_request(
    omni: AsyncOmni,
    *,
    req_index: int,
    request_id: str,
    prompt_text: str,
    prompt_ids: list[int],
    prompt_mask: list[int],
    negative_prompt_ids: list[int],
    negative_prompt_mask: list[int],
    sampling_params: OmniDiffusionSamplingParams,
    out_root: Path,
    is_remote: bool = False,
) -> tuple[OmniRequestOutput, dict[str, Any]]:
    logger.info(
        "[MigrationTrace] Starting request request_id=%s req_index=%s is_remote=%s out_root=%s",
        request_id,
        req_index,
        is_remote,
        out_root,
    )
    final_output: OmniRequestOutput | None = None
    try:
        async for out in omni.generate(
            prompt={
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "negative_prompt_ids": negative_prompt_ids,
                "negative_prompt_mask": negative_prompt_mask,
            },
            request_id=request_id,
            sampling_params_list=[sampling_params],
            is_remote=is_remote,
        ):
            logger.info(
                "[MigrationTrace] Streaming output request_id=%s finished=%s stage_id=%s",
                request_id,
                getattr(out, "finished", None),
                getattr(out, "stage_id", None),
            )
            final_output = out
    except Exception:
        logger.exception("[MigrationTrace] Request failed request_id=%s is_remote=%s", request_id, is_remote)
        raise
    if final_output is None:
        raise RuntimeError(f"No final output for request_id={request_id}")
    payload = _write_request_result(
        out_root=out_root,
        req_index=req_index,
        prompt_text=prompt_text,
        prompt_ids=prompt_ids,
        negative_prompt_ids=negative_prompt_ids,
        output=final_output,
    )
    logger.info(
        "[MigrationTrace] Completed request request_id=%s req_index=%s is_remote=%s",
        request_id,
        req_index,
        is_remote,
    )
    return final_output, payload


async def _run_migration(tmp_path: Path) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 GPUs.")

    model = os.getenv("VLLM_OMNI_STEPWISE_TEST_MODEL")
    if not model:
        pytest.skip("Set VLLM_OMNI_STEPWISE_TEST_MODEL to a valid stepwise-capable model path/name.")

    tokenizer = _load_tokenizer_from_model(model)
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

    out_root = Path("output/test_migration")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    config0 = tmp_path / "stage_gpu0.yaml"
    config1 = tmp_path / "stage_gpu1.yaml"
    _write_single_stage_config(config0, 0)
    _write_single_stage_config(config1, 1)

    omni0 = _create_async_omni(model=model, config_path=config0)
    omni1 = _create_async_omni(model=model, config_path=config1)

    try:
        req0_id = "migration-req-0"
        req1_id = "migration-req-1"
        logger.info("[MigrationTrace] Created AsyncOmni instances; launching local requests")

        task0 = asyncio.create_task(
            _run_request(
                omni0,
                req_index=0,
                request_id=req0_id,
                prompt_text=_PROMPT_TEXT,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_mask=negative_prompt_mask,
                sampling_params=_make_sampling_params(),
                out_root=out_root / "paused",
            )
        )
        task1 = asyncio.create_task(
            _run_request(
                omni0,
                req_index=1,
                request_id=req1_id,
                prompt_text=_PROMPT_TEXT,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_mask=negative_prompt_mask,
                sampling_params=_make_sampling_params(),
                out_root=out_root / "baseline",
            )
        )

        logger.info("[MigrationTrace] Sleeping before pause request_id=%s", req0_id)
        await asyncio.sleep(3)
        logger.info("[MigrationTrace] Calling pause for request_id=%s", req0_id)
        await omni0.pause([req0_id])
        logger.info("[MigrationTrace] Pause returned for request_id=%s", req0_id)

        logger.info("[MigrationTrace] Awaiting paused task request_id=%s", req0_id)
        paused_output, paused_payload = await task0
        logger.info(
            "[MigrationTrace] Paused task completed request_id=%s output_request_id=%s",
            req0_id,
            paused_output.request_id,
        )

        logger.info("[MigrationTrace] Launching remote replay request_id=%s", req0_id)
        remote_task = asyncio.create_task(
            _run_request(
                omni1,
                req_index=0,
                request_id=req0_id,
                prompt_text=_PROMPT_TEXT,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_mask=negative_prompt_mask,
                sampling_params=_make_sampling_params(),
                out_root=out_root / "remote",
                is_remote=True,
            )
        )

        logger.info("[MigrationTrace] Awaiting baseline task request_id=%s", req1_id)
        baseline_output, baseline_payload = await task1
        logger.info(
            "[MigrationTrace] Baseline task completed request_id=%s output_request_id=%s",
            req1_id,
            baseline_output.request_id,
        )
        logger.info("[MigrationTrace] Awaiting remote task request_id=%s", req0_id)
        remote_output, remote_payload = await remote_task
        logger.info(
            "[MigrationTrace] Remote task completed request_id=%s output_request_id=%s",
            req0_id,
            remote_output.request_id,
        )

        normalized_baseline = dict(baseline_payload)
        normalized_remote = dict(remote_payload)
        normalized_baseline.pop("request_id", None)
        normalized_baseline.pop("image_path", None)
        normalized_remote.pop("request_id", None)
        normalized_remote.pop("image_path", None)

        assert normalized_baseline == normalized_remote, "Migrated output does not match baseline output."

        baseline_img = Image.open(out_root / "baseline" / "req_1" / "output.png").convert("RGB")
        remote_img = Image.open(out_root / "remote" / "req_0" / "output.png").convert("RGB")
        assert np.array_equal(np.array(baseline_img), np.array(remote_img)), "Migrated image does not match baseline image."

        # Persist a compact comparison artifact for debugging.
        comparison = {
            "paused_request_id": req0_id,
            "baseline_request_id": req1_id,
            "remote_request_id": req0_id,
            "baseline_vs_remote_equal": True,
            "paused_payload": paused_payload,
            "baseline_payload": baseline_payload,
            "remote_payload": remote_payload,
            "paused_output_request_id": paused_output.request_id,
            "baseline_output_request_id": baseline_output.request_id,
            "remote_output_request_id": remote_output.request_id,
        }
        (out_root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=True), encoding="utf-8")
    except Exception:
        logger.exception("[MigrationTrace] Migration flow failed before cleanup")
        raise
    finally:
        logger.info("[MigrationTrace] Entering cleanup for migration test")
        omni0.close()
        omni1.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_migration(tmp_path: Path):
    asyncio.run(_run_migration(tmp_path))
