# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import json
import os
import random
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
_STRESS_TEST_NUM_PAIRS = 32
_STRESS_TEST_PAIR_INTERVAL_S = 1.0
_STRESS_TEST_RANDOM_SEED = 20260331
_STRESS_TEST_PROMPTS = [
    "A red apple on a white plate.",
    "A blue car on a city street.",
    "A yellow sunflower in a glass vase.",
    "A white cat sleeping on a sofa.",
    "A glass of lemonade with ice cubes.",
    "A small red toy robot on a desk.",
    "A bowl of strawberries on a table.",
    "A brown dog sitting in a park.",
    "A blue butterfly on a green leaf.",
    "A stack of colorful books on a shelf.",
    "A bowl of ramen with chopsticks.",
    "A bright rainbow over a field.",
    "A blue bicycle leaning against a wall.",
    "A candle on a wooden table.",
    "A pair of yellow rain boots by a door.",
    "A silver watch on a black cloth.",
    "A vase of pink flowers on a windowsill.",
    "A hot air balloon in a clear sky.",
    "A lighthouse by the ocean.",
    "A slice of chocolate cake on a plate.",
    "A green frog sitting on a lily pad.",
    "A paper airplane on a desk.",
    "A bowl of oranges on a kitchen counter.",
    "A snowy cabin in the woods.",
    "A red umbrella on a rainy sidewalk.",
    "A potted cactus on a sunny windowsill.",
    "A mountain lake at sunrise.",
    "A pair of running shoes on the floor.",
    "A wooden boat floating on calm water.",
    "A blue teapot with steam rising.",
    "A city skyline at sunset.",
    "A striped beach ball on sand.",
]


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


def _normalize_mm_token_field(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.long)
    return torch.tensor(list(value), dtype=torch.long)


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
    latent_trace = {}
    if isinstance(custom_output, dict):
        latent_trace = custom_output.get("latent_trace") or {}
    (req_dir / "latents.json").write_text(json.dumps(latent_trace, ensure_ascii=True), encoding="utf-8")
    return payload


def _normalize_custom_output(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    custom_output = normalized.get("custom_output")
    if isinstance(custom_output, dict):
        custom_output = dict(custom_output)
        custom_output.pop("resume_from_step_idx", None)
        custom_output.pop("executed_step_count", None)
        custom_output.pop("latent_trace", None)
        normalized["custom_output"] = custom_output
    return normalized


def _normalize_comparison_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("request_id", None)
    normalized.pop("image_path", None)
    normalized.pop("output_tensor", None)
    return _normalize_custom_output(normalized)


def _collect_differences(left: Any, right: Any, *, path: str = "", limit: int = 32) -> list[str]:
    diffs: list[str] = []

    def walk(lhs: Any, rhs: Any, cur_path: str) -> None:
        if len(diffs) >= limit:
            return
        if type(lhs) is not type(rhs):
            diffs.append(
                f"{cur_path or '<root>'}: type mismatch {type(lhs).__name__} != {type(rhs).__name__}"
            )
            return
        if isinstance(lhs, dict):
            keys = sorted(set(lhs.keys()) | set(rhs.keys()), key=str)
            for key in keys:
                if len(diffs) >= limit:
                    return
                next_path = f"{cur_path}.{key}" if cur_path else str(key)
                if key not in lhs:
                    diffs.append(f"{next_path}: missing on left")
                elif key not in rhs:
                    diffs.append(f"{next_path}: missing on right")
                else:
                    walk(lhs[key], rhs[key], next_path)
            return
        if isinstance(lhs, list):
            if len(lhs) != len(rhs):
                diffs.append(f"{cur_path or '<root>'}: list length mismatch {len(lhs)} != {len(rhs)}")
                if len(diffs) >= limit:
                    return
            for idx, (l_item, r_item) in enumerate(zip(lhs, rhs)):
                if len(diffs) >= limit:
                    return
                walk(l_item, r_item, f"{cur_path}[{idx}]")
            return
        if isinstance(lhs, tuple):
            if len(lhs) != len(rhs):
                diffs.append(f"{cur_path or '<root>'}: tuple length mismatch {len(lhs)} != {len(rhs)}")
                if len(diffs) >= limit:
                    return
            for idx, (l_item, r_item) in enumerate(zip(lhs, rhs)):
                if len(diffs) >= limit:
                    return
                walk(l_item, r_item, f"{cur_path}({idx})")
            return
        if lhs != rhs:
            diffs.append(f"{cur_path or '<root>'}: {lhs!r} != {rhs!r}")

    walk(left, right, path)
    return diffs


def _compare_images(left: Image.Image, right: Image.Image) -> str | None:
    left_np = np.array(left.convert("RGB"))
    right_np = np.array(right.convert("RGB"))
    if left_np.shape != right_np.shape:
        return f"image shape mismatch: {left_np.shape} != {right_np.shape}"
    if np.array_equal(left_np, right_np):
        return None
    diff_mask = left_np != right_np
    differing_pixels = int(np.any(diff_mask, axis=-1).sum())
    max_abs_diff = int(np.abs(left_np.astype(np.int16) - right_np.astype(np.int16)).max())
    first_diff = np.argwhere(diff_mask)
    if first_diff.size == 0:
        return "image arrays differ but no differing pixel could be located"
    y, x, channel = first_diff[0].tolist()
    return (
        "image mismatch: "
        f"first_diff=(y={y}, x={x}, channel={channel}), "
        f"left={int(left_np[y, x, channel])}, right={int(right_np[y, x, channel])}, "
        f"differing_pixels={differing_pixels}, max_abs_diff={max_abs_diff}"
    )


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_pair_comparison(
    pair_dir: Path,
    *,
    pair_index: int,
    pause_step_idx: int,
    baseline_payload_summary: dict[str, Any],
    pause_migrate_payload_summary: dict[str, Any],
    baseline_image_path: Path,
    pause_migrate_image_path: Path,
    mismatch_details: list[str] | None = None,
) -> None:
    _copy_image(baseline_image_path, pair_dir / "baseline.png")
    _copy_image(pause_migrate_image_path, pair_dir / "pause_migrate.png")

    comparison = {
        "pair_index": pair_index,
        "pause_step_idx": pause_step_idx,
        "baseline_payload": baseline_payload_summary,
        "pause_migrate_payload": pause_migrate_payload_summary,
        "images_match": mismatch_details is None,
        "mismatch_details": mismatch_details or [],
    }
    (pair_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=True), encoding="utf-8")


def _make_pause_step_indices(*, num_inference_steps: int, num_pairs: int, seed: int) -> list[int]:
    if num_inference_steps < 2:
        raise ValueError(f"num_inference_steps must be >= 2, got {num_inference_steps}")
    max_valid_pause_idx = num_inference_steps - 2
    unique_values = max_valid_pause_idx + 1
    if num_pairs > unique_values:
        raise ValueError(
            f"Cannot sample {num_pairs} unique pause_step_idx values from 0..{max_valid_pause_idx}"
        )
    rng = random.Random(seed)
    return rng.sample(range(unique_values), num_pairs)


def _make_prompt_variants(tokenizer) -> list[tuple[str, list[int], list[int] | None]]:
    variants: list[tuple[str, list[int], list[int] | None]] = []
    for prompt_text in _STRESS_TEST_PROMPTS:
        prompt_messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        prompt_ids, prompt_mask = _chat_template_to_ids(tokenizer, prompt_messages)
        variants.append((prompt_text, prompt_ids, prompt_mask))
    return variants


def _load_latent_trace(path: Path) -> dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(trace, dict):
        raise RuntimeError(f"Expected latent trace JSON object at {path}, got {type(trace).__name__}")
    return trace


def _find_first_latent_divergence(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[str | None, list[str]]:
    common_steps = sorted(set(left.keys()) & set(right.keys()), key=lambda s: int(s))
    for step in common_steps:
        if left[step] != right[step]:
            return step, common_steps
    return None, common_steps


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
                "      max_batch_size: 16",
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


def _make_sampling_params(
    *,
    sde_window_size: int = 2,
    sde_window_range: tuple[int, int] = (0, 5),
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_inference_steps=50,
        true_cfg_scale=4.0,
        width=512,
        height=512,
        output_type="pil",
        seed=42,
        extra_args={
            "noise_level": 1,
            "sde_type": "sde",
            "logprobs": True,
            "sde_window_size": sde_window_size,
            "sde_window_range": list(sde_window_range),
        },
    )


def _assert_tensor_fields_match(
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    *,
    field_names: tuple[str, ...],
) -> None:
    left_custom_output = left_payload.get("custom_output")
    right_custom_output = right_payload.get("custom_output")
    if not isinstance(left_custom_output, dict):
        raise RuntimeError(f"Expected left custom_output to be a dict, got {type(left_custom_output).__name__}")
    if not isinstance(right_custom_output, dict):
        raise RuntimeError(f"Expected right custom_output to be a dict, got {type(right_custom_output).__name__}")

    for field_name in field_names:
        left_value = left_custom_output.get(field_name)
        right_value = right_custom_output.get(field_name)
        if left_value != right_value:
            raise RuntimeError(f"Mismatch in custom_output.{field_name}")


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
    pause_step_idx: int | None = None,
) -> tuple[OmniRequestOutput, dict[str, Any]]:
    logger.info(
        "[MigrationTrace] Starting request request_id=%s req_index=%s is_remote=%s out_root=%s",
        request_id,
        req_index,
        is_remote,
        out_root,
    )
    final_output: OmniRequestOutput | None = None
    prompt_ids_tensor = _normalize_mm_token_field(prompt_ids)
    try:
        async for out in omni.generate(
            prompt={
                "prompt_ids": prompt_ids_tensor,
                "prompt_mask": prompt_mask,
                "negative_prompt_ids": negative_prompt_ids,
                "negative_prompt_mask": negative_prompt_mask,
            },
            request_id=request_id,
            sampling_params_list=[sampling_params],
            is_remote=is_remote,
            pause_step_idx=pause_step_idx,
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
    negative_prompt_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _NEG_PROMPT_TEXT},
    ]
    negative_prompt_ids, negative_prompt_mask = _chat_template_to_ids(tokenizer, negative_prompt_messages)
    prompt_variants = _make_prompt_variants(tokenizer)
    if len(prompt_variants) != _STRESS_TEST_NUM_PAIRS:
        raise RuntimeError(
            f"Expected {_STRESS_TEST_NUM_PAIRS} stress prompts, got {len(prompt_variants)}"
        )

    out_root = Path("output/test_migration")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    config0 = tmp_path / "stage_gpu0.yaml"
    config1 = tmp_path / "stage_gpu1.yaml"
    _write_single_stage_config(config0, 0)
    _write_single_stage_config(config1, 1)

    # Construct both AsyncOmni instances in parallel so the two GPU stages
    # come up independently instead of serializing startup.
    omni0, omni1 = await asyncio.gather(
        asyncio.to_thread(_create_async_omni, model=model, config_path=config0),
        asyncio.to_thread(_create_async_omni, model=model, config_path=config1),
    )

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
        paused_custom_output = paused_output.custom_output or {}
        paused_step_idx = paused_custom_output.get("step_idx")
        paused_executed_step_count = paused_custom_output.get("executed_step_count")

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
        remote_custom_output = remote_output.custom_output or {}
        remote_resume_from_step_idx = remote_custom_output.get("resume_from_step_idx")
        remote_executed_step_count = remote_custom_output.get("executed_step_count")
        if remote_resume_from_step_idx != paused_step_idx:
            raise RuntimeError(
                f"Remote request resumed from unexpected step_idx for request_id={req0_id}: "
                f"expected {paused_step_idx}, got {remote_resume_from_step_idx}"
            )
        expected_remote_steps = int(remote_custom_output.get("max_steps", 0)) - int(paused_step_idx)
        if remote_executed_step_count != expected_remote_steps:
            raise RuntimeError(
                f"Remote request executed unexpected number of steps for request_id={req0_id}: "
                f"expected {expected_remote_steps}, got {remote_executed_step_count}"
            )
        if paused_executed_step_count != paused_step_idx:
            raise RuntimeError(
                f"Paused request executed unexpected number of steps for request_id={req0_id}: "
                f"expected {paused_step_idx}, got {paused_executed_step_count}"
            )

        normalized_baseline = _normalize_custom_output(dict(baseline_payload))
        normalized_remote = _normalize_custom_output(dict(remote_payload))
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


async def _run_migration_with_pause_step_idx(
    tmp_path: Path,
    *,
    sampling_params: OmniDiffusionSamplingParams | None = None,
    out_root_name: str = "output/test_migration_pause_step_idx",
    config_suffix: str = "pause_step_idx",
    pause_step_idx: int = 25,
) -> None:
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

    out_root = Path(out_root_name)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    config0 = tmp_path / f"stage_gpu0_{config_suffix}.yaml"
    config1 = tmp_path / f"stage_gpu1_{config_suffix}.yaml"
    _write_single_stage_config(config0, 0)
    _write_single_stage_config(config1, 1)

    if sampling_params is None:
        sampling_params = _make_sampling_params()

    # Construct both AsyncOmni instances in parallel so the two GPU stages
    # come up independently instead of serializing startup.
    omni0, omni1 = await asyncio.gather(
        asyncio.to_thread(_create_async_omni, model=model, config_path=config0),
        asyncio.to_thread(_create_async_omni, model=model, config_path=config1),
    )

    try:
        req0_id = "migration-req-pause-step-idx-0"
        req1_id = "migration-req-pause-step-idx-1"
        logger.info(
            "[MigrationTrace] Created AsyncOmni instances; launching local requests with pause_step_idx=%d",
            pause_step_idx,
        )

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
                sampling_params=sampling_params,
                out_root=out_root / "paused",
                pause_step_idx=pause_step_idx,
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
                sampling_params=sampling_params,
                out_root=out_root / "baseline",
            )
        )

        logger.info("[MigrationTrace] Awaiting auto-paused task request_id=%s", req0_id)
        paused_output, paused_payload = await task0
        paused_custom_output = paused_payload.get("custom_output")
        if not isinstance(paused_custom_output, dict):
            raise RuntimeError(
                f"Missing custom_output for paused request_id={req0_id}: {type(paused_custom_output).__name__}"
            )
        paused_finish_reason = paused_custom_output.get("finish_reason")
        if paused_finish_reason != "paused":
            raise RuntimeError(
                f"Expected paused finish_reason for request_id={req0_id}, got {paused_finish_reason!r}"
            )
        paused_step_idx = paused_custom_output.get("step_idx")
        expected_paused_step_idx = pause_step_idx + 1
        if paused_step_idx != expected_paused_step_idx:
            raise RuntimeError(
                f"Unexpected paused step_idx for request_id={req0_id}: "
                f"expected {expected_paused_step_idx}, got {paused_step_idx}"
            )
        paused_executed_step_count = paused_custom_output.get("executed_step_count")
        if paused_executed_step_count != expected_paused_step_idx:
            raise RuntimeError(
                f"Unexpected paused executed_step_count for request_id={req0_id}: "
                f"expected {expected_paused_step_idx}, got {paused_executed_step_count}"
            )
        logger.info(
            "[MigrationTrace] Auto-paused task completed request_id=%s step_idx=%s",
            req0_id,
            paused_step_idx,
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
                sampling_params=sampling_params,
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
        remote_custom_output = remote_output.custom_output or {}
        remote_resume_from_step_idx = remote_custom_output.get("resume_from_step_idx")
        if remote_resume_from_step_idx != expected_paused_step_idx:
            raise RuntimeError(
                f"Remote request resumed from unexpected step_idx for request_id={req0_id}: "
                f"expected {expected_paused_step_idx}, got {remote_resume_from_step_idx}"
            )
        remote_executed_step_count = remote_custom_output.get("executed_step_count")
        expected_remote_steps = int(remote_custom_output.get("max_steps", 0)) - expected_paused_step_idx
        if remote_executed_step_count != expected_remote_steps:
            raise RuntimeError(
                f"Remote request executed unexpected number of steps for request_id={req0_id}: "
                f"expected {expected_remote_steps}, got {remote_executed_step_count}"
            )

        baseline_trace = _load_latent_trace(out_root / "baseline" / "req_1" / "latents.json")
        paused_trace = _load_latent_trace(out_root / "paused" / "req_0" / "latents.json")
        remote_trace = _load_latent_trace(out_root / "remote" / "req_0" / "latents.json")
        paused_divergence_step, paused_common_steps = _find_first_latent_divergence(paused_trace, remote_trace)
        if paused_divergence_step is not None:
            raise RuntimeError(
                f"Paused and remote latent traces diverged at step_idx={paused_divergence_step}; "
                f"common_steps={paused_common_steps}"
            )
        divergence_step, common_steps = _find_first_latent_divergence(baseline_trace, remote_trace)
        if divergence_step is not None:
            raise RuntimeError(
                f"Latent traces diverged at step_idx={divergence_step}; common_steps={common_steps}"
            )

        normalized_baseline = _normalize_custom_output(dict(baseline_payload))
        normalized_remote = _normalize_custom_output(dict(remote_payload))
        normalized_baseline.pop("request_id", None)
        normalized_baseline.pop("image_path", None)
        normalized_remote.pop("request_id", None)
        normalized_remote.pop("image_path", None)

        _assert_tensor_fields_match(
            normalized_baseline,
            normalized_remote,
            field_names=("all_latents", "all_timesteps", "rollout_log_probs"),
        )
        assert normalized_baseline == normalized_remote, "Migrated output does not match baseline output."

        baseline_img = Image.open(out_root / "baseline" / "req_1" / "output.png").convert("RGB")
        remote_img = Image.open(out_root / "remote" / "req_0" / "output.png").convert("RGB")
        assert np.array_equal(np.array(baseline_img), np.array(remote_img)), "Migrated image does not match baseline image."

        comparison = {
            "paused_request_id": req0_id,
            "baseline_request_id": req1_id,
            "remote_request_id": req0_id,
            "baseline_vs_remote_equal": True,
            "pause_step_idx": pause_step_idx,
            "paused_remote_common_steps": list(paused_common_steps),
            "latent_trace_common_steps": list(common_steps),
            "paused_payload": paused_payload,
            "baseline_payload": baseline_payload,
            "remote_payload": remote_payload,
            "paused_output_request_id": paused_output.request_id,
            "baseline_output_request_id": baseline_output.request_id,
            "remote_output_request_id": remote_output.request_id,
        }
        (out_root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=True), encoding="utf-8")
    except Exception:
        logger.exception("[MigrationTrace] Migration flow with pause_step_idx failed before cleanup")
        raise
    finally:
        logger.info("[MigrationTrace] Entering cleanup for migration pause_step_idx test")
        omni0.close()
        omni1.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_migration_with_pause_step_idx(tmp_path: Path):
    asyncio.run(_run_migration_with_pause_step_idx(tmp_path))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_migration_with_pause_step_idx_inside_sde_window(tmp_path: Path):
    asyncio.run(
        _run_migration_with_pause_step_idx(
            tmp_path,
            sampling_params=_make_sampling_params(sde_window_size=6, sde_window_range=(20, 30)),
            out_root_name="output/test_migration_pause_step_idx_inside_sde_window",
            config_suffix="pause_step_idx_inside_sde_window",
            pause_step_idx=25,
        )
    )


async def _run_migration_with_pause_step_idx_stress(tmp_path: Path) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 GPUs.")

    model = os.getenv("VLLM_OMNI_STEPWISE_TEST_MODEL")
    if not model:
        pytest.skip("Set VLLM_OMNI_STEPWISE_TEST_MODEL to a valid stepwise-capable model path/name.")

    tokenizer = _load_tokenizer_from_model(model)
    negative_prompt_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _NEG_PROMPT_TEXT},
    ]
    negative_prompt_ids, negative_prompt_mask = _chat_template_to_ids(tokenizer, negative_prompt_messages)
    prompt_variants = _make_prompt_variants(tokenizer)
    if len(prompt_variants) != _STRESS_TEST_NUM_PAIRS:
        raise RuntimeError(f"Expected {_STRESS_TEST_NUM_PAIRS} stress prompts, got {len(prompt_variants)}")

    sampling_params_template = _make_sampling_params()
    pause_step_indices = _make_pause_step_indices(
        num_inference_steps=int(sampling_params_template.num_inference_steps),
        num_pairs=_STRESS_TEST_NUM_PAIRS,
        seed=_STRESS_TEST_RANDOM_SEED,
    )

    out_root = Path("output/test_migration_pause_step_idx_stress")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    config0 = tmp_path / "stage_gpu0_pause_step_idx_stress.yaml"
    config1 = tmp_path / "stage_gpu1_pause_step_idx_stress.yaml"
    _write_single_stage_config(config0, 0)
    _write_single_stage_config(config1, 1)

    omni0, omni1 = await asyncio.gather(
        asyncio.to_thread(_create_async_omni, model=model, config_path=config0),
        asyncio.to_thread(_create_async_omni, model=model, config_path=config1),
    )

    failures: list[dict[str, Any]] = []
    try:
        for pair_index, pause_step_idx in enumerate(pause_step_indices):
            prompt_text, prompt_ids, prompt_mask = prompt_variants[pair_index]
            pair_dir = out_root / f"pair_{pair_index:03d}"
            pair_dir.mkdir(parents=True, exist_ok=True)

            baseline_request_id = f"migration-stress-{pair_index:03d}-baseline"
            pause_request_id = f"migration-stress-{pair_index:03d}-pause"

            logger.info(
                "[MigrationTrace] Starting stress pair pair_index=%03d pause_step_idx=%d prompt=%r baseline_request_id=%s pause_request_id=%s",
                pair_index,
                pause_step_idx,
                prompt_text,
                baseline_request_id,
                pause_request_id,
            )

            baseline_task = asyncio.create_task(
                _run_request(
                    omni0,
                    req_index=0,
                    request_id=baseline_request_id,
                    prompt_text=_PROMPT_TEXT,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    negative_prompt_ids=negative_prompt_ids,
                    negative_prompt_mask=negative_prompt_mask,
                    sampling_params=_make_sampling_params(),
                    out_root=pair_dir / "baseline",
                )
            )
            pause_task = asyncio.create_task(
                _run_request(
                    omni0,
                    req_index=1,
                    request_id=pause_request_id,
                    prompt_text=_PROMPT_TEXT,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    negative_prompt_ids=negative_prompt_ids,
                    negative_prompt_mask=negative_prompt_mask,
                    sampling_params=_make_sampling_params(),
                    out_root=pair_dir / "pause_migrate",
                    pause_step_idx=pause_step_idx,
                )
            )

            paused_output, paused_payload = await pause_task
            paused_custom_output = paused_payload.get("custom_output")
            if not isinstance(paused_custom_output, dict):
                raise RuntimeError(
                    f"Missing custom_output for paused request_id={pause_request_id}: {type(paused_custom_output).__name__}"
                )
            paused_finish_reason = paused_custom_output.get("finish_reason")
            if paused_finish_reason != "paused":
                raise RuntimeError(
                    f"Expected paused finish_reason for request_id={pause_request_id}, got {paused_finish_reason!r}"
                )
            expected_paused_step_idx = pause_step_idx + 1
            paused_step_idx = paused_custom_output.get("step_idx")
            if paused_step_idx != expected_paused_step_idx:
                raise RuntimeError(
                    f"Unexpected paused step_idx for request_id={pause_request_id}: "
                    f"expected {expected_paused_step_idx}, got {paused_step_idx}"
                )
            paused_executed_step_count = paused_custom_output.get("executed_step_count")
            if paused_executed_step_count != expected_paused_step_idx:
                raise RuntimeError(
                    f"Unexpected paused executed_step_count for request_id={pause_request_id}: "
                    f"expected {expected_paused_step_idx}, got {paused_executed_step_count}"
                )

            remote_task = asyncio.create_task(
                _run_request(
                    omni1,
                    req_index=0,
                    request_id=pause_request_id,
                    prompt_text=_PROMPT_TEXT,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    negative_prompt_ids=negative_prompt_ids,
                    negative_prompt_mask=negative_prompt_mask,
                    sampling_params=_make_sampling_params(),
                    out_root=pair_dir / "pause_migrate_remote",
                    is_remote=True,
                )
            )

            baseline_output, baseline_payload = await baseline_task
            remote_output, remote_payload = await remote_task

            remote_custom_output = remote_output.custom_output or {}
            remote_resume_from_step_idx = remote_custom_output.get("resume_from_step_idx")
            if remote_resume_from_step_idx != expected_paused_step_idx:
                raise RuntimeError(
                    f"Remote request resumed from unexpected step_idx for request_id={pause_request_id}: "
                    f"expected {expected_paused_step_idx}, got {remote_resume_from_step_idx}"
                )
            remote_executed_step_count = remote_custom_output.get("executed_step_count")
            expected_remote_steps = int(remote_custom_output.get("max_steps", 0)) - expected_paused_step_idx
            if remote_executed_step_count != expected_remote_steps:
                raise RuntimeError(
                    f"Remote request executed unexpected number of steps for request_id={pause_request_id}: "
                    f"expected {expected_remote_steps}, got {remote_executed_step_count}"
                )

            baseline_trace = _load_latent_trace(pair_dir / "baseline" / "req_0" / "latents.json")
            remote_trace = _load_latent_trace(pair_dir / "pause_migrate_remote" / "req_0" / "latents.json")
            divergence_step, common_steps = _find_first_latent_divergence(baseline_trace, remote_trace)

            _, baseline_image = _extract_output_tensor_and_image(baseline_output)
            _, remote_image = _extract_output_tensor_and_image(remote_output)

            baseline_image_path = pair_dir / "baseline" / "req_0" / "output.png"
            remote_image_path = pair_dir / "pause_migrate_remote" / "req_0" / "output.png"
            image_diff = _compare_images(baseline_image, remote_image)

            normalized_baseline = _normalize_comparison_payload(baseline_payload)
            normalized_remote = _normalize_comparison_payload(remote_payload)
            payload_diffs = _collect_differences(normalized_baseline, normalized_remote)

            pair_mismatch_details: list[str] = []
            if divergence_step is not None:
                pair_mismatch_details.append(
                    f"latent_trace diverged at step_idx={divergence_step}; common_steps={common_steps}"
                )
            if payload_diffs:
                pair_mismatch_details.append("payload diff:\n" + "\n".join(f"  - {d}" for d in payload_diffs))
            if image_diff is not None:
                pair_mismatch_details.append(image_diff)

            _write_pair_comparison(
                pair_dir,
                pair_index=pair_index,
                pause_step_idx=pause_step_idx,
                baseline_payload_summary=normalized_baseline,
                pause_migrate_payload_summary=normalized_remote,
                baseline_image_path=baseline_image_path,
                pause_migrate_image_path=remote_image_path,
                mismatch_details=pair_mismatch_details or None,
            )

            if pair_mismatch_details:
                logger.error(
                    "[MigrationTrace] Stress pair mismatch pair_index=%03d pause_step_idx=%d request_id=%s",
                    pair_index,
                    pause_step_idx,
                    pause_request_id,
                )
                for detail in pair_mismatch_details:
                    logger.error("[MigrationTrace]   %s", detail)
                failures.append(
                    {
                        "pair_index": pair_index,
                        "pause_step_idx": pause_step_idx,
                        "request_id": pause_request_id,
                        "details": pair_mismatch_details,
                    }
                )
            else:
                logger.info(
                    "[MigrationTrace] Stress pair matched pair_index=%03d pause_step_idx=%d request_id=%s",
                    pair_index,
                    pause_step_idx,
                    pause_request_id,
                )

            if pair_index < len(pause_step_indices) - 1:
                await asyncio.sleep(_STRESS_TEST_PAIR_INTERVAL_S)

        summary = {
            "total_pairs": len(pause_step_indices),
            "matched_pairs": len(pause_step_indices) - len(failures),
            "failed_pairs": failures,
            "pause_step_indices": pause_step_indices,
        }
        (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=True), encoding="utf-8")
        if failures:
            raise AssertionError(
                f"{len(failures)} stress pairs produced mismatched outputs; see output/test_migration_pause_step_idx_stress/summary.json"
            )
    except Exception:
        logger.exception("[MigrationTrace] Migration stress flow with pause_step_idx failed before cleanup")
        raise
    finally:
        logger.info("[MigrationTrace] Entering cleanup for migration pause_step_idx stress test")
        omni0.close()
        omni1.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.slow
def test_migration_with_pause_step_idx_stress(tmp_path: Path):
    asyncio.run(_run_migration_with_pause_step_idx_stress(tmp_path))
