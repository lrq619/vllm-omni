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

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.diffusion]

_REQUEST_NUM = 8
_REQUEST_INTERVAL_S = 2.0
_STEP_DIR = Path("output/step_results")
_NON_STEP_DIR = Path("output/non_step_results")
_SYSTEM_PROMPT_PREFIX = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background: happy happy happy happy"
)
_SIMPLE_PROMPTS = [
    "A red apple on a wooden table.",
    "A blue bird flying over a lake.",
    "A small house in a green field.",
    "A white cat sitting by a window.",
    "A mountain at sunrise with soft light.",
    "A yellow flower in the garden.",
    "A city street after light rain.",
    "A cup of coffee on a desk.",
]
_PROMPTS = [f"{_SYSTEM_PROMPT_PREFIX} {p}" for p in _SIMPLE_PROMPTS]


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


def _create_async_omni(model: str, enable_stepwise: bool) -> AsyncOmni:
    if enable_stepwise:
        return AsyncOmni(
            model=model,
            enable_stepwise=enable_stepwise,
            num_gpus=1,
            diffusion_load_format="custom_pipeline",
            custom_pipeline_args={
                "pipeline_class": "vllm_omni.diffusion.models.qwen_image.pipeline_qwenimage_step."
                "QwenImagePipelineWithLogProbStep"
            },
        )
    else:
        return AsyncOmni(
            model=model,
            num_gpus=1,
        )



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
    prompt_mask: list[int],
    output: OmniRequestOutput,
) -> None:
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
        "prompt_mask": prompt_mask,
        "image_path": str(image_path),
        "output_tensor": _serialize_tensor(output_tensor),
        "custom_output": _serialize_obj(custom_output),
    }
    (req_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


async def _run_workload_and_dump(enable_stepwise: bool, out_root: Path) -> None:
    model = os.getenv("VLLM_OMNI_STEPWISE_TEST_MODEL")
    if not model:
        pytest.skip("Set VLLM_OMNI_STEPWISE_TEST_MODEL to a valid stepwise-capable model path/name.")

    tokenizer = _load_tokenizer_from_model(model)
    prompts = _PROMPTS
    if len(prompts) != _REQUEST_NUM:
        raise RuntimeError(f"Expected {_REQUEST_NUM} prompts, got {len(prompts)}")

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    omni = _create_async_omni(model=model, enable_stepwise=enable_stepwise)
    try:
        tasks = []
        for i, text in enumerate(prompts):
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            prompt_ids = [int(x) for x in token_ids]
            prompt_mask = [1] * len(prompt_ids)
            sampling_params = OmniDiffusionSamplingParams(
                num_inference_steps=20,
                guidance_scale=1.0,
                true_cfg_scale=1.0,
                width=256,
                height=256,
                output_type="pil",
                seed=1000 + i,
                extra_args={"noise_level": 0.7, "sde_type": "sde", "logprobs": True},
            )
            request_id = f"gpu-async-{'step' if enable_stepwise else 'non-step'}-{i:02d}"

            async def _one_request(
                req_index: int,
                prompt_text: str,
                req_id: str,
                req_prompt_ids: list[int],
                req_prompt_mask: list[int],
                sp: OmniDiffusionSamplingParams,
            ) -> None:
                final_output: OmniRequestOutput | None = None
                async for out in omni.generate(
                    prompt={"prompt_ids": req_prompt_ids, "prompt_mask": req_prompt_mask},
                    request_id=req_id,
                    sampling_params_list=[sp],
                ):
                    final_output = out
                if final_output is None:
                    raise RuntimeError(f"No final output for request_id={req_id}")
                _write_request_result(
                    out_root=out_root,
                    req_index=req_index,
                    prompt_text=prompt_text,
                    prompt_ids=req_prompt_ids,
                    prompt_mask=req_prompt_mask,
                    output=final_output,
                )

            tasks.append(
                asyncio.create_task(_one_request(i, text, request_id, prompt_ids, prompt_mask, sampling_params))
            )
            await asyncio.sleep(_REQUEST_INTERVAL_S)

        await asyncio.gather(*tasks)
    finally:
        omni.close()


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    name = dtype_str.split(".")[-1]
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise RuntimeError(f"Unsupported dtype string: {dtype_str}")
    return dtype


def _collect_tensors(obj: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    if isinstance(obj, dict):
        if obj.get("__type__") == "tensor":
            tensors[prefix or "root"] = obj
            return tensors
        for k, v in obj.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            tensors.update(_collect_tensors(v, next_prefix))
        return tensors
    if isinstance(obj, list):
        for idx, v in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            tensors.update(_collect_tensors(v, next_prefix))
    return tensors


def _tensor_from_payload(payload: dict[str, Any]) -> torch.Tensor:
    dtype = _dtype_from_str(payload["dtype"])
    shape = payload["shape"]
    data = payload["data"]
    t = torch.tensor(data, dtype=dtype)
    return t.reshape(shape)


def _compare_tensor_payloads(a: dict[str, Any], b: dict[str, Any], tensor_name: str) -> None:
    ta = _tensor_from_payload(a)
    tb = _tensor_from_payload(b)
    assert list(ta.shape) == list(tb.shape), f"Tensor shape mismatch for {tensor_name}: {ta.shape} vs {tb.shape}"

    if ta.is_floating_point() or tb.is_floating_point():
        ta_f = ta.float()
        tb_f = tb.float()
        close = torch.allclose(ta_f, tb_f, rtol=1e-3, atol=1e-3)
        if not close:
            max_abs_diff = float((ta_f - tb_f).abs().max().item())
            raise AssertionError(
                f"Floating tensor mismatch for {tensor_name}, max_abs_diff={max_abs_diff}, "
                "allowed rtol=1e-3 atol=1e-3"
            )
    else:
        assert torch.equal(ta, tb), f"Non-floating tensor mismatch for {tensor_name}"


def _ensure_results_exist() -> None:
    step_ok = all((_STEP_DIR / f"req_{i}" / "result.json").exists() for i in range(_REQUEST_NUM))
    non_step_ok = all((_NON_STEP_DIR / f"req_{i}" / "result.json").exists() for i in range(_REQUEST_NUM))
    if not step_ok:
        asyncio.run(_run_workload_and_dump(enable_stepwise=True, out_root=_STEP_DIR))
    if not non_step_ok:
        asyncio.run(_run_workload_and_dump(enable_stepwise=False, out_root=_NON_STEP_DIR))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_stepwise():
    asyncio.run(_run_workload_and_dump(enable_stepwise=True, out_root=_STEP_DIR))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_non_stepwise():
    asyncio.run(_run_workload_and_dump(enable_stepwise=False, out_root=_NON_STEP_DIR))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_cmp():
    _ensure_results_exist()

    for i in range(_REQUEST_NUM):
        step_json = json.loads((_STEP_DIR / f"req_{i}" / "result.json").read_text(encoding="utf-8"))
        non_step_json = json.loads((_NON_STEP_DIR / f"req_{i}" / "result.json").read_text(encoding="utf-8"))

        step_tensors = _collect_tensors(step_json)
        non_step_tensors = _collect_tensors(non_step_json)

        assert set(step_tensors.keys()) == set(
            non_step_tensors.keys()
        ), f"Tensor key sets mismatch for req_{i}: {set(step_tensors.keys()) ^ set(non_step_tensors.keys())}"

        for key in sorted(step_tensors.keys()):
            _compare_tensor_payloads(step_tensors[key], non_step_tensors[key], tensor_name=f"req_{i}:{key}")
