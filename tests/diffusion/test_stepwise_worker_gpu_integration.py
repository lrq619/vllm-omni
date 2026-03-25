# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion]


def _to_image_tensor(t: torch.Tensor) -> torch.Tensor:
    # Normalize to [3, H, W] float tensor in [0, 1].
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


def _save_tensor_json(path: Path, t: torch.Tensor) -> None:
    data = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "data": t.detach().cpu().tolist(),
    }
    path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")


def _create_stepwise_omni(model: str) -> OmniDiffusion:
    return OmniDiffusion(
        model=model,
        enable_stepwise=True,
        num_gpus=1,
        diffusion_load_format="custom_pipeline",
        custom_pipeline_args={
            "pipeline_class": "vllm_omni.diffusion.models.qwen_image.pipeline_qwenimage_step."
            "QwenImagePipelineWithLogProbStep"
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_stepwise_worker_metric_and_output_dump_gpu():
    model = os.getenv("VLLM_OMNI_STEPWISE_TEST_MODEL")
    if not model:
        pytest.skip("Set VLLM_OMNI_STEPWISE_TEST_MODEL to a valid stepwise-capable model path/name.")

    out_dir = Path("output/stepwise_worker_gpu_integration")
    out_dir.mkdir(parents=True, exist_ok=True)

    omni = _create_stepwise_omni(model)
    executor = omni.engine.executor
    try:
        # Keep resolution moderate so JSON dump stays practical.
        sampling_params = OmniDiffusionSamplingParams(
            num_inference_steps=10,
            guidance_scale=1.0,
            true_cfg_scale=1.0,
            width=256,
            height=256,
            output_type="pil",
            extra_args={"noise_level": 0.7, "sde_type": "sde", "logprobs": True},
        )
        prompt_ids = list(range(80))
        prompt_mask = [1] * len(prompt_ids)
        request = OmniDiffusionRequest(
            prompts=[{"prompt_ids": prompt_ids, "prompt_mask": prompt_mask}],
            sampling_params=sampling_params,
            request_ids=["gpu-stepwise-10-step"],
        )

        future = executor.add_req(request)
        output = future.result(timeout=1800)

        metric = executor.collective_rpc(
            method="export_metric",
            unique_reply_rank=0,
        )
        metric_json = metric.dump_json()
        assert metric_json["summary"]["total_plans"] == 10
        assert metric_json["summary"]["max_batch_size"] == 1
        assert len(metric_json["plans"]) == 10
        assert all(int(m["batch_size"]) == 1 for m in metric_json["plans"])

        out_tensor = output.output
        if not isinstance(out_tensor, torch.Tensor):
            raise RuntimeError(f"Expected torch.Tensor output, got {type(out_tensor).__name__}")

        img_tensor = _to_image_tensor(out_tensor)
        image_uint8 = (img_tensor * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(image_uint8).save(out_dir / "scheduler_output.png")

        _save_tensor_json(out_dir / "scheduler_output_tensor.json", out_tensor)
        (out_dir / "stepwise_metric.json").write_text(json.dumps(metric_json, ensure_ascii=True), encoding="utf-8")
        (out_dir / "stepwise_metric.txt").write_text(metric.dump_str(), encoding="utf-8")
    finally:
        omni.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_stepwise_worker_staggered_32_requests_gpu():
    model = os.getenv("VLLM_OMNI_STEPWISE_TEST_MODEL")
    if not model:
        pytest.skip("Set VLLM_OMNI_STEPWISE_TEST_MODEL to a valid stepwise-capable model path/name.")

    out_dir = Path("output/stepwise_worker_gpu_integration/32_requests")
    out_dir.mkdir(parents=True, exist_ok=True)

    omni = _create_stepwise_omni(model)
    executor = omni.engine.executor
    try:
        sampling_params = OmniDiffusionSamplingParams(
            num_inference_steps=20,
            guidance_scale=1.0,
            true_cfg_scale=1.0,
            width=256,
            height=256,
            output_type="pil",
            extra_args={"noise_level": 0.7, "sde_type": "sde", "logprobs": True},
        )
        prompt_ids = list(range(80))
        prompt_mask = [1] * len(prompt_ids)

        futures = []
        request_num = 8
        for i in range(request_num):
            request = OmniDiffusionRequest(
                prompts=[{"prompt_ids": prompt_ids, "prompt_mask": prompt_mask}],
                sampling_params=sampling_params,
                request_ids=[f"gpu-stepwise-staggered-{i:02d}"],
            )
            futures.append(executor.add_req(request))
            time.sleep(2)

        outputs = [future.result(timeout=3600) for future in futures]
        assert len(outputs) == request_num
        assert all(isinstance(output.output, torch.Tensor) for output in outputs)

        metric = executor.collective_rpc(
            method="export_metric",
            unique_reply_rank=0,
        )
        metric_json = metric.dump_json()
        assert metric_json["summary"]["total_plans"] >= 5
        assert metric_json["summary"]["max_batch_size"] >= 1

        (out_dir / "stepwise_metric.json").write_text(json.dumps(metric_json, ensure_ascii=True), encoding="utf-8")
        (out_dir / "stepwise_metric.txt").write_text(metric.dump_str(), encoding="utf-8")
    finally:
        omni.close()
