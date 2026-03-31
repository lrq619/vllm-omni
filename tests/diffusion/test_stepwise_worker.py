# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.tensor_pool import TensorPoolManager
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.stepwise_worker import DiffusionStepwiseWorker, WorkerRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


class _FakeScheduler:
    def __init__(self) -> None:
        self.begin_index: int | None = None

    def set_begin_index(self, begin_index: int) -> None:
        self.begin_index = begin_index


class _TrackingManager:
    def __init__(self, max_bsz: int = 4) -> None:
        self.inner = TensorPoolManager(max_bsz=max_bsz)
        self.alloc_calls: list[int] = []
        self.reserve_calls: list[list[int]] = []
        self.release_calls: list[list[int]] = []
        self.add_calls: list[dict[str, object]] = []

    @property
    def pools(self):
        return self.inner.pools

    def add(self, *, name: str, shape, dtype, device) -> None:
        self.add_calls.append({"name": name, "shape": tuple(shape), "dtype": dtype, "device": device})
        self.inner.add(name=name, shape=shape, dtype=dtype, device=device)

    def alloc(self, num_rows: int) -> list[int]:
        self.alloc_calls.append(num_rows)
        return self.inner.alloc(num_rows)

    def reserve(self, indicies: list[int]) -> None:
        self.reserve_calls.append(list(indicies))
        self.inner.reserve(indicies)

    def release(self, indicies: list[int]) -> None:
        self.release_calls.append(list(indicies))
        self.inner.release(indicies)

    def get(self, name: str, indicies: list[int]) -> torch.Tensor:
        return self.inner.get(name, indicies)

    def put(self, name: str, indicies: list[int], tensor: torch.Tensor) -> None:
        self.inner.put(name, indicies, tensor)


class _FakePipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.transformer = type("Transformer", (), {"guidance_embeds": False, "in_channels": 4})()
        self.scheduler = _FakeScheduler()

    def prepare_timesteps(self, num_inference_steps: int, sigmas, latent_dim: int):
        del sigmas, latent_dim
        timesteps = torch.arange(float(num_inference_steps), 0.0, -1.0, dtype=torch.float32)
        return timesteps, None


def _build_request(request_id: str, *, remote: bool) -> OmniDiffusionRequest:
    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=4,
        guidance_scale=1.0,
        true_cfg_scale=1.0,
        output_type="pil",
        extra_args={"noise_level": 0.7, "sde_type": "sde", "logprobs": True},
    )
    prompt = {"prompt_ids": [1, 2], "prompt_mask": [1, 1]}
    return OmniDiffusionRequest(
        prompts=[prompt],
        sampling_params=sampling_params,
        request_ids=[request_id],
        is_remote_list=[remote],
    )


def test_build_pause_state_payload_excludes_row_index():
    worker = DiffusionStepwiseWorker.__new__(DiffusionStepwiseWorker)
    request = _build_request("pause-req", remote=False)
    state = WorkerRequestState(
        request_id="pause-req",
        row_index=7,
        timesteps=[4.0, 3.0, 2.0, 1.0],
        step_idx=2,
        max_steps=4,
        pause_step_idx=1,
        do_true_cfg=False,
        true_cfg_scale=1.0,
        noise_level=0.7,
        sde_window=(0, 3),
        sde_type="sde",
        logprobs=True,
        guidance_scale=1.0,
        output_type="pil",
        height=64,
        width=64,
        img_shapes=[(1, 8, 8)],
        txt_seq_len=2,
        negative_txt_seq_len=None,
        generator=None,
        request=request,
        scheduler=_FakeScheduler(),
        collected_latents=[torch.zeros((1, 4, 8, 8))],
        collected_log_probs=[None],
        collected_timesteps=[torch.tensor(4.0)],
        resume_from_step_idx=1,
        latent_trace={"1": {"shape": [1, 4, 8, 8]}},
    )

    payload = worker._build_pause_state_payload(state)

    assert "row_index" not in payload
    assert payload["request_id"] == "pause-req"
    assert payload["step_idx"] == 2


def test_remote_stepwise_admit_allocates_fresh_row_without_reserve(monkeypatch):
    worker = DiffusionStepwiseWorker.__new__(DiffusionStepwiseWorker)
    worker._states = {}
    worker._pool_specs = {}

    manager = _TrackingManager(max_bsz=4)
    pipeline = _FakePipeline()

    remote_state = {
        "request_id": "remote-req",
        "prompt": {
            "prompt_ids": [11, 22],
            "prompt_mask": [1, 1],
        },
        "step_idx": 2,
        "max_steps": 4,
        "timesteps": [4.0, 3.0, 2.0, 1.0],
        "do_true_cfg": False,
        "true_cfg_scale": 1.0,
        "noise_level": 0.7,
        "sde_window": (0, 3),
        "sde_type": "sde",
        "logprobs": True,
        "guidance_scale": 1.0,
        "output_type": "pil",
        "height": 32,
        "width": 32,
        "img_shapes": [(1, 4, 4)],
        "txt_seq_len": 2,
        "negative_txt_seq_len": None,
        "sampling_params": {"num_inference_steps": 4, "sigmas": None},
        "tensor_pool_names": [
            "latents",
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
        ],
    }
    loaded_tensors = {
        "latents": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
        "prompt_embeds": torch.zeros((1, 6, 8), dtype=torch.float32),
        "prompt_embeds_mask": torch.ones((1, 6), dtype=torch.float32),
        "negative_prompt_embeds": torch.zeros((1, 6, 8), dtype=torch.float32),
        "negative_prompt_embeds_mask": torch.zeros((1, 6), dtype=torch.float32),
    }

    monkeypatch.setattr(worker, "_ensure_runtime", lambda: manager)
    monkeypatch.setattr(worker, "_pipeline", lambda: pipeline)
    monkeypatch.setattr(worker, "_load_remote_prompt_payload", lambda request_id: (copy.deepcopy(remote_state), loaded_tensors))

    request = _build_request("remote-req", remote=True)
    result = worker.stepwise_admit_request("remote-req", request)

    assert manager.alloc_calls == [1]
    assert manager.reserve_calls == []
    assert result.admitted is True
    assert result.row_indices == [0]
    assert result.step_idx == 2
    assert result.current_timestep == pytest.approx(2.0)
