# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import hashlib
import pickle
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwenimage_step import (
    QwenImagePipelineWithLogProbStep,
)
from vllm_omni.diffusion.mooncake_store import MooncakeStore
from vllm_omni.diffusion.models.qwen_image.tensor_pool import TensorPoolManager
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.stepwise_scheduler import (
    AdmissionResult,
    SchedulerBatchPlan,
    StepExecutionResult,
)
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

logger = init_logger(__name__)

# REMOTE_PROMPT_TENSOR_NAMES = (
#     "prompt_ids",
#     "prompt_mask",
#     "negative_prompt_ids",
#     "negative_prompt_mask",
# )
PAUSE_STATE_TENSOR_FIELDS = (
    "generator_state",
    "collected_latents",
    "collected_log_probs",
    "collected_timesteps",
)


def _cuda_mem_mb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    return float(allocated), float(reserved)


@dataclass
class WorkerRequestState:
    request_id: str
    row_index: int
    timesteps: list[float]
    step_idx: int
    max_steps: int
    pause_step_idx: int | None
    do_true_cfg: bool
    true_cfg_scale: float
    noise_level: float
    sde_window: tuple[int, int]
    sde_type: str
    logprobs: bool
    guidance_scale: float
    output_type: str
    height: int
    width: int
    img_shapes: list[tuple[int, int, int]]
    txt_seq_len: int
    negative_txt_seq_len: int | None
    generator: torch.Generator | list[torch.Generator] | None
    request: OmniDiffusionRequest
    scheduler: object
    collected_latents: list[torch.Tensor] = field(default_factory=list)
    collected_log_probs: list[torch.Tensor | None] = field(default_factory=list)
    collected_timesteps: list[torch.Tensor] = field(default_factory=list)
    resume_from_step_idx: int = 0
    latent_trace: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class StepwisePlanMetric:
    metric_id: int
    loop_end_timestamp_ms: float
    plan_id: str
    batch_size: int
    request_ids: list[str]
    row_indices: list[int]
    step_indices: list[int]
    finished_count: int
    latency_ms: float
    cuda_allocated_mb: float
    cuda_reserved_mb: float


class StepwiseWorkerMetric:
    def __init__(self) -> None:
        self._plan_metrics: list[StepwisePlanMetric] = []

    def reset(self) -> None:
        self._plan_metrics.clear()

    def record_plan(
        self,
        *,
        metric_id: int,
        loop_end_timestamp_ms: float,
        plan_id: str,
        request_ids: list[str],
        row_indices: list[int],
        step_indices: list[int],
        finished_count: int,
        latency_ms: float,
        cuda_allocated_mb: float,
        cuda_reserved_mb: float,
    ) -> dict[str, Any]:
        metric = StepwisePlanMetric(
            metric_id=int(metric_id),
            loop_end_timestamp_ms=float(loop_end_timestamp_ms),
            plan_id=plan_id,
            batch_size=len(request_ids),
            request_ids=list(request_ids),
            row_indices=list(row_indices),
            step_indices=list(step_indices),
            finished_count=int(finished_count),
            latency_ms=float(latency_ms),
            cuda_allocated_mb=float(cuda_allocated_mb),
            cuda_reserved_mb=float(cuda_reserved_mb),
        )
        self._plan_metrics.append(metric)
        return asdict(metric)

    def dump_json(self) -> dict[str, Any]:
        total_plans = len(self._plan_metrics)
        if total_plans == 0:
            return {
                "summary": {
                    "total_plans": 0,
                    "avg_batch_size": 0.0,
                    "max_batch_size": 0,
                    "avg_latency_ms": 0.0,
                    "latest_cuda_allocated_mb": 0.0,
                    "latest_cuda_reserved_mb": 0.0,
                },
                "plans": [],
            }
        batch_sizes = [m.batch_size for m in self._plan_metrics]
        latencies = [m.latency_ms for m in self._plan_metrics]
        latest = self._plan_metrics[-1]
        return {
            "summary": {
                "total_plans": total_plans,
                "avg_batch_size": float(sum(batch_sizes) / total_plans),
                "max_batch_size": int(max(batch_sizes)),
                "avg_latency_ms": float(sum(latencies) / total_plans),
                "latest_cuda_allocated_mb": float(latest.cuda_allocated_mb),
                "latest_cuda_reserved_mb": float(latest.cuda_reserved_mb),
            },
            "plans": [asdict(m) for m in self._plan_metrics],
        }

    def dump_str(self) -> str:
        payload = self.dump_json()
        summary = payload["summary"]
        return (
            "StepwiseWorkerMetric("
            f"total_plans={summary['total_plans']}, "
            f"avg_batch_size={summary['avg_batch_size']:.2f}, "
            f"max_batch_size={summary['max_batch_size']}, "
            f"avg_latency_ms={summary['avg_latency_ms']:.3f}, "
            f"latest_cuda_allocated_mb={summary['latest_cuda_allocated_mb']:.2f}, "
            f"latest_cuda_reserved_mb={summary['latest_cuda_reserved_mb']:.2f})"
        )


def _maybe_to_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    return v


def _serialize_generator_state(
    generator: torch.Generator | list[torch.Generator] | None,
) -> torch.Tensor | list[torch.Tensor] | None:
    if generator is None:
        return None
    if isinstance(generator, torch.Generator):
        return generator.get_state().detach().cpu()
    if isinstance(generator, list):
        return [g.get_state().detach().cpu() for g in generator]
    raise TypeError(f"Unsupported generator type: {type(generator).__name__}")


def _restore_generator_state(
    generator_state: torch.Tensor | list[torch.Tensor] | None,
    device: torch.device,
) -> torch.Generator | list[torch.Generator] | None:
    if generator_state is None:
        return None
    if isinstance(generator_state, torch.Tensor):
        generator = torch.Generator(device=device)
        generator.set_state(generator_state.cpu())
        return generator
    if isinstance(generator_state, list):
        generators: list[torch.Generator] = []
        for state in generator_state:
            generator = torch.Generator(device=device)
            generator.set_state(state.cpu())
            generators.append(generator)
        return generators
    raise TypeError(f"Unsupported generator state type: {type(generator_state).__name__}")


def _restore_tensor_list(
    values: list[Any] | tuple[Any, ...] | None,
    device: torch.device,
) -> list[torch.Tensor | None]:
    restored: list[torch.Tensor | None] = []
    if values is None:
        return restored
    for value in values:
        if value is None:
            restored.append(None)
        elif isinstance(value, torch.Tensor):
            restored.append(value.to(device))
        else:
            raise TypeError(f"Unsupported tensor list entry type: {type(value).__name__}")
    return restored


def _serialize_sampling_params(sampling_params: Any) -> dict[str, Any]:
    # Store the schedule-related request params needed to rebuild timesteps on resume.
    return {
        "num_inference_steps": int(getattr(sampling_params, "num_inference_steps", 0)),
        "sigmas": copy.deepcopy(getattr(sampling_params, "sigmas", None)),
        "seed": getattr(sampling_params, "seed", None),
        "guidance_scale": getattr(sampling_params, "guidance_scale", None),
        "guidance_scale_2": getattr(sampling_params, "guidance_scale_2", None),
        "guidance_scale_provided": getattr(sampling_params, "guidance_scale_provided", None),
        "true_cfg_scale": getattr(sampling_params, "true_cfg_scale", None),
        "height": getattr(sampling_params, "height", None),
        "width": getattr(sampling_params, "width", None),
        "output_type": getattr(sampling_params, "output_type", None),
        "extra_args": copy.deepcopy(getattr(sampling_params, "extra_args", {})),
    }


def _pause_state_tensor_key(request_id: str, field_name: str, index: int | None = None) -> str:
    if index is None:
        return f"{request_id}::state::{field_name}"
    return f"{request_id}::state::{field_name}::{index}"


def _string_list(value: Any, *, request_id: str, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(
            f"Remote state payload for request_id={request_id} must contain list {field_name}."
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RuntimeError(
                f"Remote state payload for request_id={request_id} must contain string entries in {field_name}."
            )
        result.append(item)
    return result


def _optional_string_list(value: Any, *, request_id: str, field_name: str) -> list[str | None]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(
            f"Remote state payload for request_id={request_id} must contain list {field_name}."
        )
    result: list[str | None] = []
    for item in value:
        if item is None:
            result.append(None)
            continue
        if not isinstance(item, str):
            raise RuntimeError(
                f"Remote state payload for request_id={request_id} must contain string or None entries in {field_name}."
            )
        result.append(item)
    return result


def _key_list_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"count": 0, "keys": []}
    if not isinstance(value, list):
        return {"type": type(value).__name__}
    return {"count": len(value), "keys": list(value)}


def _tensor_summary(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return {
        "type": type(value).__name__,
    }


def _latent_trace_summary(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    cpu_float = cpu_tensor.float()
    flat = cpu_tensor.reshape(-1)
    return {
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "sha256": hashlib.sha256(cpu_float.numpy().tobytes()).hexdigest(),
        "mean": float(cpu_float.mean().item()) if cpu_tensor.numel() > 0 else 0.0,
        "std": float(cpu_float.std(unbiased=False).item()) if cpu_tensor.numel() > 0 else 0.0,
        "min": float(cpu_float.min().item()) if cpu_tensor.numel() > 0 else 0.0,
        "max": float(cpu_float.max().item()) if cpu_tensor.numel() > 0 else 0.0,
        "sample": [float(x) for x in flat[:8].tolist()],
    }


def _summarize_pause_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": payload.get("request_id"),
        "row_index": payload.get("row_index"),
        "step_idx": payload.get("step_idx"),
        "max_steps": payload.get("max_steps"),
        "pause_step_idx": payload.get("pause_step_idx"),
        "executed_step_count": payload.get("executed_step_count"),
        "current_timestep": payload.get("current_timestep"),
        "timesteps_len": len(payload.get("timesteps", [])),
        "do_true_cfg": payload.get("do_true_cfg"),
        "true_cfg_scale": payload.get("true_cfg_scale"),
        "noise_level": payload.get("noise_level"),
        "sde_window": payload.get("sde_window"),
        "sde_type": payload.get("sde_type"),
        "logprobs": payload.get("logprobs"),
        "guidance_scale": payload.get("guidance_scale"),
        "output_type": payload.get("output_type"),
        "height": payload.get("height"),
        "width": payload.get("width"),
        "txt_seq_len": payload.get("txt_seq_len"),
        "negative_txt_seq_len": payload.get("negative_txt_seq_len"),
        "collected_latents_count": payload.get("collected_latents_count"),
        "collected_log_probs_count": payload.get("collected_log_probs_count"),
        "collected_timesteps_count": payload.get("collected_timesteps_count"),
        "tensor_names": list(payload.get("tensor_names", [])),
        "tensor_pool_names": list(payload.get("tensor_pool_names", [])),
        "generator_state_is_list": payload.get("generator_state_is_list"),
        "generator_state_keys": _key_list_summary(payload.get("generator_state_keys")),
        "collected_latents_keys": _key_list_summary(payload.get("collected_latents_keys")),
        "collected_log_probs_keys": _key_list_summary(payload.get("collected_log_probs_keys")),
        "collected_timesteps_keys": _key_list_summary(payload.get("collected_timesteps_keys")),
    }


def _summarize_remote_state_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": state.get("request_id"),
        "row_index": state.get("row_index"),
        "step_idx": state.get("step_idx"),
        "max_steps": state.get("max_steps"),
        "pause_step_idx": state.get("pause_step_idx"),
        "executed_step_count": state.get("executed_step_count"),
        "current_timestep": state.get("current_timestep"),
        "timesteps_len": len(state.get("timesteps", [])),
        "do_true_cfg": state.get("do_true_cfg"),
        "true_cfg_scale": state.get("true_cfg_scale"),
        "noise_level": state.get("noise_level"),
        "sde_window": state.get("sde_window"),
        "sde_type": state.get("sde_type"),
        "logprobs": state.get("logprobs"),
        "guidance_scale": state.get("guidance_scale"),
        "output_type": state.get("output_type"),
        "height": state.get("height"),
        "width": state.get("width"),
        "txt_seq_len": state.get("txt_seq_len"),
        "negative_txt_seq_len": state.get("negative_txt_seq_len"),
        "collected_latents_count": len(state.get("collected_latents", [])),
        "collected_log_probs_count": len(state.get("collected_log_probs", [])),
        "collected_timesteps_count": len(state.get("collected_timesteps", [])),
        "tensor_names": list(state.get("tensor_names", [])),
        "tensor_pool_names": list(state.get("tensor_pool_names", [])),
        "generator_state_is_list": state.get("generator_state_is_list"),
        "generator_state_keys": _key_list_summary(state.get("generator_state_keys")),
        "collected_latents_keys": _key_list_summary(state.get("collected_latents_keys")),
        "collected_log_probs_keys": _key_list_summary(state.get("collected_log_probs_keys")),
        "collected_timesteps_keys": _key_list_summary(state.get("collected_timesteps_keys")),
    }


class DiffusionStepwiseWorker(DiffusionWorker):
    """
    Worker-side runtime for one-step denoise execution.

    The scheduler drives lifecycle and sends SchedulerBatchPlan.
    This worker owns only tensor content and deterministic step execution.
    """

    def __init__(
        self,
        local_rank: int,
        rank: int,
        od_config: OmniDiffusionConfig,
        skip_load_model: bool = False,
    ) -> None:
        super().__init__(
            local_rank=local_rank,
            rank=rank,
            od_config=od_config,
            skip_load_model=skip_load_model,
        )
        self._manager: TensorPoolManager | None = None
        self._states: dict[str, WorkerRequestState] = {}
        self._max_bsz: int | None = None
        self._pool_specs: dict[str, tuple[tuple[int, ...], torch.dtype, torch.device]] = {}
        self._metrics = StepwiseWorkerMetric()
        self._non_empty_loop_counter: int = 0
        self._mooncake_store: MooncakeStore | None = None
        self._pause_commit_queue: deque[list[str]] = deque()
        if od_config.enable_stepwise:
            if MooncakeStore is None:
                raise ImportError("MooncakeStore is required when enable_stepwise=True.")
            self._mooncake_store = MooncakeStore()
            self._mooncake_store.initialize()

    def _pipeline(self) -> QwenImagePipelineWithLogProbStep:
        assert self.model_runner is not None, "Model runner not initialized"
        pipeline = self.model_runner.pipeline
        if not isinstance(pipeline, QwenImagePipelineWithLogProbStep):
            logger.error(
                "Stepwise worker requires QwenImagePipelineWithLogProbStep, got %s",
                type(pipeline).__name__,
            )
            raise TypeError(
                f"Stepwise worker requires QwenImagePipelineWithLogProbStep, got {type(pipeline).__name__}."
            )
        return pipeline

    def stepwise_init_runtime(self, max_bsz: int) -> dict[str, int]:
        if max_bsz <= 0:
            logger.error("Invalid max_bsz for stepwise runtime: %d", max_bsz)
            raise ValueError(f"Invalid max_bsz for stepwise runtime: {max_bsz}")
        self._manager = TensorPoolManager(max_bsz=max_bsz)
        self._states.clear()
        self._pool_specs.clear()
        self._pause_commit_queue.clear()
        self._metrics.reset()
        self._non_empty_loop_counter = 0
        self._max_bsz = max_bsz
        logger.info("StepwiseWorker runtime initialized with max_bsz=%d", max_bsz)
        return {"max_bsz": max_bsz}

    def export_metric(self) -> StepwiseWorkerMetric:
        return self._metrics

    def _ensure_runtime(self) -> TensorPoolManager:
        if self._manager is None:
            logger.error("Stepwise runtime is not initialized.")
            raise RuntimeError("Stepwise runtime is not initialized.")
        return self._manager

    def _ensure_mooncake_store(self) -> MooncakeStore:
        if self._mooncake_store is None:
            logger.error("Mooncake store is not initialized for stepwise worker.")
            raise RuntimeError("Mooncake store is not initialized for stepwise worker.")
        return self._mooncake_store

    @staticmethod
    def _mooncake_key(request_id: str, tensor_name: str) -> str:
        return f"{request_id}::{tensor_name}"

    @staticmethod
    def _strip_tensors_for_state(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return None
        if isinstance(value, dict):
            return {k: DiffusionStepwiseWorker._strip_tensors_for_state(v) for k, v in value.items()}
        if isinstance(value, list):
            return [DiffusionStepwiseWorker._strip_tensors_for_state(v) for v in value]
        if isinstance(value, tuple):
            return tuple(DiffusionStepwiseWorker._strip_tensors_for_state(v) for v in value)
        return value

    def _build_pause_state_payload(self, state: WorkerRequestState) -> dict[str, Any]:
        prompt = state.request.prompts[0] if state.request.prompts else {}
        if isinstance(prompt, dict):
            # pause_step_idx is a local control for the originating runtime.
            # Do not forward it through remote payload to avoid auto-pausing
            # again when replaying/resuming on another worker.
            prompt = dict(prompt)
            prompt.pop("pause_step_idx", None)
        current_timestep = None
        if 0 <= state.step_idx < len(state.timesteps):
            current_timestep = float(state.timesteps[state.step_idx])

        generator_state = _serialize_generator_state(state.generator)
        if generator_state is None:
            generator_state_keys: list[str] = []
            generator_state_is_list = False
        elif isinstance(generator_state, torch.Tensor):
            generator_state_keys = [_pause_state_tensor_key(state.request_id, "generator_state")]
            generator_state_is_list = False
        else:
            generator_state_keys = [
                _pause_state_tensor_key(state.request_id, "generator_state", index)
                for index in range(len(generator_state))
            ]
            generator_state_is_list = True

        collected_latents_keys = [
            _pause_state_tensor_key(state.request_id, "collected_latents", index)
            for index in range(len(state.collected_latents))
        ]
        collected_log_probs_keys: list[str | None] = []
        for index, value in enumerate(state.collected_log_probs):
            if value is None:
                collected_log_probs_keys.append(None)
                continue
            collected_log_probs_keys.append(_pause_state_tensor_key(state.request_id, "collected_log_probs", index))
        collected_timesteps_keys = [
            _pause_state_tensor_key(state.request_id, "collected_timesteps", index)
            for index in range(len(state.collected_timesteps))
        ]

        return {
            "version": 1,
            "request_id": state.request_id,
            "row_index": state.row_index,
            "step_idx": state.step_idx,
            "max_steps": state.max_steps,
            "pause_step_idx": state.pause_step_idx,
            "executed_step_count": state.step_idx,
            "current_timestep": current_timestep,
            "timesteps": list(state.timesteps),
            "do_true_cfg": state.do_true_cfg,
            "true_cfg_scale": state.true_cfg_scale,
            "noise_level": state.noise_level,
            "sde_window": tuple(state.sde_window),
            "sde_type": state.sde_type,
            "logprobs": state.logprobs,
            "guidance_scale": state.guidance_scale,
            "output_type": state.output_type,
            "height": state.height,
            "width": state.width,
            "img_shapes": list(state.img_shapes),
            "txt_seq_len": state.txt_seq_len,
            "negative_txt_seq_len": state.negative_txt_seq_len,
            "sampling_params": _serialize_sampling_params(state.request.sampling_params),
            "prompt": self._strip_tensors_for_state(prompt),
            # "tensor_names": list(REMOTE_PROMPT_TENSOR_NAMES),
            "tensor_pool_names": list(self._ensure_runtime().pools.keys()),
            "generator_state_is_list": generator_state_is_list,
            "generator_state_keys": generator_state_keys,
            "collected_latents_keys": collected_latents_keys,
            "collected_log_probs_keys": collected_log_probs_keys,
            "collected_timesteps_keys": collected_timesteps_keys,
            "collected_latents_count": len(state.collected_latents),
            "collected_log_probs_count": len(state.collected_log_probs),
            "collected_timesteps_count": len(state.collected_timesteps),
        }

    def _load_remote_prompt_payload(
        self,
        *,
        request_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        store = self._ensure_mooncake_store()
        state_key = self._mooncake_key(request_id, "state")
        payload = store.get_bytes(state_key)
        state = pickle.loads(payload)
        if not isinstance(state, dict):
            raise RuntimeError(f"Remote state payload for request_id={request_id} must decode to dict.")

        custom_prompt = state.get("prompt")
        if custom_prompt is None:
            custom_prompt = {}
        if not isinstance(custom_prompt, dict):
            raise RuntimeError(
                f"Remote state payload for request_id={request_id} must contain dict prompt payload."
            )

        # tensor_names = _string_list(
        #     state.get("tensor_names"),
        #     request_id=request_id,
        #     field_name="tensor_names",
        # )
        # if tensor_names != list(REMOTE_PROMPT_TENSOR_NAMES):
        #     raise RuntimeError(
        #         f"Remote state payload for request_id={request_id} has unexpected tensor_names={tensor_names!r}"
        #     )
        tensor_pool_names = _string_list(
            state.get("tensor_pool_names"),
            request_id=request_id,
            field_name="tensor_pool_names",
        )

        loaded_tensors: dict[str, Any] = {}
        for tensor_name in list(tensor_pool_names):
            tensor_key = self._mooncake_key(request_id, str(tensor_name))
            tensor = store.get_tensor(tensor_key, device=self._pipeline().device)
            loaded_tensors[str(tensor_name)] = tensor

        generator_state_keys = _string_list(
            state.get("generator_state_keys"),
            request_id=request_id,
            field_name="generator_state_keys",
        )
        generator_state_values = [
            store.get_tensor(key, device=self._pipeline().device) for key in generator_state_keys
        ]
        if state.get("generator_state_is_list", False):
            generator_state: torch.Tensor | list[torch.Tensor] | None = generator_state_values
        else:
            if len(generator_state_values) > 1:
                raise RuntimeError(
                    f"Remote state payload for request_id={request_id} marked generator_state as a single tensor "
                    f"but contains {len(generator_state_values)} keys."
                )
            generator_state = generator_state_values[0] if generator_state_values else None

        collected_latents_keys = _string_list(
            state.get("collected_latents_keys"),
            request_id=request_id,
            field_name="collected_latents_keys",
        )
        collected_latents = [
            store.get_tensor(key, device=self._pipeline().device) for key in collected_latents_keys
        ]
        collected_log_probs_keys = _optional_string_list(
            state.get("collected_log_probs_keys"),
            request_id=request_id,
            field_name="collected_log_probs_keys",
        )
        collected_log_probs: list[torch.Tensor | None] = []
        for key in collected_log_probs_keys:
            if key is None:
                collected_log_probs.append(None)
                continue
            collected_log_probs.append(store.get_tensor(key, device=self._pipeline().device))
        collected_timesteps_keys = _string_list(
            state.get("collected_timesteps_keys"),
            request_id=request_id,
            field_name="collected_timesteps_keys",
        )
        collected_timesteps = [
            store.get_tensor(key, device=self._pipeline().device).reshape(())
            for key in collected_timesteps_keys
        ]

        state["generator_state"] = generator_state
        state["collected_latents"] = collected_latents
        state["collected_log_probs"] = collected_log_probs
        state["collected_timesteps"] = collected_timesteps

        logger.info(
            "Loaded remote prompt payload request_id=%s state=%s loaded_tensors=%s",
            request_id,
            _summarize_remote_state_payload(state),
            {name: _tensor_summary(tensor) for name, tensor in loaded_tensors.items()},
        )

        return state, loaded_tensors

    def stepwise_pause_requests(self, request_ids: list[str]) -> dict[str, Any]:
        if not isinstance(request_ids, list):
            raise TypeError(f"request_ids must be list[str], got {type(request_ids).__name__}")
        if not request_ids:
            return {"queued_request_ids": []}

        missing = [request_id for request_id in request_ids if request_id not in self._states]
        if missing:
            raise RuntimeError(f"Pause requested for unknown request_id(s): {missing}")

        self._pause_commit_queue.append(list(request_ids))
        logger.info("Queued pause commit request_ids=%s", request_ids)
        return {"queued_request_ids": list(request_ids)}

    def _drain_pause_commit_queue(self) -> list[str]:
        drained: list[str] = []
        while self._pause_commit_queue:
            drained.extend(self._pause_commit_queue.popleft())
        return drained

    def _commit_paused_requests(self, request_ids: list[str]) -> None:
        if not request_ids:
            return

        store = self._ensure_mooncake_store()
        manager = self._ensure_runtime()
        pool_names = list(manager.pools.keys())

        for request_id in request_ids:
            if request_id not in self._states:
                raise RuntimeError(f"Cannot commit paused request_id={request_id}: state missing")

            state = self._states[request_id]
            row_index = state.row_index

            payload = self._build_pause_state_payload(state)
            prompt = state.request.prompts[0] if state.request.prompts else {}
            if not isinstance(prompt, dict):
                raise RuntimeError(
                    f"Paused request_id={request_id} must have dict prompt payload for prompt tensor export."
                )

            # prompt_tensors: dict[str, torch.Tensor] = {}
            # for tensor_name in REMOTE_PROMPT_TENSOR_NAMES:
            #     tensor_value = prompt.get(tensor_name)
            #     if tensor_value is None:
            #         raise RuntimeError(
            #             f"Paused request_id={request_id} is missing prompt tensor '{tensor_name}'"
            #         )
            #     if not isinstance(tensor_value, torch.Tensor):
            #         tensor_value = torch.as_tensor(tensor_value)
            #     prompt_tensors[tensor_name] = tensor_value

            # for tensor_name, tensor in prompt_tensors.items():
            #     store.put(self._mooncake_key(request_id, tensor_name), tensor)

            generator_state = _serialize_generator_state(state.generator)
            if generator_state is None:
                generator_state_values: list[torch.Tensor] = []
            elif isinstance(generator_state, torch.Tensor):
                generator_state_values = [generator_state]
            else:
                generator_state_values = list(generator_state)

            for key, tensor in zip(payload["generator_state_keys"], generator_state_values, strict=True):
                store.put(key, tensor)

            for key, tensor in zip(payload["collected_latents_keys"], state.collected_latents, strict=True):
                store.put(key, tensor)

            for key, tensor in zip(payload["collected_log_probs_keys"], state.collected_log_probs, strict=True):
                if key is None:
                    if tensor is not None:
                        raise RuntimeError(
                            f"Paused request_id={request_id} has tensor log_prob value without a Mooncake key."
                        )
                    continue
                if tensor is None:
                    raise RuntimeError(
                        f"Paused request_id={request_id} has Mooncake key for missing log_prob tensor."
                    )
                store.put(key, tensor)

            for key, tensor in zip(payload["collected_timesteps_keys"], state.collected_timesteps, strict=True):
                store.put(key, tensor)

            logger.info(
                "Preparing paused request state for Mooncake request_id=%s payload=%s",
                request_id,
                _summarize_pause_state_payload(payload),
            )
            store.put(self._mooncake_key(request_id, "state"), pickle.dumps(payload))

            for pool_name in pool_names:
                tensor = manager.get(pool_name, [row_index])
                store.put(self._mooncake_key(request_id, pool_name), tensor)

            logger.info(
                "Committed paused request_id=%s row=%d to Mooncake with pools=%s",
                request_id,
                row_index,
                pool_names,
            )

    def _ensure_pool(self, name: str, tensor: torch.Tensor) -> None:
        manager = self._ensure_runtime()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Pool '{name}' expects a torch.Tensor, got {type(tensor).__name__}")
        shape = tuple(tensor.shape[1:])
        dtype = tensor.dtype
        device = tensor.device

        if name not in self._pool_specs:
            manager.add(name=name, shape=shape, dtype=dtype, device=device)
            self._pool_specs[name] = (shape, dtype, device)
            logger.info("Stepwise pool created name=%s shape=%s dtype=%s device=%s", name, shape, dtype, device)
            return

        expected_shape, expected_dtype, expected_device = self._pool_specs[name]
        if shape != expected_shape:
            logger.error("Pool shape mismatch name=%s expected=%s got=%s", name, expected_shape, shape)
            raise RuntimeError(f"Pool shape mismatch name={name} expected={expected_shape} got={shape}")
        if dtype != expected_dtype:
            logger.error("Pool dtype mismatch name=%s expected=%s got=%s", name, expected_dtype, dtype)
            raise RuntimeError(f"Pool dtype mismatch name={name} expected={expected_dtype} got={dtype}")
        if device != expected_device:
            logger.error("Pool device mismatch name=%s expected=%s got=%s", name, expected_device, device)
            raise RuntimeError(f"Pool device mismatch name={name} expected={expected_device} got={device}")

    def _pad_prompt_to_len(self, embeds: torch.Tensor, mask: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if embeds.shape[1] > target_len:
            embeds = embeds[:, :target_len, :]
            mask = mask[:, :target_len]
        elif embeds.shape[1] < target_len:
            pad_len = target_len - embeds.shape[1]
            embeds = torch.cat(
                [embeds, torch.zeros((embeds.shape[0], pad_len, embeds.shape[2]), dtype=embeds.dtype, device=embeds.device)],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], pad_len), dtype=mask.dtype, device=mask.device)],
                dim=1,
            )
        return embeds, mask

    @staticmethod
    def _trim_prompt_to_mask_len(embeds: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embeds.ndim != 3 or mask.ndim != 2:
            raise RuntimeError(
                f"Invalid prompt tensor rank: embeds.ndim={embeds.ndim}, mask.ndim={mask.ndim}"
            )
        if embeds.shape[0] != mask.shape[0] or embeds.shape[1] != mask.shape[1]:
            raise RuntimeError(
                "Prompt tensor shape mismatch: "
                f"embeds.shape={tuple(embeds.shape)} mask.shape={tuple(mask.shape)}"
            )
        valid_lens = mask.to(dtype=torch.int64).sum(dim=1)
        max_valid_len = int(valid_lens.max().item()) if valid_lens.numel() > 0 else 0
        return embeds[:, :max_valid_len, :], mask[:, :max_valid_len]

    def stepwise_admit_request(self, request_id: str, request: OmniDiffusionRequest) -> AdmissionResult:
        manager = self._ensure_runtime()
        if request_id in self._states:
            logger.error("Duplicate request admission request_id=%s", request_id)
            raise RuntimeError(f"Duplicate request admission request_id={request_id}")
        pipeline = self._pipeline()
        row_index = -1

        try:
            with torch.inference_mode():
                custom_prompt = request.prompts[0] if request.prompts else {}
                if not isinstance(custom_prompt, dict):
                    logger.error("Stepwise admission expects dict prompt payload for Qwen step runtime.")
                    raise RuntimeError("Stepwise admission expects dict prompt payload for Qwen step runtime.")

                is_remote_request = bool(request.is_remote_list[0]) if request.is_remote_list else False
                remote_state: dict[str, Any] | None = None
                loaded_tensors: dict[str, Any] = {}
                resume_from_step_idx = 0
                if is_remote_request:
                    remote_state, loaded_tensors = self._load_remote_prompt_payload(request_id=request_id)
                    remote_prompt = remote_state.get("prompt", {})
                    if not isinstance(remote_prompt, dict):
                        logger.error("Remote state prompt payload must be a dict request_id=%s", request_id)
                        raise RuntimeError(f"Remote state prompt payload must be a dict request_id={request_id}")
                    row_index = int(remote_state.get("row_index", -1))
                    if row_index < 0:
                        raise RuntimeError(f"Remote state payload for request_id={request_id} is missing row_index.")
                    manager.reserve([row_index])
                    custom_prompt = dict(remote_prompt)
                    for tensor_name in remote_state.get("tensor_names", []):
                        custom_prompt[str(tensor_name)] = loaded_tensors[str(tensor_name)]
                    if request.prompts:
                        request.prompts[0] = custom_prompt
                    resume_from_step_idx = int(remote_state.get("step_idx", 0))
                else:
                    row_indices = manager.alloc(1)
                    if len(row_indices) != 1:
                        logger.error("Expected one allocated row per request, got rows=%s", row_indices)
                        raise RuntimeError(f"Expected one allocated row per request, got rows={row_indices}")
                    row_index = row_indices[0]

                prompt_ids = custom_prompt.get("prompt_ids")
                prompt_mask = custom_prompt.get("prompt_mask")
                negative_prompt_ids = custom_prompt.get("negative_prompt_ids")
                negative_prompt_mask = custom_prompt.get("negative_prompt_mask")
                pause_step_idx_raw = custom_prompt.get("pause_step_idx")
                if prompt_ids is None:
                    logger.warning("Missing prompt_ids in request_id=%s", request_id)
                    manager.release([row_index])
                    return AdmissionResult(
                        request_id=request_id,
                        row_indices=[],
                        max_steps=0,
                        current_timestep=0.0,
                        admitted=False,
                        rejection_reason=f"Missing prompt_ids in request_id={request_id}",
                    )

                sp = request.sampling_params
                height = sp.height or pipeline.default_sample_size * pipeline.vae_scale_factor
                width = sp.width or pipeline.default_sample_size * pipeline.vae_scale_factor
                num_inference_steps = int(sp.num_inference_steps)
                pause_step_idx: int | None = None
                if is_remote_request:
                    assert remote_state is not None
                    timesteps = [float(t) for t in remote_state.get("timesteps", [])]
                    if not timesteps:
                        raise RuntimeError(f"Remote state payload for request_id={request_id} is missing timesteps.")
                    resume_from_step_idx = int(remote_state.get("step_idx", 0))
                    max_steps = int(remote_state.get("max_steps", len(timesteps)))
                    if resume_from_step_idx < 0 or resume_from_step_idx >= max_steps:
                        raise RuntimeError(
                            f"Remote state payload for request_id={request_id} has invalid step_idx={resume_from_step_idx} "
                            f"for max_steps={max_steps}"
                        )
                    do_true_cfg = bool(remote_state.get("do_true_cfg", False))
                    true_cfg_scale = float(remote_state.get("true_cfg_scale", 4.0))
                    noise_level = float(remote_state.get("noise_level", 0.7))
                    sde_window = tuple(remote_state.get("sde_window", (0, max_steps - 1)))  # type: ignore[arg-type]
                    sde_type = str(remote_state.get("sde_type", "sde"))
                    logprobs = bool(remote_state.get("logprobs", True))
                    output_type = str(remote_state.get("output_type", "pil"))
                    height = int(remote_state.get("height", height))
                    width = int(remote_state.get("width", width))
                    img_shapes = [tuple(shape) for shape in remote_state.get("img_shapes", [])]
                    txt_seq_len = int(remote_state.get("txt_seq_len", 0))
                    negative_txt_seq_len = remote_state.get("negative_txt_seq_len", None)
                    if negative_txt_seq_len is not None:
                        negative_txt_seq_len = int(negative_txt_seq_len)
                    generator = _restore_generator_state(remote_state.get("generator_state"), pipeline.device)
                    prompt_embeds = loaded_tensors["prompt_embeds"]
                    prompt_embeds_mask = loaded_tensors["prompt_embeds_mask"]
                    negative_prompt_embeds = loaded_tensors["negative_prompt_embeds"]
                    negative_prompt_embeds_mask = loaded_tensors["negative_prompt_embeds_mask"]
                    latents = loaded_tensors["latents"]
                    latent_trace = {str(resume_from_step_idx): _latent_trace_summary(latents)}
                    guidance = loaded_tensors.get("guidance")
                    if guidance is None and pipeline.transformer.guidance_embeds:
                        guidance = torch.full((1, 1), float(remote_state.get("guidance_scale", sp.guidance_scale)), dtype=torch.float32, device=pipeline.device)
                    pause_step_idx = None
                    collected_latents = _restore_tensor_list(remote_state.get("collected_latents", []), pipeline.device)
                    collected_log_probs = _restore_tensor_list(remote_state.get("collected_log_probs", []), pipeline.device)
                    collected_timesteps = _restore_tensor_list(remote_state.get("collected_timesteps", []), pipeline.device)
                    remote_sampling_params = remote_state.get("sampling_params", {})
                    resume_num_inference_steps = int(
                        remote_sampling_params.get("num_inference_steps", remote_state.get("max_steps", len(timesteps)))
                    )
                    resume_sigmas = remote_sampling_params.get("sigmas", sp.sigmas)
                    timesteps, _ = pipeline.prepare_timesteps(
                        resume_num_inference_steps,
                        resume_sigmas,
                        latents.shape[1],
                    )
                    timesteps = timesteps.detach().cpu().tolist()
                    req_scheduler = copy.deepcopy(pipeline.scheduler)
                    req_scheduler.set_begin_index(resume_from_step_idx)
                else:
                    max_sequence_length = int(pipeline.tokenizer_max_length + pipeline.prompt_template_encode_start_idx)
                    if sp.max_sequence_length is not None and int(sp.max_sequence_length) > max_sequence_length:
                        logger.error(
                            "Requested max_sequence_length=%s exceeds model limit=%d for request_id=%s",
                            sp.max_sequence_length,
                            max_sequence_length,
                            request_id,
                        )
                        raise ValueError(
                            f"Requested max_sequence_length={sp.max_sequence_length} exceeds model limit={max_sequence_length} "
                            f"for request_id={request_id}"
                        )
                    if pause_step_idx_raw is not None:
                        if isinstance(pause_step_idx_raw, bool) or not isinstance(pause_step_idx_raw, int):
                            logger.error(
                                "Invalid pause_step_idx request_id=%s value=%r type=%s",
                                request_id,
                                pause_step_idx_raw,
                                type(pause_step_idx_raw).__name__,
                            )
                            raise RuntimeError(
                                f"Invalid pause_step_idx for request_id={request_id}: {pause_step_idx_raw!r}"
                            )
                        if pause_step_idx_raw < 0:
                            logger.error(
                                "Invalid pause_step_idx request_id=%s value=%d (must be >= 0)",
                                request_id,
                                pause_step_idx_raw,
                            )
                            raise RuntimeError(
                                f"Invalid pause_step_idx for request_id={request_id}: must be >= 0, got {pause_step_idx_raw}"
                            )
                        if pause_step_idx_raw >= num_inference_steps:
                            logger.error(
                                "Invalid pause_step_idx request_id=%s pause_step_idx=%d num_inference_steps=%d",
                                request_id,
                                pause_step_idx_raw,
                                num_inference_steps,
                            )
                            raise RuntimeError(
                                f"Invalid pause_step_idx for request_id={request_id}: {pause_step_idx_raw} "
                                f">= num_inference_steps({num_inference_steps})"
                            )
                        pause_step_idx = int(pause_step_idx_raw)

                    true_cfg_scale = float(sp.true_cfg_scale or 4.0)
                    noise_level = float(sp.extra_args.get("noise_level", 0.7))
                    sde_window_size = sp.extra_args.get("sde_window_size", None)
                    sde_window_range = sp.extra_args.get("sde_window_range", (0, 5))
                    sde_window_override = sp.extra_args.get("sde_window_override", None)
                    sde_type = str(sp.extra_args.get("sde_type", "sde"))
                    logprobs = bool(sp.extra_args.get("logprobs", True))
                    output_type = sp.output_type or "pil"

                    generator = sp.generator
                    if generator is None and sp.seed is not None:
                        generator = torch.Generator(device=pipeline.device).manual_seed(sp.seed)

                    if isinstance(prompt_ids, list):
                        prompt_ids = torch.tensor(prompt_ids, device=pipeline.device)
                    if prompt_mask is not None and isinstance(prompt_mask, list):
                        prompt_mask = torch.tensor(prompt_mask, device=pipeline.device)
                    if isinstance(negative_prompt_ids, list):
                        negative_prompt_ids = torch.tensor(negative_prompt_ids, device=pipeline.device)
                    if negative_prompt_mask is not None and isinstance(negative_prompt_mask, list):
                        negative_prompt_mask = torch.tensor(negative_prompt_mask, device=pipeline.device)

                    has_neg_prompt = negative_prompt_ids is not None
                    do_true_cfg = bool(true_cfg_scale > 1 and has_neg_prompt)

                    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                        prompt_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        num_images_per_prompt=1,
                        max_sequence_length=max_sequence_length,
                    )
                    prompt_embeds, prompt_embeds_mask = self._pad_prompt_to_len(
                        prompt_embeds, prompt_embeds_mask, max_sequence_length
                    )

                    if do_true_cfg:
                        negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
                            prompt_ids=negative_prompt_ids,
                            attention_mask=negative_prompt_mask,
                            num_images_per_prompt=1,
                            max_sequence_length=max_sequence_length,
                        )
                        negative_prompt_embeds, negative_prompt_embeds_mask = self._pad_prompt_to_len(
                            negative_prompt_embeds,
                            negative_prompt_embeds_mask,
                            max_sequence_length,
                        )
                    else:
                        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                        negative_prompt_embeds_mask = torch.zeros_like(prompt_embeds_mask)

                    num_channels_latents = pipeline.transformer.in_channels // 4
                    latents = pipeline.prepare_latents(
                        batch_size=1,
                        num_channels_latents=num_channels_latents,
                        height=height,
                        width=width,
                        dtype=prompt_embeds.dtype,
                        device=pipeline.device,
                        generator=generator,
                        latents=sp.latents,
                    )

                    timesteps, _ = pipeline.prepare_timesteps(num_inference_steps, sp.sigmas, latents.shape[1])
                    timesteps = timesteps.detach().cpu().tolist()
                    req_scheduler = copy.deepcopy(pipeline.scheduler)
                    req_scheduler.set_begin_index(0)

                    if pipeline.transformer.guidance_embeds:
                        guidance = torch.full((1, 1), float(sp.guidance_scale), dtype=torch.float32, device=pipeline.device)
                    else:
                        guidance = None

                    if sde_window_override is not None:
                        if not isinstance(sde_window_override, (tuple, list)) or len(sde_window_override) != 2:
                            logger.error(
                                "Invalid sde_window_override format request_id=%s override=%r type=%s",
                                request_id,
                                sde_window_override,
                                type(sde_window_override).__name__,
                            )
                            raise RuntimeError(
                                f"Invalid sde_window_override for request_id={request_id}: "
                                f"expected [start, end], got {sde_window_override!r}"
                            )
                        start_raw, end_raw = sde_window_override
                        if any((not isinstance(x, int) or isinstance(x, bool)) for x in (start_raw, end_raw)):
                            logger.error(
                                "Invalid sde_window_override values request_id=%s override=%r "
                                "(start/end must be int)",
                                request_id,
                                sde_window_override,
                            )
                            raise RuntimeError(
                                f"Invalid sde_window_override for request_id={request_id}: "
                                f"start/end must be int, got {sde_window_override!r}"
                            )
                        start = int(start_raw)
                        end = int(end_raw)
                        max_steps = len(timesteps)
                        if end <= start:
                            logger.error(
                                "Invalid sde_window_override ordering request_id=%s start=%d end=%d",
                                request_id,
                                start,
                                end,
                            )
                            raise RuntimeError(
                                f"Invalid sde_window_override for request_id={request_id}: "
                                f"end({end}) must be > start({start})"
                            )
                        if start < 0 or start >= max_steps or end > max_steps:
                            logger.error(
                                "Invalid sde_window_override range request_id=%s override=(%d, %d) "
                                "valid_start=[0,%d] valid_end=[1,%d]",
                                request_id,
                                start,
                                end,
                                max_steps - 1,
                                max_steps,
                            )
                            raise RuntimeError(
                                f"Invalid sde_window_override for request_id={request_id}: "
                                f"override=({start}, {end}) out of valid range with max_steps={max_steps} "
                                "(end is exclusive)"
                            )
                        sde_window = (start, end)
                        sde_window_source = "override"
                    elif sde_window_size is not None:
                        if not isinstance(sde_window_range, (tuple, list)) or len(sde_window_range) != 2:
                            logger.error("Invalid sde_window_range=%s for request_id=%s", sde_window_range, request_id)
                            raise RuntimeError(f"Invalid sde_window_range={sde_window_range} request_id={request_id}")
                        start = int(
                            torch.randint(
                                int(sde_window_range[0]),
                                int(sde_window_range[1]) - int(sde_window_size) + 1,
                                (1,),
                                generator=generator,
                                device=pipeline.device,
                            ).item()
                        )
                        end = start + int(sde_window_size)
                        sde_window = (start, end)
                        sde_window_source = "sampled"
                    else:
                        sde_window = (0, len(timesteps) - 1)
                        sde_window_source = "sampled"

                    logger.info(
                        "Stepwise request window selected request_id=%s source=%s sde_window=(%d, %d)",
                        request_id,
                        sde_window_source,
                        sde_window[0],
                        sde_window[1],
                    )

                    txt_seq_len = int(prompt_embeds_mask.sum(dim=1).item())
                    negative_txt_seq_len = int(negative_prompt_embeds_mask.sum(dim=1).item()) if do_true_cfg else None
                    img_shapes = [(1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2)]
                    collected_latents = []
                    collected_log_probs = []
                    collected_timesteps = []
                    resume_from_step_idx = 0
                    latent_trace = {str(resume_from_step_idx): _latent_trace_summary(latents)}

                self._ensure_pool("latents", latents)
                self._ensure_pool("prompt_embeds", prompt_embeds)
                self._ensure_pool("prompt_embeds_mask", prompt_embeds_mask)
                self._ensure_pool("negative_prompt_embeds", negative_prompt_embeds)
                self._ensure_pool("negative_prompt_embeds_mask", negative_prompt_embeds_mask)
                if guidance is not None:
                    self._ensure_pool("guidance", guidance)

                manager.put("latents", [row_index], latents)
                manager.put("prompt_embeds", [row_index], prompt_embeds)
                manager.put("prompt_embeds_mask", [row_index], prompt_embeds_mask)
                manager.put("negative_prompt_embeds", [row_index], negative_prompt_embeds)
                manager.put("negative_prompt_embeds_mask", [row_index], negative_prompt_embeds_mask)
                if guidance is not None:
                    manager.put("guidance", [row_index], guidance)

                self._states[request_id] = WorkerRequestState(
                    request_id=request_id,
                    row_index=row_index,
                    timesteps=[float(t) for t in timesteps],
                    step_idx=resume_from_step_idx,
                    max_steps=len(timesteps),
                    pause_step_idx=pause_step_idx,
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=true_cfg_scale,
                    noise_level=noise_level,
                    sde_window=sde_window,
                    sde_type=sde_type,
                    logprobs=logprobs,
                    guidance_scale=float(sp.guidance_scale),
                    output_type=output_type,
                    height=height,
                    width=width,
                    img_shapes=img_shapes,
                    txt_seq_len=txt_seq_len,
                    negative_txt_seq_len=negative_txt_seq_len,
                    generator=generator,
                    request=request,
                    scheduler=req_scheduler,
                    collected_latents=collected_latents,
                    collected_log_probs=collected_log_probs,
                    collected_timesteps=collected_timesteps,
                    resume_from_step_idx=resume_from_step_idx,
                    latent_trace=latent_trace,
                )
                logger.info(
                    "Stepwise admission complete request_id=%s row=%d max_steps=%d resume_from_step_idx=%d first_t=%.6f pause_step_idx=%s",
                    request_id,
                    row_index,
                    len(timesteps),
                    resume_from_step_idx,
                    float(timesteps[resume_from_step_idx]),
                    pause_step_idx,
                )
                return AdmissionResult(
                    request_id=request_id,
                    row_indices=[row_index],
                    max_steps=len(timesteps),
                    current_timestep=float(timesteps[resume_from_step_idx]),
                    step_idx=resume_from_step_idx,
                )
        except Exception:
            if row_index >= 0:
                try:
                    manager.release([row_index])
                except Exception:
                    pass
            raise

    def stepwise_execute_plan(self, plan: SchedulerBatchPlan) -> StepExecutionResult:
        start_time = time.perf_counter()
        manager = self._ensure_runtime()
        pipeline = self._pipeline()
        if not plan.request_ids:
            logger.error("Received empty SchedulerBatchPlan plan_id=%s", plan.plan_id)
            raise RuntimeError(f"Received empty SchedulerBatchPlan plan_id={plan.plan_id}")
        if len(plan.request_ids) != len(plan.row_indices):
            logger.error("Plan shape mismatch request_ids=%d row_indices=%d", len(plan.request_ids), len(plan.row_indices))
            raise RuntimeError("Plan shape mismatch: request_ids vs row_indices")
        self._non_empty_loop_counter += 1
        metric_id = self._non_empty_loop_counter

        for req_id, row_idx in zip(plan.request_ids, plan.row_indices, strict=True):
            if req_id not in self._states:
                logger.error("Plan references unknown request_id=%s", req_id)
                raise RuntimeError(f"Plan references unknown request_id={req_id}")
            if self._states[req_id].row_index != row_idx:
                logger.error(
                    "Plan row mismatch request_id=%s expected_row=%d got_row=%d",
                    req_id,
                    self._states[req_id].row_index,
                    row_idx,
                )
                raise RuntimeError(
                    f"Plan row mismatch request_id={req_id} expected_row={self._states[req_id].row_index} got_row={row_idx}"
                )

        with torch.inference_mode():
            row_indices = list(plan.row_indices)
            latents = manager.get("latents", row_indices)
            prompt_embeds, prompt_embeds_mask, prompt_max_seq_len = manager.get_prompt_batch(
                "prompt_embeds",
                "prompt_embeds_mask",
                row_indices,
            )
            guidance = None
            if pipeline.transformer.guidance_embeds:
                guidance = manager.get("guidance", row_indices).squeeze(-1)

            # Keep scheduler/trajectory timesteps in fp32 to match non-stepwise semantics.
            timesteps = torch.tensor(plan.timesteps, dtype=torch.float32, device=latents.device)
            # Transformer path follows latent dtype (bf16/fp16/fp32), matching non-stepwise behavior.
            timesteps_model = timesteps.to(dtype=latents.dtype)
            if timesteps.ndim != 1 or timesteps.shape[0] != len(plan.request_ids):
                raise RuntimeError(
                    f"Invalid plan.timesteps shape={tuple(timesteps.shape)} for batch={len(plan.request_ids)}."
                )
            img_shapes = [self._states[r].img_shapes for r in plan.request_ids]
            txt_seq_lens = [self._states[r].txt_seq_len for r in plan.request_ids]

        # Prepare attention/context attributes expected by denoise logic.
            pipeline._attention_kwargs = {}
            pipeline._current_timestep = None
            pipeline.transformer.do_true_cfg = False

            logger.debug(
                "Stepwise prompt batch trimmed plan_id=%s request_ids=%s max_seq_len=%d",
                plan.plan_id,
                plan.request_ids,
                prompt_max_seq_len,
            )
            transformer_kwargs = dict(
                hidden_states=latents,
                timestep=timesteps_model / 1000,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=pipeline.attention_kwargs,
                return_dict=False,
            )
            if guidance is not None:
                transformer_kwargs["guidance"] = guidance
            noise_pred = pipeline.transformer(**transformer_kwargs)[0]

            cfg_indices = [i for i, req_id in enumerate(plan.request_ids) if self._states[req_id].do_true_cfg]
            if cfg_indices:
                idx_tensor = torch.tensor(cfg_indices, dtype=torch.long, device=latents.device)
                neg_img_shapes = [img_shapes[i] for i in cfg_indices]
                neg_txt_seq_lens = [self._states[plan.request_ids[i]].negative_txt_seq_len for i in cfg_indices]
                if any(x is None for x in neg_txt_seq_lens):
                    raise RuntimeError("CFG request is missing negative_txt_seq_len.")

                negative_prompt_embeds, negative_prompt_embeds_mask, negative_prompt_max_seq_len = manager.get_prompt_batch(
                    "negative_prompt_embeds",
                    "negative_prompt_embeds_mask",
                    row_indices,
                )
                logger.debug(
                    "Stepwise negative prompt batch trimmed plan_id=%s request_ids=%s max_seq_len=%d",
                    plan.plan_id,
                    plan.request_ids,
                    negative_prompt_max_seq_len,
                )

                neg_transformer_kwargs = dict(
                    hidden_states=latents[idx_tensor],
                    timestep=timesteps_model[idx_tensor] / 1000,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask[idx_tensor],
                    encoder_hidden_states=negative_prompt_embeds[idx_tensor],
                    img_shapes=neg_img_shapes,
                    txt_seq_lens=[int(x) for x in neg_txt_seq_lens],
                    attention_kwargs=pipeline.attention_kwargs,
                    return_dict=False,
                )
                if guidance is not None:
                    neg_transformer_kwargs["guidance"] = guidance[idx_tensor]
                neg_noise_pred = pipeline.transformer(**neg_transformer_kwargs)[0]

                for local_idx, batch_idx in enumerate(cfg_indices):
                    req_id = plan.request_ids[batch_idx]
                    true_cfg_scale = self._states[req_id].true_cfg_scale
                    comb_pred = neg_noise_pred[local_idx] + true_cfg_scale * (noise_pred[batch_idx] - neg_noise_pred[local_idx])
                    cond_norm = torch.norm(noise_pred[batch_idx], dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred[batch_idx] = comb_pred * (cond_norm / noise_norm)

            next_latents: list[torch.Tensor] = []
            next_step_indices: list[int] = []
            next_timesteps: list[float | None] = []
            finished: list[bool] = []
            auto_paused_request_ids: list[str] = []

            for i, req_id in enumerate(plan.request_ids):
                state = self._states[req_id]
                step_index = state.step_idx
                # Scheduler and exported all_timesteps keep fp32 precision.
                t = torch.tensor(state.timesteps[step_index], dtype=torch.float32, device=latents.device)

                if step_index < state.sde_window[0]:
                    cur_noise_level = 0.0
                elif step_index < state.sde_window[1]:
                    cur_noise_level = state.noise_level
                else:
                    cur_noise_level = 0.0

                if step_index == state.sde_window[0]:
                    # ROLL contract expects K+1 latents for a K-step SDE window:
                    # one window-start latent + K step outputs.
                    state.collected_latents.append(latents[i : i + 1].detach().clone())

                next_latent, log_prob, _, _ = state.scheduler.step(
                    noise_pred[i : i + 1],
                    t,
                    latents[i : i + 1],
                    generator=state.generator,
                    noise_level=cur_noise_level,
                    sde_type=state.sde_type,
                    logprobs=state.logprobs,
                    return_dict=False,
                )
                next_latents.append(next_latent)

                if step_index >= state.sde_window[0] and step_index < state.sde_window[1]:
                    state.collected_latents.append(next_latent.detach().clone())
                    state.collected_log_probs.append(log_prob.detach().clone() if log_prob is not None else None)
                    state.collected_timesteps.append(t.detach().clone())

                state.step_idx += 1
                done = state.step_idx >= state.max_steps
                finished.append(done)
                next_step_indices.append(state.step_idx)
                state.latent_trace[str(state.step_idx)] = _latent_trace_summary(next_latent)
                if done:
                    next_timesteps.append(None)
                else:
                    next_timesteps.append(float(state.timesteps[state.step_idx]))

                if state.pause_step_idx is not None:
                    if step_index > state.pause_step_idx:
                        logger.error(
                            "Auto pause invariant violated request_id=%s step_index=%d pause_step_idx=%d",
                            req_id,
                            step_index,
                            state.pause_step_idx,
                        )
                        raise RuntimeError(
                            "Auto pause invariant violated "
                            f"for request_id={req_id}: step_index={step_index} > pause_step_idx={state.pause_step_idx}"
                        )
                    if step_index == state.pause_step_idx and not done:
                        auto_paused_request_ids.append(req_id)
                        logger.info(
                            "Auto pause triggered request_id=%s pause_step_idx=%d next_step_idx=%d",
                            req_id,
                            state.pause_step_idx,
                            state.step_idx,
                        )

            manager.put("latents", row_indices, torch.cat(next_latents, dim=0))
            paused_request_ids = self._drain_pause_commit_queue()
            if auto_paused_request_ids:
                paused_request_ids.extend(auto_paused_request_ids)
            if paused_request_ids:
                paused_request_ids = list(dict.fromkeys(paused_request_ids))
            paused_request_set = set(paused_request_ids)
            if paused_request_ids:
                self._commit_paused_requests(paused_request_ids)

            final_finished: list[bool] = []
            final_next_timesteps: list[float | None] = []
            final_finish_reasons: list[str | None] = []
            for i, request_id in enumerate(plan.request_ids):
                if finished[i]:
                    final_finished.append(True)
                    final_next_timesteps.append(None)
                    final_finish_reasons.append("max_steps_reached")
                elif request_id in paused_request_set:
                    final_finished.append(True)
                    final_next_timesteps.append(None)
                    final_finish_reasons.append("paused")
                else:
                    final_finished.append(False)
                    final_next_timesteps.append(next_timesteps[i])
                    final_finish_reasons.append(None)

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            cuda_allocated_mb, cuda_reserved_mb = _cuda_mem_mb(latents.device)
            loop_end_timestamp_ms = time.time() * 1000.0
            step_metric = self._metrics.record_plan(
                metric_id=metric_id,
                loop_end_timestamp_ms=loop_end_timestamp_ms,
                plan_id=plan.plan_id,
                request_ids=plan.request_ids,
                row_indices=row_indices,
                step_indices=next_step_indices,
                finished_count=sum(1 for x in final_finished),
                latency_ms=latency_ms,
                cuda_allocated_mb=cuda_allocated_mb,
                cuda_reserved_mb=cuda_reserved_mb,
            )
            logger.debug(
                "Step execution done plan_id=%s request_ids=%s finish_reasons=%s stepwise step metric: %s",
                plan.plan_id,
                plan.request_ids,
                final_finish_reasons,
                step_metric,
            )
            # logger.info("Step execution done plan_id=%s request_ids=%s", plan.plan_id, plan.request_ids)

            return StepExecutionResult(
                plan_id=plan.plan_id,
                request_ids=list(plan.request_ids),
                row_indices=row_indices,
                step_indices=next_step_indices,
                next_timesteps=final_next_timesteps,
                finished=final_finished,
                finish_reasons=final_finish_reasons,
            )

    def _finalize_request(self, request_id: str, *, require_trajectory: bool, finish_reason: str) -> DiffusionOutput:
        manager = self._ensure_runtime()
        pipeline = self._pipeline()
        if request_id not in self._states:
            logger.error("Finalize called for unknown request_id=%s", request_id)
            raise RuntimeError(f"Finalize called for unknown request_id={request_id}")
        with torch.inference_mode():
            state = self._states[request_id]
            row = [state.row_index]

            latents = manager.get("latents", row)
            prompt_embeds = manager.get("prompt_embeds", row)
            prompt_embeds_mask = manager.get("prompt_embeds_mask", row)
            negative_prompt_embeds = manager.get("negative_prompt_embeds", row)
            negative_prompt_embeds_mask = manager.get("negative_prompt_embeds_mask", row)
            prompt_embeds, prompt_embeds_mask = self._trim_prompt_to_mask_len(prompt_embeds, prompt_embeds_mask)
            negative_prompt_embeds, negative_prompt_embeds_mask = self._trim_prompt_to_mask_len(
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
            )

            all_latents = None
            all_log_probs = None
            all_timesteps = None
            if require_trajectory:
                if state.step_idx < state.max_steps:
                    logger.error(
                        "Finalize requested before max steps request_id=%s step_idx=%d max_steps=%d finish_reason=%s",
                        request_id,
                        state.step_idx,
                        state.max_steps,
                        finish_reason,
                    )
                    raise RuntimeError(
                        "Finalize requested before max steps "
                        f"for request_id={request_id}: step_idx={state.step_idx}, max_steps={state.max_steps}"
                    )
                if not state.collected_latents:
                    logger.error(
                        "No trajectory latents collected request_id=%s step_idx=%d max_steps=%d sde_window=%s "
                        "collected_latents=%d collected_timesteps=%d collected_log_probs=%d",
                        request_id,
                        state.step_idx,
                        state.max_steps,
                        state.sde_window,
                        len(state.collected_latents),
                        len(state.collected_timesteps),
                        len(state.collected_log_probs),
                    )
                    raise RuntimeError(
                        "No trajectory latents collected "
                        f"for request_id={request_id} at step_idx={state.step_idx}, sde_window={state.sde_window}"
                    )
                expected_k = int(state.sde_window[1] - state.sde_window[0])
                if len(state.collected_timesteps) != expected_k:
                    logger.error(
                        "Trajectory timestep length mismatch request_id=%s expected_k=%d timesteps=%d sde_window=%s",
                        request_id,
                        expected_k,
                        len(state.collected_timesteps),
                        state.sde_window,
                    )
                    raise RuntimeError(
                        "Trajectory timestep length mismatch "
                        f"for request_id={request_id}: expected_k={expected_k} "
                        f"timesteps={len(state.collected_timesteps)} sde_window={state.sde_window}"
                    )
                if len(state.collected_log_probs) != expected_k:
                    logger.error(
                        "Trajectory log_prob length mismatch request_id=%s expected_k=%d log_probs=%d sde_window=%s",
                        request_id,
                        expected_k,
                        len(state.collected_log_probs),
                        state.sde_window,
                    )
                    raise RuntimeError(
                        "Trajectory log_prob length mismatch "
                        f"for request_id={request_id}: expected_k={expected_k} "
                        f"log_probs={len(state.collected_log_probs)} sde_window={state.sde_window}"
                    )
                if len(state.collected_latents) != expected_k + 1:
                    logger.error(
                        "Trajectory latent length mismatch request_id=%s expected=%d latents=%d "
                        "timesteps=%d log_probs=%d sde_window=%s",
                        request_id,
                        expected_k + 1,
                        len(state.collected_latents),
                        len(state.collected_timesteps),
                        len(state.collected_log_probs),
                        state.sde_window,
                    )
                    raise RuntimeError(
                        "Trajectory latent length mismatch "
                        f"for request_id={request_id}: expected_latents={expected_k + 1} "
                        f"latents={len(state.collected_latents)} "
                        f"timesteps={len(state.collected_timesteps)} "
                        f"log_probs={len(state.collected_log_probs)} "
                        f"sde_window={state.sde_window}"
                    )

                all_latents = torch.stack(state.collected_latents, dim=1)
                if all(x is None for x in state.collected_log_probs):
                    all_log_probs = torch.zeros(
                        (latents.shape[0], len(state.collected_timesteps)),
                        dtype=latents.dtype,
                        device=latents.device,
                    )
                elif any(x is None for x in state.collected_log_probs):
                    logger.error(
                        "Trajectory log_prob consistency error request_id=%s collected_log_probs=%s",
                        request_id,
                        [x is None for x in state.collected_log_probs],
                    )
                    raise RuntimeError(
                        f"Inconsistent trajectory log_probs for request_id={request_id}: mixed None and Tensor values"
                    )
                else:
                    all_log_probs = torch.stack([x for x in state.collected_log_probs if x is not None], dim=1)
                all_timesteps = torch.stack(state.collected_timesteps).unsqueeze(0).expand(latents.shape[0], -1)

            if state.output_type == "latent":
                image = latents
            else:
                latents_unpacked = pipeline._unpack_latents(latents, state.height, state.width, pipeline.vae_scale_factor)
                latents_unpacked = latents_unpacked.to(pipeline.vae.dtype)
                latents_mean = (
                    torch.tensor(pipeline.vae.config.latents_mean)
                    .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
                    .to(latents_unpacked.device, latents_unpacked.dtype)
                )
                latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(
                    1,
                    pipeline.vae.config.z_dim,
                    1,
                    1,
                    1,
                ).to(latents_unpacked.device, latents_unpacked.dtype)
                latents_unpacked = latents_unpacked / latents_std + latents_mean
                image = pipeline.vae.decode(latents_unpacked, return_dict=False)[0][:, :, 0]

            custom_output = {
                "responses": _maybe_to_cpu(image),
                "prompt_embeds": _maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": _maybe_to_cpu(prompt_embeds_mask),
                "negative_prompt_embeds": _maybe_to_cpu(negative_prompt_embeds),
                "negative_prompt_embeds_mask": _maybe_to_cpu(negative_prompt_embeds_mask),
                "finish_reason": finish_reason,
                "step_idx": int(state.step_idx),
                "resume_from_step_idx": int(state.resume_from_step_idx),
                "executed_step_count": int(state.step_idx - state.resume_from_step_idx),
                "max_steps": int(state.max_steps),
                "pause_step_idx": state.pause_step_idx,
                "latent_trace": state.latent_trace,
            }
            if require_trajectory:
                custom_output["all_latents"] = _maybe_to_cpu(all_latents)
                custom_output["rollout_log_probs"] = _maybe_to_cpu(all_log_probs)
                custom_output["all_timesteps"] = _maybe_to_cpu(all_timesteps)

            logger.info(
                "Finalize complete request_id=%s finish_reason=%s step_idx=%d max_steps=%d include_trajectory=%s",
                request_id,
                finish_reason,
                state.step_idx,
                state.max_steps,
                require_trajectory,
            )
            return DiffusionOutput(output=_maybe_to_cpu(image), custom_output=custom_output)

    def stepwise_finalize_request(self, request_id: str) -> DiffusionOutput:
        return self._finalize_request(request_id, require_trajectory=True, finish_reason="max_steps_reached")

    def stepwise_finalize_paused_request(self, request_id: str) -> DiffusionOutput:
        return self._finalize_request(request_id, require_trajectory=False, finish_reason="paused")

    def stepwise_deadmit_request(self, request_id: str) -> None:
        manager = self._ensure_runtime()
        if request_id not in self._states:
            logger.error("Deadmit called for unknown request_id=%s", request_id)
            raise RuntimeError(f"Deadmit called for unknown request_id={request_id}")
        state = self._states.pop(request_id)
        manager.release([state.row_index])
        # TODO: Mooncake payload cleanup is intentionally deferred until the
        # remote request lifecycle is fully validated.
        logger.info("Deadmit complete request_id=%s row=%d", request_id, state.row_index)
