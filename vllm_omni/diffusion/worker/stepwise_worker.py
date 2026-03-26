# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwenimage_step import (
    QwenImagePipelineWithLogProbStep,
)
from vllm_omni.diffusion.models.qwen_image.tensor_pool import TensorPoolManager
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.stepwise_scheduler import (
    AdmissionResult,
    SchedulerBatchPlan,
    StepExecutionResult,
)
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

logger = init_logger(__name__)


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

    def _ensure_pool(self, name: str, tensor: torch.Tensor) -> None:
        manager = self._ensure_runtime()
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

    def stepwise_admit_request(self, request_id: str, request: OmniDiffusionRequest) -> AdmissionResult:
        manager = self._ensure_runtime()
        if request_id in self._states:
            logger.error("Duplicate request admission request_id=%s", request_id)
            raise RuntimeError(f"Duplicate request admission request_id={request_id}")

        row_indices = manager.alloc(1)
        if len(row_indices) != 1:
            logger.error("Expected one allocated row per request, got rows=%s", row_indices)
            raise RuntimeError(f"Expected one allocated row per request, got rows={row_indices}")
        row_index = row_indices[0]
        pipeline = self._pipeline()

        try:
            with torch.inference_mode():
                custom_prompt = request.prompts[0] if request.prompts else {}
                if not isinstance(custom_prompt, dict):
                    logger.error("Stepwise admission expects dict prompt payload for Qwen step runtime.")
                    raise RuntimeError("Stepwise admission expects dict prompt payload for Qwen step runtime.")

                prompt_ids = custom_prompt.get("prompt_ids")
                prompt_mask = custom_prompt.get("prompt_mask")
                negative_prompt_ids = custom_prompt.get("negative_prompt_ids")
                negative_prompt_mask = custom_prompt.get("negative_prompt_mask")
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
                true_cfg_scale = float(sp.true_cfg_scale or 4.0)
                noise_level = float(sp.extra_args.get("noise_level", 0.7))
                sde_window_size = sp.extra_args.get("sde_window_size", None)
                sde_window_range = sp.extra_args.get("sde_window_range", (0, 5))
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

                # Prepare timesteps and then clone scheduler state for this request.
                timesteps, _ = pipeline.prepare_timesteps(num_inference_steps, sp.sigmas, latents.shape[1])
                timesteps = timesteps.detach().cpu().tolist()
                req_scheduler = copy.deepcopy(pipeline.scheduler)
                req_scheduler.set_begin_index(0)

                if pipeline.transformer.guidance_embeds:
                    guidance = torch.full((1, 1), float(sp.guidance_scale), dtype=torch.float32, device=pipeline.device)
                else:
                    guidance = None

                if sde_window_size is not None:
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
                else:
                    sde_window = (0, len(timesteps) - 1)

                txt_seq_len = int(prompt_embeds_mask.sum(dim=1).item())
                negative_txt_seq_len = int(negative_prompt_embeds_mask.sum(dim=1).item()) if do_true_cfg else None
                # QwenEmbedRope.forward() expects a flat list of (frame, height, width) tuples here.
                img_shapes = [(1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2)]

                # Ensure pools are initialized and shape-consistent.
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
                    step_idx=0,
                    max_steps=len(timesteps),
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
                )
                logger.info(
                    "Stepwise admission complete request_id=%s row=%d max_steps=%d first_t=%.6f",
                    request_id,
                    row_index,
                    len(timesteps),
                    float(timesteps[0]),
                )
                return AdmissionResult(
                    request_id=request_id,
                    row_indices=[row_index],
                    max_steps=len(timesteps),
                    current_timestep=float(timesteps[0]),
                )
        except Exception:
            manager.release([row_index])
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

            timesteps = torch.tensor(plan.timesteps, dtype=latents.dtype, device=latents.device)
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
                timestep=timesteps / 1000,
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
                    timestep=timesteps[idx_tensor] / 1000,
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

            for i, req_id in enumerate(plan.request_ids):
                state = self._states[req_id]
                step_index = state.step_idx
                t = torch.tensor(state.timesteps[step_index], dtype=latents.dtype, device=latents.device)

                if step_index < state.sde_window[0]:
                    cur_noise_level = 0.0
                elif step_index < state.sde_window[1]:
                    cur_noise_level = state.noise_level
                else:
                    cur_noise_level = 0.0

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
                if done:
                    next_timesteps.append(None)
                else:
                    next_timesteps.append(float(state.timesteps[state.step_idx]))

            manager.put("latents", row_indices, torch.cat(next_latents, dim=0))
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
                finished_count=sum(1 for x in finished if x),
                latency_ms=latency_ms,
                cuda_allocated_mb=cuda_allocated_mb,
                cuda_reserved_mb=cuda_reserved_mb,
            )
            logger.debug("Step execution done plan_id=%s request_ids=%s, stepwise step metric: %s", plan.plan_id, plan.request_ids, step_metric)
            # logger.info("Step execution done plan_id=%s request_ids=%s", plan.plan_id, plan.request_ids)

            return StepExecutionResult(
                plan_id=plan.plan_id,
                request_ids=list(plan.request_ids),
                row_indices=row_indices,
                step_indices=next_step_indices,
                next_timesteps=next_timesteps,
                finished=finished,
            )

    def stepwise_finalize_request(self, request_id: str) -> DiffusionOutput:
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

            if not state.collected_latents:
                logger.error("No trajectory latents collected for request_id=%s", request_id)
                raise RuntimeError(f"No trajectory latents collected for request_id={request_id}")
            all_latents = torch.stack(state.collected_latents, dim=1)

            if state.collected_log_probs and state.collected_log_probs[0] is not None:
                all_log_probs = torch.stack([x for x in state.collected_log_probs if x is not None], dim=1)
            else:
                all_log_probs = torch.zeros((latents.shape[0], len(state.collected_timesteps)), dtype=latents.dtype, device=latents.device)
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

            logger.info("Finalize complete request_id=%s", request_id)
            return DiffusionOutput(
                output=_maybe_to_cpu(image),
                custom_output={
                    "responses": _maybe_to_cpu(image),
                    "all_latents": _maybe_to_cpu(all_latents),
                    "rollout_log_probs": _maybe_to_cpu(all_log_probs),
                    "all_timesteps": _maybe_to_cpu(all_timesteps),
                    "prompt_embeds": _maybe_to_cpu(prompt_embeds),
                    "prompt_embeds_mask": _maybe_to_cpu(prompt_embeds_mask),
                    "negative_prompt_embeds": _maybe_to_cpu(negative_prompt_embeds),
                    "negative_prompt_embeds_mask": _maybe_to_cpu(negative_prompt_embeds_mask),
                },
            )

    def stepwise_deadmit_request(self, request_id: str) -> None:
        manager = self._ensure_runtime()
        if request_id not in self._states:
            logger.error("Deadmit called for unknown request_id=%s", request_id)
            raise RuntimeError(f"Deadmit called for unknown request_id={request_id}")
        state = self._states.pop(request_id)
        manager.release([state.row_index])
        logger.info("Deadmit complete request_id=%s row=%d", request_id, state.row_index)
