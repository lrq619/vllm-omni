# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


@dataclass
class AdmissionResult:
    request_id: str
    row_indices: list[int]
    max_steps: int
    current_timestep: float
    admitted: bool = True
    rejection_reason: str | None = None


@dataclass
class RequestRuntimeState:
    request_id: str
    status: str
    row_indices: list[int]
    step_idx: int
    max_steps: int
    current_timestep: float | None
    is_remote: bool = False
    finish_reason: str | None = None
    error_message: str | None = None
    result_sink_key: str = ""
    future: Future[DiffusionOutput] = field(default_factory=Future)
    result: DiffusionOutput | None = None

    # Per-step controls.
    do_true_cfg: bool = False
    guidance: float = 1.0
    true_cfg_scale: float = 1.0
    noise_level: float = 0.0
    sde_type: str = "sde"
    logprobs: bool = True


@dataclass
class SchedulerBatchPlan:
    plan_id: str
    request_ids: list[str]
    row_indices: list[int]
    step_indices: list[int]
    timesteps: list[float]
    do_true_cfg: list[bool]
    guidance: list[float]
    true_cfg_scale: list[float]
    noise_level: list[float]
    sde_type: list[str]
    logprobs: list[bool]
    pool_keys: dict[str, str]


@dataclass
class StepExecutionResult:
    plan_id: str
    request_ids: list[str]
    row_indices: list[int]
    step_indices: list[int]
    next_timesteps: list[float | None]
    finished: list[bool]


class StepwiseScheduler:
    """
    A dedicated scheduler for QwenImagePipelineWithLogProb step-wise execution.

    This scheduler owns request metadata/state transitions and drives worker-side
    one-step denoise execution through explicit SchedulerBatchPlan messages.
    """

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        existing_mq = getattr(self, "mq", None)
        if existing_mq is not None and not existing_mq.closed:
            logger.warning("StepwiseScheduler is already initialized. Re-initializing.")
            self.close()

        self.num_workers = od_config.num_gpus
        self.od_config = od_config
        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self._stop = False
        self._started = False

        self.mq = MessageQueue(
            n_reader=self.num_workers,
            n_local_reader=self.num_workers,
            local_reader_ranks=list(range(self.num_workers)),
        )
        self.result_mq = None

        self._pending: deque[tuple[OmniDiffusionRequest, RequestRuntimeState]] = deque()
        self._active: dict[str, RequestRuntimeState] = {}
        self._max_bsz = int(od_config.max_step_batch_size)
        if self._max_bsz <= 0:
            logger.error("Invalid max_step_batch_size=%d", self._max_bsz)
            raise ValueError(f"Invalid max_step_batch_size={self._max_bsz}")

    def initialize_result_queue(self, handle) -> None:
        self.result_mq = MessageQueue.create_from_handle(handle, rank=0)
        logger.info("StepwiseScheduler initialized result MessageQueue")
        self._rpc_call(
            "stepwise_init_runtime",
            args=(self._max_bsz,),
            kwargs={},
            output_rank=0,
            exec_all_ranks=True,
        )
        self._stop = False
        self._worker_thread = threading.Thread(target=self._main_loop, name="stepwise-scheduler", daemon=True)
        self._worker_thread.start()
        self._started = True
        logger.info("StepwiseScheduler main loop started with max_bsz=%d", self._max_bsz)

    def get_broadcast_handle(self):
        return self.mq.export_handle()

    def add_req(self, request: OmniDiffusionRequest) -> Future[DiffusionOutput]:
        if not self._started:
            logger.error("StepwiseScheduler add_req called before start.")
            raise RuntimeError("StepwiseScheduler not started.")
        if not request.request_ids:
            logger.error("Request is missing request_ids for stepwise runtime.")
            raise ValueError("Stepwise runtime requires request.request_ids[0].")
        request_id = request.request_ids[0]
        is_remote = bool(request.is_remote_list[0]) if request.is_remote_list else False
        state = RequestRuntimeState(
            request_id=request_id,
            status="pending",
            row_indices=[],
            step_idx=0,
            max_steps=0,
            current_timestep=None,
            is_remote=is_remote,
            result_sink_key=request_id,
        )

        with self._cv:
            self._pending.append((request, state))
            self._cv.notify_all()

        return state.future

    def _resolve_future(self, state: RequestRuntimeState, output: DiffusionOutput) -> None:
        if state.future.done():
            logger.error("Future already resolved request_id=%s", state.request_id)
            raise RuntimeError(f"Future already resolved request_id={state.request_id}")
        state.future.set_result(output)

    def _admit_local(self, request: OmniDiffusionRequest, state: RequestRuntimeState) -> AdmissionResult:
        return self._admit_request(request, state, remote=False)

    def _admit_request(self, request: OmniDiffusionRequest, state: RequestRuntimeState, *, remote: bool) -> AdmissionResult:
        request_id = state.request_id
        if request_id in self._active:
            logger.error("Duplicate active request_id=%s", request_id)
            raise RuntimeError(f"Duplicate active request_id={request_id}")
        logger.info("Admission start request_id=%s remote=%s", request_id, remote)
        result = self._rpc_call(
            "stepwise_admit_request",
            args=(request_id, request),
            kwargs={},
            output_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, AdmissionResult):
            logger.error("Admission returned invalid type=%s request_id=%s", type(result).__name__, request_id)
            raise RuntimeError(f"Invalid admission result type={type(result).__name__} request_id={request_id}")
        if not result.admitted:
            logger.warning(
                "Admission rejected request_id=%s reason=%s",
                request_id,
                result.rejection_reason,
            )
            return result
        logger.info(
            "Admission end request_id=%s remote=%s rows=%s max_steps=%d first_t=%.6f",
            request_id,
            remote,
            result.row_indices,
            result.max_steps,
            result.current_timestep,
        )
        return result

    def _admit_remote_unimplemented(self, request: OmniDiffusionRequest) -> AdmissionResult:
        request_id = request.request_ids[0] if request.request_ids else "<missing>"
        logger.info("Remote admission routed through Mooncake restore request_id=%s", request_id)
        state = RequestRuntimeState(
            request_id=request_id,
            status="pending",
            row_indices=[],
            step_idx=0,
            max_steps=0,
            current_timestep=None,
            is_remote=True,
            result_sink_key=request_id,
        )
        return self._admit_request(request, state, remote=True)

    def _rpc_call(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output_rank: int | None,
        exec_all_ranks: bool,
    ) -> Any:
        if self.result_mq is None:
            logger.error("Result queue not initialized before RPC method=%s", method)
            raise RuntimeError("Result queue not initialized.")

        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": output_rank,
            "exec_all_ranks": exec_all_ranks,
        }
        with self._lock:
            self.mq.enqueue(rpc_request)
            response = self.result_mq.dequeue()
        if isinstance(response, DiffusionOutput) and response.error is not None:
            logger.error("RPC method=%s failed: %s", method, response.error)
            raise RuntimeError(f"Worker RPC {method} failed: {response.error}")
        if isinstance(response, dict) and response.get("status") == "error":
            logger.error("RPC method=%s failed: %s", method, response.get("error"))
            raise RuntimeError(f"Worker RPC {method} failed: {response.get('error')}")
        return response

    def _build_batch_plan(self) -> SchedulerBatchPlan:
        active_states = list(self._active.values())
        if not active_states:
            raise RuntimeError("Cannot build batch plan with no active requests.")

        plan_id = str(uuid.uuid4())
        request_ids = [s.request_id for s in active_states]
        row_indices = [s.row_indices[0] for s in active_states]
        step_indices = [s.step_idx for s in active_states]
        timesteps: list[float] = []
        for s in active_states:
            if s.current_timestep is None:
                logger.error("Active request has no current_timestep request_id=%s", s.request_id)
                raise RuntimeError(f"Active request has no current_timestep request_id={s.request_id}")
            timesteps.append(float(s.current_timestep))

        plan = SchedulerBatchPlan(
            plan_id=plan_id,
            request_ids=request_ids,
            row_indices=row_indices,
            step_indices=step_indices,
            timesteps=timesteps,
            do_true_cfg=[s.do_true_cfg for s in active_states],
            guidance=[s.guidance for s in active_states],
            true_cfg_scale=[s.true_cfg_scale for s in active_states],
            noise_level=[s.noise_level for s in active_states],
            sde_type=[s.sde_type for s in active_states],
            logprobs=[s.logprobs for s in active_states],
            pool_keys={
                "latents": "latents",
                "prompt_embeds": "prompt_embeds",
                "prompt_embeds_mask": "prompt_embeds_mask",
                "negative_prompt_embeds": "negative_prompt_embeds",
                "negative_prompt_embeds_mask": "negative_prompt_embeds_mask",
                "guidance": "guidance",
            },
        )
        logger.info(
            "Batch plan created plan_id=%s request_ids=%s rows=%s step_indices=%s",
            plan.plan_id,
            plan.request_ids,
            plan.row_indices,
            plan.step_indices,
        )
        return plan

    def _finish_request(self, request_id: str, finish_reason: str) -> None:
        if request_id not in self._active:
            logger.error("Cannot finish non-active request_id=%s", request_id)
            raise RuntimeError(f"Cannot finish non-active request_id={request_id}")
        state = self._active[request_id]

        output = self._rpc_call(
            "stepwise_finalize_request",
            args=(request_id,),
            kwargs={},
            output_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(output, DiffusionOutput):
            logger.error("Finalize returned invalid type=%s request_id=%s", type(output).__name__, request_id)
            raise RuntimeError(f"Finalize returned invalid type={type(output).__name__} request_id={request_id}")

        self._rpc_call(
            "stepwise_deadmit_request",
            args=(request_id,),
            kwargs={},
            output_rank=0,
            exec_all_ranks=True,
        )

        state.status = "finished"
        state.finish_reason = finish_reason
        state.result = output
        self._resolve_future(state, output)
        del self._active[request_id]
        logger.info("Deadmission complete request_id=%s rows=%s reason=%s", request_id, state.row_indices, finish_reason)

    def _fail_request(self, request_id: str, error_message: str) -> None:
        if request_id not in self._active:
            logger.error("Cannot fail non-active request_id=%s", request_id)
            raise RuntimeError(f"Cannot fail non-active request_id={request_id}")
        state = self._active[request_id]
        deadmit_error: str | None = None
        try:
            self._rpc_call(
                "stepwise_deadmit_request",
                args=(request_id,),
                kwargs={},
                output_rank=0,
                exec_all_ranks=True,
            )
        except Exception as exc:
            logger.error("Deadmit RPC failed request_id=%s error=%s", request_id, exc)
            deadmit_error = str(exc)

        state.status = "error"
        if deadmit_error is not None:
            state.error_message = f"{error_message}; deadmit_failed={deadmit_error}"
        else:
            state.error_message = error_message
        state.result = DiffusionOutput(error=state.error_message)
        self._resolve_future(state, state.result)
        del self._active[request_id]
        logger.error("Request failed request_id=%s error=%s", request_id, state.error_message)

    def _main_loop(self) -> None:
        while True:
            try:
                with self._cv:
                    while not self._stop and not self._pending and not self._active:
                        self._cv.wait(timeout=0.1)
                    if self._stop:
                        break

                # Admit new requests while rows are available.
                while True:
                    with self._cv:
                        if not self._pending:
                            break
                        request, state = self._pending.popleft()
                    try:
                        admission = self._admit_remote_unimplemented(request) if state.is_remote else self._admit_local(request, state)
                        if not admission.admitted:
                            state.status = "finished"
                            state.finish_reason = "admission_rejected"
                            state.result = DiffusionOutput(output=None, custom_output={})
                            self._resolve_future(state, state.result)
                            logger.info(
                                "Admission rejected request_id=%s; returned dummy DiffusionOutput. reason=%s",
                                state.request_id,
                                admission.rejection_reason,
                            )
                            continue
                        state.status = "active"
                        state.row_indices = admission.row_indices
                        state.max_steps = admission.max_steps
                        state.current_timestep = admission.current_timestep
                        # Scheduler side metadata mirrors worker-side immutable controls.
                        sp = request.sampling_params
                        state.do_true_cfg = bool(sp.true_cfg_scale and sp.true_cfg_scale > 1.0)
                        state.guidance = float(sp.guidance_scale)
                        state.true_cfg_scale = float(sp.true_cfg_scale or 1.0)
                        state.noise_level = float(sp.extra_args.get("noise_level", 0.7))
                        state.sde_type = str(sp.extra_args.get("sde_type", "sde"))
                        state.logprobs = bool(sp.extra_args.get("logprobs", True))
                        self._active[state.request_id] = state
                        logger.info("Request activated request_id=%s rows=%s", state.request_id, state.row_indices)
                    except Exception as exc:
                        err_text = str(exc)
                        if "Insufficient free rows for alloc(" in err_text:
                            with self._cv:
                                self._pending.appendleft((request, state))
                            logger.info(
                                "Admission deferred due to capacity request_id=%s error=%s",
                                state.request_id,
                                err_text,
                            )
                            break
                        state.status = "error"
                        state.error_message = err_text
                        state.result = DiffusionOutput(error=err_text)
                        self._resolve_future(state, state.result)
                        logger.error("Admission failed request_id=%s error=%s", state.request_id, exc)

                if not self._active:
                    continue

                plan = self._build_batch_plan()
                try:
                    step_result = self._rpc_call(
                        "stepwise_execute_plan",
                        args=(plan,),
                        kwargs={},
                        output_rank=0,
                        exec_all_ranks=True,
                    )
                except Exception as exc:
                    # Fail all active requests on worker-step failure.
                    active_ids = list(self._active.keys())
                    for request_id in active_ids:
                        self._fail_request(request_id, f"Worker step failed: {exc}")
                    continue

                if not isinstance(step_result, StepExecutionResult):
                    active_ids = list(self._active.keys())
                    err = f"Invalid step result type={type(step_result).__name__}"
                    logger.error(err)
                    for request_id in active_ids:
                        self._fail_request(request_id, err)
                    continue

                if step_result.plan_id != plan.plan_id:
                    active_ids = list(self._active.keys())
                    err = f"Plan/result mismatch plan_id={plan.plan_id} result_plan_id={step_result.plan_id}"
                    logger.error(err)
                    for request_id in active_ids:
                        self._fail_request(request_id, err)
                    continue

                for i, request_id in enumerate(step_result.request_ids):
                    if request_id not in self._active:
                        raise RuntimeError(f"Step result contains unknown request_id={request_id}")
                    state = self._active[request_id]
                    state.step_idx = step_result.step_indices[i]
                    state.current_timestep = step_result.next_timesteps[i]
                    if step_result.finished[i]:
                        self._finish_request(request_id, finish_reason="max_steps_reached")
            except Exception as exc:
                logger.error("StepwiseScheduler main loop fatal error: %s", exc, exc_info=True)
                active_ids = list(self._active.keys())
                for request_id in active_ids:
                    self._fail_request(request_id, f"Scheduler main loop error: {exc}")

    def close(self) -> None:
        if getattr(self, "_started", False):
            with self._cv:
                self._stop = True
                self._cv.notify_all()
            self._worker_thread.join(timeout=10)
            self._started = False

        # Drain pending requests with deterministic error.
        if hasattr(self, "_pending"):
            while self._pending:
                _, state = self._pending.popleft()
                state.status = "error"
                state.error_message = "Scheduler closed before request execution."
                state.result = DiffusionOutput(error=state.error_message)
                self._resolve_future(state, state.result)

        if hasattr(self, "_active"):
            for request_id in list(self._active.keys()):
                state = self._active[request_id]
                state.status = "error"
                state.error_message = "Scheduler closed during request execution."
                state.result = DiffusionOutput(error=state.error_message)
                self._resolve_future(state, state.result)
                del self._active[request_id]

        self.mq = None
        self.result_mq = None
