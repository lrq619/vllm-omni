# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import pytest

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.stepwise_scheduler import (
    AdmissionResult,
    SchedulerBatchPlan,
    StepExecutionResult,
    StepwiseScheduler,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion]


@dataclass
class _ReqState:
    request_id: str
    row: int
    step_idx: int
    max_steps: int
    timesteps: list[float]


class MockStepwiseRPCWorker:
    """Mock worker-side RPC implementation for StepwiseScheduler tests."""

    def __init__(self, per_step_sleep_s: float = 0.0):
        self._lock = threading.Lock()
        self._free_rows: list[int] = []
        self._states: dict[str, _ReqState] = {}
        self._final_outputs: dict[str, DiffusionOutput] = {}
        self.per_step_sleep_s = per_step_sleep_s

        self.rpc_calls: list[str] = []
        self.plan_history: list[SchedulerBatchPlan] = []

    def stepwise_init_runtime(self, max_bsz: int) -> dict[str, int]:
        self.rpc_calls.append("stepwise_init_runtime")
        self._free_rows = list(range(max_bsz))
        self._states.clear()
        self._final_outputs.clear()
        return {"max_bsz": max_bsz}

    def stepwise_admit_request(self, request_id: str, request: OmniDiffusionRequest) -> AdmissionResult:
        self.rpc_calls.append("stepwise_admit_request")
        with self._lock:
            if request_id in self._states:
                raise RuntimeError(f"Duplicate request_id={request_id}")
            if not self._free_rows:
                raise RuntimeError("Insufficient free rows for alloc(1): free=0, max_bsz=0.")
            row = self._free_rows.pop(0)

            max_steps = int(request.sampling_params.num_inference_steps)
            if max_steps <= 0:
                raise RuntimeError(f"Invalid num_inference_steps={max_steps}")
            timesteps = [float(max_steps - i) for i in range(max_steps)]

            self._states[request_id] = _ReqState(
                request_id=request_id,
                row=row,
                step_idx=0,
                max_steps=max_steps,
                timesteps=timesteps,
            )
            self._final_outputs[request_id] = DiffusionOutput(
                output=None,
                custom_output={"request_id": request_id, "row": row, "max_steps": max_steps},
            )
            return AdmissionResult(
                request_id=request_id,
                row_indices=[row],
                max_steps=max_steps,
                current_timestep=timesteps[0],
            )

    def stepwise_execute_plan(self, plan: SchedulerBatchPlan) -> StepExecutionResult:
        self.rpc_calls.append("stepwise_execute_plan")
        self.plan_history.append(plan)
        print(
            f"[MockStepwiseRPCWorker] plan_id={plan.plan_id[:8]} "
            f"bsz={len(plan.request_ids)} reqs={plan.request_ids} rows={plan.row_indices} steps={plan.step_indices}"
        )
        if self.per_step_sleep_s > 0:
            time.sleep(self.per_step_sleep_s)

        next_step_indices: list[int] = []
        next_timesteps: list[float | None] = []
        finished: list[bool] = []
        for req_id, row in zip(plan.request_ids, plan.row_indices, strict=True):
            state = self._states[req_id]
            if state.row != row:
                raise RuntimeError(f"Row mismatch req={req_id} expected={state.row} got={row}")
            state.step_idx += 1
            done = state.step_idx >= state.max_steps
            next_step_indices.append(state.step_idx)
            next_timesteps.append(None if done else state.timesteps[state.step_idx])
            finished.append(done)

        return StepExecutionResult(
            plan_id=plan.plan_id,
            request_ids=list(plan.request_ids),
            row_indices=list(plan.row_indices),
            step_indices=next_step_indices,
            next_timesteps=next_timesteps,
            finished=finished,
            finish_reasons=["max_steps_reached" if is_finished else None for is_finished in finished],
        )

    def stepwise_finalize_request(self, request_id: str) -> DiffusionOutput:
        self.rpc_calls.append("stepwise_finalize_request")
        return self._final_outputs[request_id]

    def stepwise_deadmit_request(self, request_id: str) -> None:
        self.rpc_calls.append("stepwise_deadmit_request")
        state = self._states.pop(request_id)
        self._free_rows.append(state.row)
        self._free_rows.sort()


def _build_request(request_id: str, steps: int) -> OmniDiffusionRequest:
    sp = OmniDiffusionSamplingParams(
        num_inference_steps=steps,
        guidance_scale=1.0,
        true_cfg_scale=1.0,
        extra_args={"noise_level": 0.7, "sde_type": "sde", "logprobs": True},
    )
    prompt = {"prompt_ids": [1, 2, 3], "prompt_mask": [1, 1, 1]}
    return OmniDiffusionRequest(prompts=[prompt], sampling_params=sp, request_ids=[request_id])


def _create_started_scheduler(worker: MockStepwiseRPCWorker, max_bsz: int = 8) -> StepwiseScheduler:
    scheduler = StepwiseScheduler()
    scheduler.od_config = OmniDiffusionConfig(model="dummy", num_gpus=1, enable_stepwise=True)
    scheduler.num_workers = 1
    scheduler._lock = threading.Lock()
    scheduler._cv = threading.Condition()
    scheduler._stop = False
    scheduler._started = True
    scheduler._pending = deque()
    scheduler._active = {}
    scheduler._max_bsz = max_bsz
    scheduler.mq = None
    scheduler.result_mq = object()

    def _rpc_call(method, args, kwargs, output_rank, exec_all_ranks):
        del output_rank, exec_all_ranks
        fn = getattr(worker, method)
        return fn(*args, **kwargs)

    scheduler._rpc_call = _rpc_call  # type: ignore[method-assign]
    scheduler._rpc_call("stepwise_init_runtime", args=(max_bsz,), kwargs={}, output_rank=0, exec_all_ranks=True)
    scheduler._worker_thread = threading.Thread(target=scheduler._main_loop, name="stepwise-scheduler-test", daemon=True)
    scheduler._worker_thread.start()
    return scheduler


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timeout while waiting for condition")


def test_stepwise_scheduler_single_request_end_to_end_and_cleanup():
    worker = MockStepwiseRPCWorker()
    scheduler = _create_started_scheduler(worker, max_bsz=4)
    request = _build_request("req-single", steps=4)

    try:
        future = scheduler.add_req(request)
        result = future.result(timeout=10)

        assert result is worker._final_outputs["req-single"]
        assert scheduler._active == {}
        assert len(scheduler._pending) == 0
        assert worker._states == {}
        assert worker._free_rows == [0, 1, 2, 3]

        assert worker.rpc_calls.count("stepwise_admit_request") == 1
        assert worker.rpc_calls.count("stepwise_execute_plan") == 4
        assert worker.rpc_calls.count("stepwise_finalize_request") == 1
        assert worker.rpc_calls.count("stepwise_deadmit_request") == 1

        assert len(worker.plan_history) == 4
        assert all(plan.request_ids == ["req-single"] for plan in worker.plan_history)
        assert [plan.step_indices[0] for plan in worker.plan_history] == [0, 1, 2, 3]
        assert all(plan.row_indices == [0] for plan in worker.plan_history)
    finally:
        scheduler.close()


def test_stepwise_scheduler_two_requests_batch_grows_to_two_midway():
    worker = MockStepwiseRPCWorker(per_step_sleep_s=0.01)
    scheduler = _create_started_scheduler(worker, max_bsz=4)
    req1 = _build_request("req-A", steps=10)
    req2 = _build_request("req-B", steps=10)

    try:
        future_a = scheduler.add_req(req1)
        _wait_until(lambda: len(worker.plan_history) >= 3, timeout_s=5.0)
        future_b = scheduler.add_req(req2)

        result_a = future_a.result(timeout=10)
        result_b = future_b.result(timeout=10)

        assert result_a is worker._final_outputs["req-A"]
        assert result_b is worker._final_outputs["req-B"]

        batch_sizes = [len(plan.request_ids) for plan in worker.plan_history]
        print(f"[MockStepwiseRPCWorker] batch_sizes_over_time={batch_sizes}")
        assert 2 in batch_sizes, f"Expected at least one batch of size 2, got {batch_sizes}"

        assert scheduler._active == {}
        assert worker._states == {}
    finally:
        scheduler.close()


def test_stepwise_scheduler_defers_pending_requests_when_at_capacity():
    worker = MockStepwiseRPCWorker(per_step_sleep_s=0.05)
    scheduler = _create_started_scheduler(worker, max_bsz=1)
    req1 = _build_request("req-capacity-A", steps=4)
    req2 = _build_request("req-capacity-B", steps=2)

    try:
        future_a = scheduler.add_req(req1)
        _wait_until(lambda: len(worker.plan_history) >= 1, timeout_s=5.0)

        future_b = scheduler.add_req(req2)
        time.sleep(0.02)
        assert worker.rpc_calls.count("stepwise_admit_request") == 1
        assert len(scheduler._pending) == 1
        assert len(scheduler._active) == 1

        result_a = future_a.result(timeout=10)
        result_b = future_b.result(timeout=10)

        assert result_a is worker._final_outputs["req-capacity-A"]
        assert result_b is worker._final_outputs["req-capacity-B"]
        assert worker.rpc_calls.count("stepwise_admit_request") == 2
        assert worker.rpc_calls.count("stepwise_execute_plan") == 6
        assert [plan.request_ids for plan in worker.plan_history[:4]] == [["req-capacity-A"]] * 4
        assert [plan.request_ids for plan in worker.plan_history[4:]] == [["req-capacity-B"]] * 2
        assert scheduler._pending == deque()
        assert scheduler._active == {}
    finally:
        scheduler.close()
