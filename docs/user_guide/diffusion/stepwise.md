# Stepwise Diffusion Guide

This document explains how to enable and use the stepwise diffusion runtime in vLLM-Omni, including Mooncake settings, `generate` parameters, and pause/resume workflows.

## What Stepwise Mode Does

When stepwise mode is enabled:

- Diffusion execution uses the stepwise scheduler/worker path.
- Request state can be paused mid-generation and committed to Mooncake.
- A paused request can be resumed remotely (for migration/failover style flows).

At runtime, stepwise mode is selected by `enable_stepwise=True` and uses:

- `StepwiseScheduler`
- `DiffusionStepwiseWorker`

## 1. Enable Stepwise in `AsyncOmni`

### Python API (recommended)

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni

omni = AsyncOmni(
    model="your-model",
    enable_stepwise=True,
    max_step_batch_size=8,  # optional, default is 8
    diffusion_load_format="custom_pipeline",
    custom_pipeline_args={
        "pipeline_class": (
            "vllm_omni.diffusion.models.qwen_image.pipeline_qwenimage_step."
            "QwenImagePipelineWithLogProbStep"
        )
    },
)
```

### Stage config YAML

If you use stage config files, set stepwise in diffusion stage `engine_args`:

```yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    runtime:
      process: true
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_stage: diffusion
      enable_stepwise: true
      max_step_batch_size: 8
      diffusion_load_format: custom_pipeline
      custom_pipeline_args:
        pipeline_class: vllm_omni.diffusion.models.qwen_image.pipeline_qwenimage_step.QwenImagePipelineWithLogProbStep
```

### CLI

Use:

- `--step-wise` (alias: `--enable-step-wise`, `--enable_step_wise`)
- `--max-step-batch-size <int>` (default: `8`)

### Pipeline compatibility (important)

Current stepwise worker enforces a step-compatible pipeline type.  
In the current codebase, this means using:

- `vllm_omni.diffusion.models.qwen_image.pipeline_qwenimage_step.QwenImagePipelineWithLogProbStep`

If you enable stepwise but load a non-step pipeline, worker admission fails fast with an explicit error (e.g. type mismatch in stepwise worker pipeline check).

## 2. Mooncake Environment Variables

Stepwise pause/resume relies on `MooncakeStore`. The following env vars are read by `vllm_omni/diffusion/mooncake_store.py`.

| Env var | Meaning | Default |
|---|---|---|
| `MOONCAKE_STORAGE_HOST` | Mooncake metadata/storage host | `hostname -i` result (fallback `localhost`) |
| `MOONCAKE_STORAGE_PORT` | Mooncake metadata/storage port | `10619` |
| `MOONCAKE_MASTER_HOST` | Mooncake master host | same as `MOONCAKE_STORAGE_HOST` |
| `MOONCAKE_MASTER_PORT` | Mooncake master port | `50051` |
| `MOONCAKE_PROTOCOL` | Mooncake transport protocol | `tcp` |
| `MOONCAKE_NODE_ADDR` | local node address for Mooncake client setup | `hostname -i` result (fallback `localhost`) |
| `MOONCAKE_SEGMENT_SIZE_GB` | segment size in GB for Mooncake setup | `32` |

Notes:

- `MOONCAKE_SEGMENT_SIZE_GB` must be a positive integer.
- Runtime computes bytes as:
  - `segment_size_bytes = MOONCAKE_SEGMENT_SIZE_GB * (1024 ** 3)`

## 3. `generate` Parameters Related to Stepwise

`AsyncOmni.generate(...)` now supports:

- `pause_step_idx: int | None = None`
- `is_remote: bool = False` (existing)

### `pause_step_idx`

- If `None`: normal behavior (no auto-pause).
- If set to integer `k`: request auto-pauses after executing step `k`.
- The request then finalizes with paused flow (`finish_reason="paused"`).

Validation (fail-fast):

- Must be `int` and not `bool`.
- Must be `>= 0`.
- In worker admission it must be `< num_inference_steps`.
- Prompt must be dict-based payload when using `pause_step_idx`.

### `is_remote`

- Used to resume from previously committed Mooncake state.
- Remote diffusion requests require stepwise mode enabled.

## 4. Two Pause Modes

### A) External pause API

Call from orchestrator side:

```python
await omni.pause([request_id])
```

Behavior:

- Queues pause commit for those request IDs.
- On next stepwise loop, state/tensors are committed to Mooncake.
- Request finishes through paused finalize path.

### B) Internal auto-pause via `pause_step_idx`

Set on request submission:

```python
async for out in omni.generate(
    prompt=prompt_dict,
    request_id="req-0",
    sampling_params_list=[sampling_params],
    pause_step_idx=25,
):
    ...
```

Behavior:

- Worker checks each step.
- When current executed step index equals `pause_step_idx`, request is auto-paused.
- No external `omni.pause()` call is needed.

## 5. Resume/Migration Flow (Stepwise)

Typical flow:

1. Run request on instance A (using either pause mechanism).
2. Wait until paused output returns.
3. Submit same `request_id` on instance B with `is_remote=True`.
4. Continue generation to completion.
5. Compare with baseline if needed.

Minimal shape:

```python
# stage A
paused_out = None
async for out in omni_a.generate(
    prompt=prompt_dict,
    request_id=req_id,
    sampling_params_list=[sp],
    pause_step_idx=25,  # or call await omni_a.pause([req_id]) externally
):
    paused_out = out

# stage B
async for out in omni_b.generate(
    prompt=prompt_dict,
    request_id=req_id,
    sampling_params_list=[sp],
    is_remote=True,
):
    final_out = out
```

## 6. Output Semantics

For stepwise diffusion output, `custom_output` includes fields such as:

- `finish_reason`: `"paused"` or `"max_steps_reached"`
- `step_idx`
- `max_steps`
- `pause_step_idx`

Trajectory fields are included only for full completion path (`max_steps_reached`), not for paused finalize path.

## 7. Common Fail-Fast Errors

- `pause() requires at least one diffusion stage with enable_stepwise=True`
- invalid `pause_step_idx` type/range
- `pause_step_idx` used with non-dict prompt payload
- remote request while stepwise disabled
- missing Mooncake state/tensor keys during remote resume

These are intentional fail-fast checks with explicit logs to make configuration/runtime issues visible early.
