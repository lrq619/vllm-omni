# TeaCache Configuration Guide

TeaCache speeds up diffusion model inference by caching transformer computations when consecutive timesteps are similar. This typically provides **1.5x-2.0x speedup** with minimal quality loss.

## Quick Start

Enable TeaCache by setting `cache_backend` to `"tea_cache"`:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Simple configuration - model_type is automatically extracted from pipeline.__class__.__name__
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.2  # Optional, defaults to 0.2
    }
)
outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
    ),
)
```

### Using Environment Variable

You can also enable TeaCache via environment variable:

```bash
export DIFFUSION_CACHE_BACKEND=tea_cache
```

Then initialize without explicitly setting `cache_backend`:

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_config={"rel_l1_thresh": 0.2}  # Optional
)
```

## Online Serving (OpenAI-Compatible)

Enable TeaCache for online serving by passing `--cache-backend tea_cache` when starting the server:

```bash
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh": 0.2}'
```

## Configuration Parameters

### `rel_l1_thresh` (float, default: `0.2`)

Controls the balance between speed and quality. Lower values prioritize quality, higher values prioritize speed.

**Recommended values:**

- `0.2` - **~1.5x speedup** with minimal quality loss (recommended)
- `0.4` - **~1.8x speedup** with slight quality loss
- `0.6` - **~2.0x speedup** with noticeable quality loss
- `0.8` - **~2.25x speedup** with significant quality loss

## Examples

### Python API

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2}
)
outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
    ),
)
```

### AsyncOmni + Custom Pipeline (Qwen non-step)

Use this mode when you load a custom diffusion pipeline via `diffusion_load_format="custom_pipeline"`, for example:
`vllm_omni.diffusion.models.qwen_image.non_step.pipeline_qwenimage.QwenImagePipelineWithLogProb`.

```python
import asyncio

from transformers import AutoTokenizer

from vllm_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "/tmp/models/Qwen/Qwen-Image"


def build_custom_prompt(tokenizer):
    # This custom pipeline expects prompt ids/masks in prompt dict.
    messages = [
        {"role": "system", "content": "Describe the image in detail."},
        {"role": "user", "content": "A green apple on a wooden table."},
    ]
    neg_messages = [
        {"role": "system", "content": "Describe the image in detail."},
        {"role": "user", "content": " "},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    negative_prompt_ids = tokenizer.apply_chat_template(neg_messages, tokenize=True, add_generation_prompt=True)
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": None,
        "negative_prompt_ids": negative_prompt_ids,
        "negative_prompt_mask": None,
    }


async def main():
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL}/tokenizer", local_files_only=True, trust_remote_code=True)
    prompt = build_custom_prompt(tokenizer)

    omni = AsyncOmni(
        model=MODEL,
        num_gpus=1,
        diffusion_load_format="custom_pipeline",
        custom_pipeline_args={
            "pipeline_class": (
                "vllm_omni.diffusion.models.qwen_image.non_step.pipeline_qwenimage."
                "QwenImagePipelineWithLogProb"
            )
        },
        cache_backend="tea_cache",
        cache_config={"rel_l1_thresh": 0.2},
    )

    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=50,
        true_cfg_scale=4.0,
        width=512,
        height=512,
        output_type="pil",
        seed=42,
        extra_args={
            "noise_level": 1.0,
            "sde_type": "sde",
            "logprobs": True,
            "sde_window_size": 2,
            "sde_window_range": [0, 5],
            # New behavior: do not skip any step inside sde_window.
            "teacache_no_skip_in_sde_window": True,
        },
    )

    final_output = None
    try:
        async for out in omni.generate(
            prompt=prompt,
            request_id="async-teacache-custom-pipeline-demo",
            sampling_params_list=[sampling_params],
        ):
            final_output = out
    finally:
        omni.close()

    teacache_metrics = final_output.custom_output["teacache_metrics"]
    print("executed steps:", teacache_metrics["denoise_executed_steps"])
    print("fully skipped steps:", teacache_metrics["denoise_fully_skipped_steps"])
    print("skipped calls in sde_window:", teacache_metrics["transformer_skipped_calls_in_sde_window"])


asyncio.run(main())
```

## AsyncOmni Parameters (Custom Pipeline + TeaCache)

### Engine-level parameters

- `cache_backend`: set to `"tea_cache"` to enable TeaCache.
- `cache_config.rel_l1_thresh`: TeaCache threshold controlling speed/quality tradeoff.
- `diffusion_load_format`: set to `"custom_pipeline"` when loading custom pipeline classes.
- `custom_pipeline_args.pipeline_class`: full import path of your custom pipeline class.

### Sampling-level parameters (`OmniDiffusionSamplingParams.extra_args`)

- `noise_level` (`float`): SDE noise injection strength in the SDE window.
- `sde_type` (`"sde"` or `"cps"`): scheduler branch used in the SDE step rule.
- `logprobs` (`bool`): whether to compute rollout logprobs.
- `sde_window_size` (`int`): random SDE window length.
- `sde_window_range` (`[start, end]`): sampling range for random SDE window start.
- `teacache_no_skip_in_sde_window` (`bool`, default `True` in this pipeline):
  when `True`, TeaCache will force compute all denoise steps inside `sde_window` (no skip).

## Getting Actual Executed Steps

For this custom pipeline, TeaCache metrics are returned in:

```python
metrics = final_output.custom_output["teacache_metrics"]
```

Important fields:

- `denoise_executed_steps`: number of denoise steps that were actually executed (at least once).
- `denoise_fully_skipped_steps`: number of denoise steps fully skipped by TeaCache.
- `transformer_skipped_calls`: skipped transformer calls (call-level metric).
- `transformer_skipped_calls_in_sde_window`: skipped calls inside `sde_window` (expected `0` when `teacache_no_skip_in_sde_window=True`).
- `sde_window`: actual `[start, end]` window used in this generation.

Note:

- Step-level metrics and call-level metrics are different under CFG.
- With CFG enabled, one denoise step may invoke transformer multiple times (positive/negative branches).

## Error Signaling (No Silent Fallback in New Path)

For the new AsyncOmni custom-pipeline TeaCache path, invalid metadata is reported with explicit errors and logs:

- Missing TeaCache hook while `cache_backend="tea_cache"`: raises `RuntimeError`.
- Invalid `teacache_step_index` / `teacache_num_denoise_steps` / `teacache_sde_window`: raises `TypeError` or `ValueError`.
- Inconsistent step metadata across one request: raises `ValueError`.

## Performance Tuning

Start with the default `rel_l1_thresh=0.2` and adjust based on your needs:

- **Maximum quality**: Use `0.1-0.2`
- **Balanced**: Use `0.2-0.4` (recommended)
- **Maximum speed**: Use `0.6-0.8` (may reduce quality)

## Troubleshooting

### Quality Degradation

If you notice quality issues, lower the threshold:

```python
cache_config={"rel_l1_thresh": 0.1}  # More conservative caching
```

## Supported Models

### ImageGen

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| Architecture | Models | Example HF Models |
|--------------|--------|-------------------|
| `QwenImagePipeline` | Qwen-Image | `Qwen/Qwen-Image` |
| `QwenImageEditPipeline` | Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` |
| `QwenImageLayeredPipeline` | Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` |
| `BagelForConditionalGeneration` | BAGEL (DiT-only) | `ByteDance-Seed/BAGEL-7B-MoT` |

### VideoGen

No VideoGen models are supported by TeaCache yet.

### Coming Soon

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| Architecture | Models | Example HF Models |
|--------------|--------|-------------------|
| `FluxPipeline` | Flux | - |
| `CogVideoXPipeline` | CogVideoX | - |
