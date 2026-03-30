# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import pickle
import uuid
from collections.abc import Sequence

import torch
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.mooncake_store import MooncakeStore
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
from vllm_omni.outputs import OmniRequestOutput


class OmniDiffusion:
    """
    It is the main class to interact with vLLM-Omni diffusion models.
    It acts as a high-level interface that prepares requests and
    delegates the actual diffusion process to the DiffusionEngine.

    You can pass either an `OmniDiffusionConfig` via `od_config`, or
    pass kwargs such as `model="Qwen/Qwen-Image"`,
    which will be forwarded to `OmniDiffusionConfig.from_kwargs`.
    """

    def __init__(self, od_config: OmniDiffusionConfig | None = None, **kwargs):
        # Capture stage info from kwargs before they might be filtered out
        stage_id = kwargs.get("stage_id")
        engine_input_source = kwargs.get("engine_input_source")
        cfg_kv_collect_func = kwargs.pop("cfg_kv_collect_func", None)

        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(**kwargs)
        elif isinstance(od_config, dict):
            # If config is dict, check it too (priority to kwargs if both exist)
            if stage_id is None:
                stage_id = od_config.get("stage_id")
            if engine_input_source is None:
                engine_input_source = od_config.get("engine_input_source")
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config

        # Inject stage info into omni_kv_config if present
        if stage_id is not None:
            self.od_config.omni_kv_config.setdefault("stage_id", stage_id)
        if engine_input_source is not None:
            self.od_config.omni_kv_config.setdefault("engine_input_source", engine_input_source)

        # Detect model class and load config
        # Diffusers-style models expose `model_index.json` with `_class_name`.
        # Non-diffusers models (e.g. Bagel, NextStep, GLM-Image) only have `config.json`,
        # so we fall back to reading that and mapping model_type manually.
        try:
            config_dict = get_hf_file_to_dict(
                "model_index.json",
                od_config.model,
            )
            if config_dict is not None:
                if od_config.model_class_name is None:
                    od_config.model_class_name = config_dict.get("_class_name", None)
                od_config.update_multimodal_support()

                tf_config_dict = get_hf_file_to_dict(
                    "transformer/config.json",
                    od_config.model,
                )
                od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
            else:
                raise FileNotFoundError("model_index.json not found")
        except (AttributeError, OSError, ValueError, FileNotFoundError):
            cfg = get_hf_file_to_dict("config.json", od_config.model)
            if cfg is None:
                raise ValueError(f"Could not find config.json or model_index.json for model {od_config.model}")

            # Map model_type or architecture to pipeline class
            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures") or []
            pipeline_class = None
            # Bagel/NextStep models don't have a model_index.json, so we set the pipeline class name manually
            if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                pipeline_class = "BagelPipeline"
            elif model_type == "nextstep":
                if od_config.model_class_name is None:
                    pipeline_class = "NextStep11Pipeline"
            elif model_type == "glm-image" or "GlmImageForConditionalGeneration" in architectures:
                pipeline_class = "GlmImagePipeline"
            elif architectures and len(architectures) == 1:
                pipeline_class = architectures[0]

            if pipeline_class is None:
                raise ValueError(f"Unknown model type: {model_type}, architectures: {architectures}")

            if od_config.model_class_name is None:
                od_config.model_class_name = pipeline_class
            od_config.tf_model_config = TransformerConfig()
            od_config.update_multimodal_support()

        if cfg_kv_collect_func is not None:
            od_config.cfg_kv_collect_func = cfg_kv_collect_func

        self.engine: DiffusionEngine = DiffusionEngine.make_engine(od_config)
        self._mooncake_store: MooncakeStore | None = None

    def _get_mooncake_store(self) -> MooncakeStore:
        if MooncakeStore is None:
            raise ImportError("MooncakeStore is required for remote diffusion requests but is unavailable.")
        if self._mooncake_store is None:
            self._mooncake_store = MooncakeStore()
            self._mooncake_store.initialize()
        return self._mooncake_store

    @staticmethod
    def _strip_tensors_for_state(value):
        if isinstance(value, torch.Tensor):
            return None
        if isinstance(value, dict):
            return {k: OmniDiffusion._strip_tensors_for_state(v) for k, v in value.items()}
        if isinstance(value, list):
            return [OmniDiffusion._strip_tensors_for_state(v) for v in value]
        if isinstance(value, tuple):
            return tuple(OmniDiffusion._strip_tensors_for_state(v) for v in value)
        return value

    def _store_remote_request_payload(
        self,
        *,
        request_id: str,
        prompt: OmniPromptType,
    ) -> None:
        if not isinstance(prompt, dict):
            raise TypeError("Remote diffusion requests currently require dict-based prompts.")

        store = self._get_mooncake_store()
        remote_prompt = self._strip_tensors_for_state(copy.deepcopy(prompt))
        tensor_names: list[str] = []
        for tensor_name in ("prompt_ids", "prompt_mask", "negative_prompt_ids", "negative_prompt_mask"):
            tensor_value = prompt.get(tensor_name)
            if tensor_value is None:
                continue
            tensor_names.append(tensor_name)
            tensor = tensor_value if isinstance(tensor_value, torch.Tensor) else torch.as_tensor(tensor_value)
            store.put(f"{request_id}::{tensor_name}", tensor)

        state_payload = {
            "version": 1,
            "request_id": request_id,
            "tensor_names": tensor_names,
            "prompt": remote_prompt,
        }
        store.put(f"{request_id}::state", pickle.dumps(state_payload))

    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        request_ids: list[str] | None = None,
        is_remote_list: list[bool] | None = None,
    ) -> list[OmniRequestOutput]:
        if isinstance(prompts, (str, dict)):
            prompts = [prompts]
        else:
            prompts = list(prompts)

        request_ids = list(request_ids or [])
        if len(request_ids) < len(prompts):
            request_ids.extend(f"{i + len(request_ids)}_{uuid.uuid4()}" for i in range(len(prompts) - len(request_ids)))
        elif len(request_ids) > len(prompts):
            request_ids = request_ids[: len(prompts)]

        if is_remote_list is None:
            is_remote_list = [False for _ in prompts]
        else:
            is_remote_list = list(is_remote_list)
            if len(is_remote_list) != len(prompts):
                raise ValueError(
                    f"is_remote_list length mismatch: expected {len(prompts)}, got {len(is_remote_list)}"
                )

        if any(is_remote_list) and not self.od_config.enable_stepwise:
            raise ValueError("Remote diffusion requests require enable_stepwise=True.")

        for request_id, prompt, is_remote in zip(request_ids, prompts, is_remote_list, strict=True):
            if is_remote:
                logger.debug(
                    "Skipping remote payload staging for request_id=%s because remote requests resume from Mooncake state.",
                    request_id,
                )

        request = OmniDiffusionRequest(prompts, sampling_params, request_ids=request_ids, is_remote_list=is_remote_list)
        return self._run_engine(request)

    def _run_engine(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        return self.engine.step(request)

    def close(self) -> None:
        if self._mooncake_store is not None:
            try:
                self._mooncake_store.shutdown()
            except Exception:
                pass
            self._mooncake_store = None
        self.engine.close()

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def start_profile(self, trace_filename: str | None = None) -> None:
        """Start profiling for the diffusion model.

        Args:
            trace_filename: Optional base filename for trace files.
                           If None, a timestamp-based name will be generated.
        """
        if hasattr(self, "engine") and self.engine:
            self.engine.start_profile(trace_filename)
        else:
            raise RuntimeError("Diffusion engine not initialized")

    def stop_profile(self) -> dict:
        """Stop profiling and return profiling results.

        Returns:
            Dictionary containing paths to trace and table files.
        """
        if hasattr(self, "engine") and self.engine:
            return self.engine.stop_profile()
        else:
            raise RuntimeError("Diffusion engine not initialized")
