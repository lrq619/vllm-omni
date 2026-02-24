from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Any, Optional, Union

from pydantic import ConfigDict
from vllm.config import ModelConfig
from vllm.config.multimodal import MMCacheType, MMEncoderTPMode
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_omni.model_executor.models as me_models

logger = init_logger(__name__)


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class OmniModelConfig(ModelConfig):
    """Configuration for Omni models, extending the base ModelConfig.

    This configuration class extends the base vLLM ModelConfig with
    omni-specific fields for multi-stage pipeline processing.

    Attributes:
        stage_id: Identifier for the stage in a multi-stage pipeline (default: 0)
        async_chunk: If set to True, perform async chunk
        model_stage: Stage type identifier, e.g., "thinker" or "talker"
            (default: "thinker")
        model_arch: Model architecture name
            (default: "Qwen2_5OmniForConditionalGeneration")
        worker_type: Model Type, e.g., "ar" or "generation"
        engine_output_type: Optional output type specification for the engine.
            Used to route outputs to appropriate processors (e.g., "image",
            "audio", "latents"). If None, output type is inferred.
        stage_connector_config: Stage connector configuration dictionary.
            Contains "name" (connector name), "extra" (extra connector config).

    Example:
        >>> config = OmniModelConfig(
        ...     stage_id=0,
        ...     model_stage="thinker",
        ...     model_arch="Qwen2_5OmniForConditionalGeneration"
        ... )
    """

    stage_id: int = 0
    async_chunk: bool = False
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    worker_type: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_config: dict[str, Any] = None
    omni_kv_config: dict | None = None
    codec_frame_rate_hz: float | None = None

    mm_encoder_tp_mode: Any = "weights"

    video_pruning_rate: float = 0.99
    skip_mm_profiling: bool = True
    mm_encoder_attn_backend: Any = None

    @property
    def registry(self):
        return me_models.OmniModelRegistry

    @property
    def architectures(self) -> list[str]:
        return [self.model_arch]

    @property
    def embedding_size(self):
        if self.hf_config_name is not None:
            stage_config = getattr(self.hf_config, self.hf_config_name, None)
            override = getattr(stage_config, "embedding_size", None)
            if override is not None:
                return override
        return super().embedding_size

    def draw_hf_text_config(self):
        # transformers' get_text_config method is used to get the text config from thinker_config.
        # to handle the case that each model stage has their own text config,
        # we need to draw the text config from the corresponding model stage.
        if self.hf_config_name is None:
            return get_hf_text_config(self.hf_config)
        try:
            # Try to get the stage-specific config (e.g., thinker_config, talker_config)
            stage_config = getattr(self.hf_config, self.hf_config_name)
            return stage_config.get_text_config()
        except AttributeError:
            # Fallback: if the attribute doesn't exist, use the default get_hf_text_config
            logger.warning(
                f"Config attribute '{self.hf_config_name}' not found in hf_config, "
                "falling back to default get_hf_text_config"
            )
            return get_hf_text_config(self.hf_config)

    def __post_init__(
        self, *args, **kwargs, 
    ) -> None:
        if self.stage_connector_config is None:
            self.stage_connector_config = {
                "name": "SharedMemoryConnector",
                "extra": {},
            }
        def extract_var(idx, key, default=None):
            return kwargs.get(key, getattr(self, key, default))
        
        limit_mm_per_prompt   = extract_var(0, "limit_mm_per_prompt")
        if isinstance(limit_mm_per_prompt, bool) or limit_mm_per_prompt is None:
            limit_mm_per_prompt = {}
        enable_mm_embeds      = extract_var(1, "enable_mm_embeds")
        media_io_kwargs       = extract_var(2, "media_io_kwargs")
        mm_processor_kwargs   = extract_var(3, "mm_processor_kwargs")
        mm_processor_cache_gb = extract_var(4, "mm_processor_cache_gb")
        mm_processor_cache_type = extract_var(5, "mm_processor_cache_type")
        mm_shm_cache_max_object_size_mb = extract_var(6, "mm_shm_cache_max_object_size_mb")
        mm_encoder_only       = extract_var(7, "mm_encoder_only", False)
        mm_encoder_tp_mode    = extract_var(8, "mm_encoder_tp_mode", "weights")
        if mm_encoder_tp_mode not in ["weights", "data"]:
            mm_encoder_tp_mode = "weights"
        self.mm_encoder_tp_mode = mm_encoder_tp_mode
        mm_encoder_attn_backend = extract_var(9, "mm_encoder_attn_backend")
        interleave_mm_strings = extract_var(10, "interleave_mm_strings", True)
        skip_mm_profiling     = extract_var(11, "skip_mm_profiling", True)
        video_pruning_rate    = extract_var(12, "video_pruning_rate", 0.99)
        if mm_encoder_attn_backend is False:
            mm_encoder_attn_backend = None
        self.mm_encoder_attn_backend = mm_encoder_attn_backend
        self.skip_mm_profiling = bool(skip_mm_profiling)
        self.video_pruning_rate = float(video_pruning_rate) if video_pruning_rate is not None else 0.99

        # Keep set served_model_name before maybe_model_redirect(self.model)
        self.served_model_name = get_served_model_name(self.model, self.served_model_name)
        self.model = maybe_model_redirect(self.model)
        # The tokenizer is consistent with the model by default.
        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.tokenizer_revision is None:
            self.tokenizer_revision = self.revision
        self.tokenizer = maybe_model_redirect(self.tokenizer)

        if isinstance(self.hf_config_path, str):
            self.hf_config_path = maybe_model_redirect(self.hf_config_path)

        if callable(self.hf_overrides):
            hf_overrides_kw = {}
            hf_overrides_fn = self.hf_overrides
            dict_overrides: dict[str, Any] = {}
        else:
            # Separate dict overrides from flat ones
            # We'll determine how to apply dict overrides after loading the config
            hf_overrides_kw = {}
            dict_overrides = {}
            for key, value in self.hf_overrides.items():
                if isinstance(value, dict):
                    dict_overrides[key] = value
                else:
                    hf_overrides_kw[key] = value
            hf_overrides_fn = None

        self.maybe_pull_model_tokenizer_for_runai(self.model, self.tokenizer)

        if self.override_attention_dtype is not None and not current_platform.is_rocm():
            warnings.warn(
                "override-attention-dtype is set but not using ROCm platform",
                stacklevel=2,
            )

        if self.enable_sleep_mode and not current_platform.is_sleep_mode_available():
            raise ValueError("Sleep mode is not supported on current platform.")

        hf_config = get_config(
            self.hf_config_path or self.model,
            self.trust_remote_code,
            self.revision,
            self.code_revision,
            self.config_format,
            hf_overrides_kw=hf_overrides_kw,
            hf_overrides_fn=hf_overrides_fn,
        )
        hf_config = maybe_patch_hf_config_from_gguf(
            self.model,
            hf_config,
        )
        # Call parent's __post_init__ to handle all standard ModelConfig initialization
        super().__post_init__(
            limit_mm_per_prompt=limit_mm_per_prompt,
            enable_mm_embeds=enable_mm_embeds,
            media_io_kwargs=media_io_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
            mm_processor_cache_gb=mm_processor_cache_gb,
            mm_processor_cache_type=mm_processor_cache_type,
            mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
            mm_encoder_only=mm_encoder_only,
            mm_encoder_tp_mode=mm_encoder_tp_mode,
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            interleave_mm_strings=interleave_mm_strings,
            skip_mm_profiling=skip_mm_profiling,
            video_pruning_rate=video_pruning_rate,
        )

        # Qwen3-TTS: infer codec frame rate from the model config for online serving.
        if self.codec_frame_rate_hz is None and self.model_arch == "Qwen3TTSTalkerForConditionalGenerationARVLLM":
            talker_cfg = getattr(self.hf_config, "talker_config", None)
            if isinstance(talker_cfg, dict):
                pos_per_sec = talker_cfg.get("position_id_per_seconds")
            else:
                pos_per_sec = getattr(talker_cfg, "position_id_per_seconds", None)
            if pos_per_sec is not None:
                try:
                    fps = float(pos_per_sec)
                except Exception:
                    fps = None
                if fps is not None and fps > 0:
                    self.codec_frame_rate_hz = fps

        if self.convert == "mm_encoder_only":
            logger.warning_once(
                "`--convert mm_encoder_only` is deprecated and "
                "will be removed in v0.15. "
                "Please use --mm-encoder-only` instead."
            )
            mm_encoder_only = True
            self.convert = "none"

        architectures = self.architectures
        registry = self.registry
        is_generative_model = registry.is_text_generation_model(architectures, self)
        is_pooling_model = registry.is_pooling_model(architectures, self)

        self.runner_type = self._get_runner_type(architectures, self.runner)
        self.convert_type = self._get_convert_type(architectures, self.runner_type, self.convert)

        if self.runner_type == "generate" and not is_generative_model:
            generate_converts = _RUNNER_CONVERTS["generate"]
            if self.convert_type not in generate_converts:
                # Currently we don't have any converters for generative models
                raise ValueError("This model does not support `--runner generate`.")
        if self.runner_type == "pooling" and not is_pooling_model:
            pooling_converts = _RUNNER_CONVERTS["pooling"]
            if self.convert_type not in pooling_converts:
                convert_option = "<" + "|".join(pooling_converts) + ">"
                raise ValueError(
                    "This model does not support `--runner pooling`. "
                    f"You can pass `--convert {convert_option} to adapt "
                    "it into a pooling model."
                )

        # Note: Initialize these attributes early because transformers fallback
        # may fail to load dynamic modules in child processes
        model_info, arch = registry.inspect_model_cls(architectures, self)
        self._model_info = model_info
        self._architecture = arch
        logger.info("Resolved architecture: %s", arch)

        # Init pooler config if needed
        if self.runner_type == "pooling":
            if self.pooler_config is None:
                self.pooler_config = PoolerConfig()

            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                # Only set values that are not overridden by the user
                for k, v in base_config.items():
                    if getattr(self.pooler_config, k) is None:
                        setattr(self.pooler_config, k, v)

            default_seq_pooling_type = self._model_info.default_seq_pooling_type
            if self.pooler_config.seq_pooling_type is None:
                self.pooler_config.seq_pooling_type = default_seq_pooling_type
            default_tok_pooling_type = self._model_info.default_tok_pooling_type
            if self.pooler_config.tok_pooling_type is None:
                self.pooler_config.tok_pooling_type = default_tok_pooling_type

        self.dtype: torch.dtype = _get_and_verify_dtype(
            self.model,
            self.hf_config,
            self.dtype,
            is_pooling_model=self.runner_type == "pooling",
            revision=self.revision,
        )

        self.original_max_model_len = self.max_model_len
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)

        if self.is_encoder_decoder:
            self.mm_processor_cache_gb = 0
            logger.info("Encoder-decoder model detected, disabling mm processor cache.")

        # Init multimodal config if needed
        if self._model_info.supports_multimodal:
            if mm_encoder_tp_mode == "data" and not self._model_info.supports_multimodal_encoder_tp_data:
                logger.warning_once(
                    "This model does not support `--mm-encoder-tp-mode data`. "
                    "Falling back to `--mm-encoder-tp-mode weights`."
                )
                mm_encoder_tp_mode = "weights"

            mm_config_kwargs = dict(
                limit_per_prompt=limit_mm_per_prompt,
                enable_mm_embeds=enable_mm_embeds,
                media_io_kwargs=media_io_kwargs,
                mm_processor_kwargs=mm_processor_kwargs,
                mm_processor_cache_gb=mm_processor_cache_gb,
                mm_processor_cache_type=mm_processor_cache_type,
                mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
                mm_encoder_only=mm_encoder_only,
                mm_encoder_tp_mode=mm_encoder_tp_mode,
                mm_encoder_attn_backend=mm_encoder_attn_backend,
                interleave_mm_strings=interleave_mm_strings,
                skip_mm_profiling=skip_mm_profiling,
                video_pruning_rate=video_pruning_rate,
            )

            mm_config_kwargs = {k: v for k, v in mm_config_kwargs.items() if v is not None}

            self.multimodal_config = MultiModalConfig(**mm_config_kwargs)

            if not hasattr(self.multimodal_config, "mm_encoder_only"):
                setattr(self.multimodal_config, "mm_encoder_only", mm_encoder_only)
            if not hasattr(self.multimodal_config, "video_pruning_rate"):
                setattr(self.multimodal_config, "video_pruning_rate", video_pruning_rate)

        # Multimodal GGUF models must use original repo for mm processing
        if is_gguf(self.tokenizer) and self.is_multimodal_model:
            raise ValueError(
                "Loading a multimodal GGUF model needs to use original "
                "tokenizer. Please specify the unquantized hf model's "
                "repo name or path using the --tokenizer argument."
            )

        if self.disable_sliding_window:
            # Set after get_and_verify_max_len to ensure that max_model_len
            # can be correctly capped to sliding window size
            self.hf_text_config.sliding_window = None

        # Avoid running try_verify_and_update_config multiple times
        self.config_updated = False
        self._try_verify_and_update_model_config()
        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()
        # Override hf_text_config with omni-specific logic for multi-stage models
        # (e.g., thinker_config, talker_config)
        new_hf_text_config = self.draw_hf_text_config()
        if new_hf_text_config is not self.hf_text_config:
            self.hf_text_config = new_hf_text_config
            # Recalculate dependent attributes
            self.attention_chunk_size = getattr(self.hf_text_config, "attention_chunk_size", None)
            # Recalculate max_model_len since it depends on hf_text_config
            self.max_model_len = self.get_and_verify_max_len(self.original_max_model_len)
            # Reset sliding_window if needed
            if self.disable_sliding_window:
                self.hf_text_config.sliding_window = None
