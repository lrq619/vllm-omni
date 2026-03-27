# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hook-based TeaCache implementation for vLLM-Omni.

This module implements a diffusers-style hook system that completely intercepts
the transformer forward pass, eliminating the need for any TeaCache-specific
code in model definitions. Model developers only need to add an extractor function
to support new models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import get_extractor
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, StateManager

logger = init_logger(__name__)


class TeaCacheHook(ModelHook):
    """
    ModelHook implementing TeaCache for transformer models.

    This hook completely intercepts the transformer's forward pass and implements
    adaptive caching based on timestep embedding similarity. It's model-agnostic
    and supports multiple model types through extractor functions.

    Key features:
    - Zero changes to model code
    - CFG-aware with separate states for positive/negative branches
    - CFG-parallel compatible: properly detects branch identity across ranks
    - Model-specific polynomial rescaling
    - Auto-detection of model types

    Attributes:
        config: TeaCache configuration with thresholds and callbacks
        rescale_func: Polynomial function for rescaling L1 distances
        state_manager: Manages TeaCacheState across forward passes
        extractor_fn: Model-specific function to extract modulated input
    """

    _HOOK_NAME = "teacache"

    def __init__(self, config: TeaCacheConfig):
        """
        Initialize TeaCacheHook.

        Args:
            config: TeaCache configuration object.
        """
        super().__init__()
        self.config = config
        self.rescale_func = np.poly1d(config.coefficients)
        self.state_manager = StateManager(TeaCacheState)
        self.extractor_fn = None
        self._forward_cnt = 0
        self._total_transformer_calls = 0
        self._executed_transformer_calls = 0
        self._skipped_transformer_calls = 0
        self._forced_compute_transformer_calls = 0
        self._skipped_transformer_calls_in_sde_window = 0
        self._seen_denoise_steps: set[int] = set()
        self._executed_denoise_steps: set[int] = set()
        self._skipped_denoise_steps: set[int] = set()
        self._forced_compute_denoise_steps_in_sde_window: set[int] = set()
        self._skipped_denoise_steps_in_sde_window: set[int] = set()
        self._num_denoise_steps_hint: int | None = None

    def _parse_step_metadata(
        self, kwargs: dict[str, Any]
    ) -> tuple[int | None, tuple[int, int] | None, bool]:
        step_index = kwargs.get("teacache_step_index")
        sde_window = kwargs.get("teacache_sde_window")
        force_no_skip_in_sde_window = bool(kwargs.get("teacache_no_skip_in_sde_window", False))
        num_denoise_steps = kwargs.get("teacache_num_denoise_steps")

        if step_index is not None:
            if not isinstance(step_index, int):
                logger.error("Invalid `teacache_step_index` type: expected int, got %s", type(step_index).__name__)
                raise TypeError(
                    f"Invalid `teacache_step_index`: expected int, got {type(step_index).__name__}."
                )
            if step_index < 0:
                logger.error("Invalid `teacache_step_index` value: expected >= 0, got %d", step_index)
                raise ValueError(f"Invalid `teacache_step_index`: expected >= 0, got {step_index}.")

        if num_denoise_steps is not None:
            if not isinstance(num_denoise_steps, int):
                logger.error(
                    "Invalid `teacache_num_denoise_steps` type: expected int, got %s",
                    type(num_denoise_steps).__name__,
                )
                raise TypeError(
                    f"Invalid `teacache_num_denoise_steps`: expected int, got {type(num_denoise_steps).__name__}."
                )
            if num_denoise_steps <= 0:
                logger.error(
                    "Invalid `teacache_num_denoise_steps` value: expected > 0, got %d",
                    num_denoise_steps,
                )
                raise ValueError(
                    f"Invalid `teacache_num_denoise_steps`: expected > 0, got {num_denoise_steps}."
                )
            if self._num_denoise_steps_hint is None:
                self._num_denoise_steps_hint = num_denoise_steps
            elif self._num_denoise_steps_hint != num_denoise_steps:
                logger.error(
                    "Inconsistent `teacache_num_denoise_steps`: previous=%d current=%d",
                    self._num_denoise_steps_hint,
                    num_denoise_steps,
                )
                raise ValueError(
                    "Inconsistent `teacache_num_denoise_steps` detected in one generation request: "
                    f"{self._num_denoise_steps_hint} vs {num_denoise_steps}."
                )
            if step_index is not None and step_index >= num_denoise_steps:
                logger.error(
                    "`teacache_step_index` out of range: step=%d num_denoise_steps=%d",
                    step_index,
                    num_denoise_steps,
                )
                raise ValueError(
                    f"Invalid `teacache_step_index`={step_index}: out of range for "
                    f"`teacache_num_denoise_steps`={num_denoise_steps}."
                )

        normalized_window = None
        if sde_window is not None:
            if not isinstance(sde_window, (tuple, list)) or len(sde_window) != 2:
                logger.error(
                    "Invalid `teacache_sde_window`: expected tuple/list of length 2, got %r",
                    sde_window,
                )
                raise TypeError(
                    "Invalid `teacache_sde_window`: expected tuple/list of length 2, "
                    f"got {sde_window!r}."
                )
            start, end = sde_window
            if not isinstance(start, int) or not isinstance(end, int):
                logger.error(
                    "Invalid `teacache_sde_window` element types: start=%s end=%s",
                    type(start).__name__,
                    type(end).__name__,
                )
                raise TypeError(
                    "Invalid `teacache_sde_window`: both start and end must be int, "
                    f"got {type(start).__name__} and {type(end).__name__}."
                )
            if start < 0 or end < start:
                logger.error(
                    "Invalid `teacache_sde_window` bounds: expected start>=0 and end>=start, got (%d, %d)",
                    start,
                    end,
                )
                raise ValueError(
                    f"Invalid `teacache_sde_window`: expected start>=0 and end>=start, got ({start}, {end})."
                )
            normalized_window = (start, end)
            if num_denoise_steps is not None and end > num_denoise_steps:
                logger.error(
                    "`teacache_sde_window` end out of range: end=%d num_denoise_steps=%d",
                    end,
                    num_denoise_steps,
                )
                raise ValueError(
                    f"Invalid `teacache_sde_window` end={end}: exceeds "
                    f"`teacache_num_denoise_steps`={num_denoise_steps}."
                )

        if force_no_skip_in_sde_window:
            if step_index is None:
                logger.error(
                    "`teacache_no_skip_in_sde_window=True` requires `teacache_step_index` in transformer kwargs."
                )
                raise ValueError(
                    "Missing `teacache_step_index` while `teacache_no_skip_in_sde_window=True`."
                )
            if normalized_window is None:
                logger.error(
                    "`teacache_no_skip_in_sde_window=True` requires `teacache_sde_window` in transformer kwargs."
                )
                raise ValueError(
                    "Missing `teacache_sde_window` while `teacache_no_skip_in_sde_window=True`."
                )

        return step_index, normalized_window, force_no_skip_in_sde_window

    @staticmethod
    def _in_sde_window(step_index: int, sde_window: tuple[int, int]) -> bool:
        return sde_window[0] <= step_index < sde_window[1]

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Initialize hook with extractor from config transformer model type.

        Args:
            module: The module to initialize the hook for.

        Returns:
            The initialized module.
        """
        # Get extractor function based on transformer_type from config
        # transformer_type is the transformer class name (e.g., "QwenImageTransformer2DModel")
        self.extractor_fn = get_extractor(self.config.transformer_type)

        # Set default context
        self.state_manager.set_context("teacache")

        return module

    def new_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        """
        Generic forward handler that works for ANY model.

        This method is completely model-agnostic. All model-specific logic
        is encapsulated in the extractor function that returns a CacheContext.

        The extractor does:
        - Model-specific preprocessing
        - Extraction of modulated input for cache decision
        - Providing transformer execution callable
        - Providing postprocessing callable

        This hook does:
        - CFG-aware state management
        - Cache decision logic (generic)
        - Residual caching and reuse

        Args:
            module: Transformer module (any architecture)
            *args: Positional arguments for model forward
            **kwargs: Keyword arguments for model forward

        Returns:
            Model output (format depends on model)
        """
        # Get model-specific context from extractor
        # The extractor encapsulates ALL model-specific logic
        ctx = self.extractor_fn(module, *args, **kwargs)

        # ============================================================================
        # GENERIC CACHING LOGIC (works for all models)
        # ============================================================================
        # Set context based on CFG branch for separate state tracking
        # With CFG-parallel, each rank processes only one branch:
        #   - cfg_rank 0: positive branch
        #   - cfg_rank > 0: negative branch
        # Without CFG-parallel, branches alternate within a single rank
        if getattr(module, "do_true_cfg", False):
            cfg_parallel_size = get_classifier_free_guidance_world_size()
            if cfg_parallel_size > 1:
                cfg_rank = get_classifier_free_guidance_rank()
                cache_branch = "negative" if cfg_rank > 0 else "positive"
            else:
                # No CFG-parallel: use forward counter to alternate branches
                cache_branch = "negative" if self._forward_cnt % 2 == 1 else "positive"
        else:
            cache_branch = "positive"

        context_name = f"teacache_{cache_branch}"
        self.state_manager.set_context(context_name)
        state = self.state_manager.get_state()

        step_index, sde_window, force_no_skip_in_sde_window = self._parse_step_metadata(kwargs)
        in_sde_window = (
            step_index is not None
            and sde_window is not None
            and self._in_sde_window(step_index, sde_window)
        )

        # Decide whether to compute or cache based on modulated input similarity.
        # If requested by caller, SDE-window steps are forced to compute.
        force_compute = bool(force_no_skip_in_sde_window and in_sde_window)
        if force_compute:
            should_compute = True
        else:
            should_compute = self._should_compute_full_transformer(state, ctx.modulated_input)

        self._total_transformer_calls += 1
        if step_index is not None:
            self._seen_denoise_steps.add(step_index)
        if force_compute:
            self._forced_compute_transformer_calls += 1
            if step_index is not None:
                self._forced_compute_denoise_steps_in_sde_window.add(step_index)

        if not should_compute and state.previous_residual is not None:
            # ============================================================================
            # FAST PATH: Reuse cached residuals
            # ============================================================================
            logger.info(
                "TEACACHE_STEP_SKIPPED transformer=%s branch=%s step_idx=%d "
                "accumulated_rel_l1_distance=%.8f rel_l1_thresh=%.8f",
                self.config.transformer_type,
                cache_branch,
                state.cnt,
                state.accumulated_rel_l1_distance,
                self.config.rel_l1_thresh,
            )
            ctx.hidden_states = ctx.hidden_states + state.previous_residual
            if state.previous_residual_encoder is not None and ctx.encoder_hidden_states is not None:
                ctx.encoder_hidden_states = ctx.encoder_hidden_states + state.previous_residual_encoder
            output = ctx.hidden_states
        else:
            # ============================================================================
            # SLOW PATH: Full transformer computation
            # ============================================================================
            ori_hidden_states = ctx.hidden_states.clone()
            ori_encoder_hidden_states = (
                ctx.encoder_hidden_states.clone() if ctx.encoder_hidden_states is not None else None
            )

            # Run transformer blocks using model-specific callable
            outputs = ctx.run_transformer_blocks()

            # Update context with outputs
            ctx.hidden_states = outputs[0]
            if len(outputs) > 1 and ctx.encoder_hidden_states is not None:
                ctx.encoder_hidden_states = outputs[1]

            # Cache residuals for next timestep
            state.previous_residual = (ctx.hidden_states - ori_hidden_states).detach()
            if ori_encoder_hidden_states is not None:
                state.previous_residual_encoder = (ctx.encoder_hidden_states - ori_encoder_hidden_states).detach()

            output = ctx.hidden_states

        did_skip = bool(not should_compute and state.previous_residual is not None)
        if did_skip:
            state.skip_calls += 1
            self._skipped_transformer_calls += 1
            if step_index is not None:
                self._skipped_denoise_steps.add(step_index)
            if in_sde_window:
                self._skipped_transformer_calls_in_sde_window += 1
                if step_index is not None:
                    self._skipped_denoise_steps_in_sde_window.add(step_index)
        else:
            state.compute_calls += 1
            self._executed_transformer_calls += 1
            if step_index is not None:
                self._executed_denoise_steps.add(step_index)

        # Update state
        state.previous_modulated_input = ctx.modulated_input.detach()
        state.cnt += 1
        self._forward_cnt += 1

        # ============================================================================
        # POSTPROCESSING (model-specific, via callable)
        # ============================================================================
        return ctx.postprocess(output)

    def _should_compute_full_transformer(self, state: TeaCacheState, modulated_inp: torch.Tensor) -> bool:
        """
        Determine whether to compute full transformer or reuse cached residual.

        This implements the core TeaCache algorithm:
        1. Always compute first timestep
        2. For intermediate steps:
           - Compute relative L1 distance between current and previous modulated inputs
           - Apply polynomial rescaling with model-specific coefficients
           - Accumulate rescaled distances
           - Compare to threshold: below = cache, above = compute

        Args:
            state: Current TeaCacheState containing counters and cached values
            modulated_inp: Modulated input extracted from first transformer block

        Returns:
            True to compute full transformer, False to reuse cached residual
        """
        # First timestep: always compute
        if state.cnt == 0:
            state.accumulated_rel_l1_distance = 0.0
            return True

        # Need previous input for comparison
        if state.previous_modulated_input is None:
            return True

        # Compute relative L1 distance between consecutive modulated inputs
        rel_distance = (
            (
                (modulated_inp - state.previous_modulated_input).abs().mean()
                / (state.previous_modulated_input.abs().mean() + 1e-8)
            )
            .cpu()
            .item()
        )

        # Apply model-specific polynomial rescaling
        rescaled_distance = float(self.rescale_func(rel_distance))
        state.accumulated_rel_l1_distance += abs(rescaled_distance)

        # Decision: below threshold = cache, above = compute
        if state.accumulated_rel_l1_distance < self.config.rel_l1_thresh:
            return False  # Use cache
        else:
            state.accumulated_rel_l1_distance = 0.0  # Reset accumulator
            return True  # Compute

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Reset all cached states for a new inference run.

        Args:
            module: The module to reset state for.

        Returns:
            The module with reset state.
        """
        self.state_manager.reset()
        self._forward_cnt = 0
        self._total_transformer_calls = 0
        self._executed_transformer_calls = 0
        self._skipped_transformer_calls = 0
        self._forced_compute_transformer_calls = 0
        self._skipped_transformer_calls_in_sde_window = 0
        self._seen_denoise_steps.clear()
        self._executed_denoise_steps.clear()
        self._skipped_denoise_steps.clear()
        self._forced_compute_denoise_steps_in_sde_window.clear()
        self._skipped_denoise_steps_in_sde_window.clear()
        self._num_denoise_steps_hint = None
        return module

    def get_metrics(self) -> dict[str, Any]:
        seen_steps = set(self._seen_denoise_steps)
        executed_steps = set(self._executed_denoise_steps)
        skipped_steps = set(self._skipped_denoise_steps)
        fully_skipped_steps = seen_steps - executed_steps

        return {
            "transformer_total_calls": self._total_transformer_calls,
            "transformer_executed_calls": self._executed_transformer_calls,
            "transformer_skipped_calls": self._skipped_transformer_calls,
            "transformer_forced_compute_calls_in_sde_window": self._forced_compute_transformer_calls,
            "transformer_skipped_calls_in_sde_window": self._skipped_transformer_calls_in_sde_window,
            "denoise_total_steps_hint": self._num_denoise_steps_hint,
            "denoise_seen_steps": len(seen_steps),
            "denoise_executed_steps": len(executed_steps),
            "denoise_steps_with_any_skip": len(skipped_steps),
            "denoise_fully_skipped_steps": len(fully_skipped_steps),
            "denoise_step_indices_seen": sorted(seen_steps),
            "denoise_step_indices_executed": sorted(executed_steps),
            "denoise_step_indices_with_any_skip": sorted(skipped_steps),
            "denoise_step_indices_fully_skipped": sorted(fully_skipped_steps),
            "denoise_step_indices_forced_compute_in_sde_window": sorted(
                self._forced_compute_denoise_steps_in_sde_window
            ),
            "denoise_step_indices_skipped_in_sde_window": sorted(self._skipped_denoise_steps_in_sde_window),
        }


def apply_teacache_hook(module: torch.nn.Module, config: TeaCacheConfig) -> None:
    """
    Apply TeaCache optimization to a transformer module.

    This function registers a TeaCacheHook that completely intercepts the
    module's forward pass, implementing adaptive caching without any changes
    to the model code.

    Args:
        module: Transformer model to optimize (e.g., QwenImageTransformer2DModel)
        config: TeaCacheConfig specifying caching parameters

    Example:
        >>> config = TeaCacheConfig(
        ...     rel_l1_thresh=0.2,
        ...     transformer_type="QwenImageTransformer2DModel"
        ... )
        >>> apply_teacache_hook(transformer, config)
        >>> # Transformer bound to the pipeline now uses TeaCache automatically,
        ... # no code changes needed!
    """
    registry = HookRegistry.get_or_create(module)
    hook = TeaCacheHook(config)
    registry.register_hook(TeaCacheHook._HOOK_NAME, hook)
