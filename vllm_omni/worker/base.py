"""Base worker class for vLLM-Omni with process-scoped GPU memory accounting."""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib, memory_profiling
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from vllm_omni.worker.gpu_memory_utils import (
    get_process_gpu_memory,
)

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniDiffusionConfig,
    OmniACK,
    OmniSleepTask,
    OmniWakeTask,
)

from vllm_omni.platforms import current_omni_platform
from contextlib import AbstractContextManager, nullcontext

logger = init_logger(__name__)


class OmniGPUWorkerBase(GPUWorker):
    """Base GPU worker for vLLM-Omni with process-scoped memory accounting.

    This class overrides determine_available_memory() to use per-process GPU
    memory tracking via pynvml, allowing multiple stages to initialize
    concurrently on the same GPU without memory accounting interference.
    """

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Process-scoped GPU memory profiling for concurrent stage initialization.

        Algorithm:
            1. requested_memory = total_gpu_memory * gpu_memory_utilization
               (computed in init_device from cache_config)

            2. process_memory = memory used by THIS process only (via pynvml)
               - Uses nvmlDeviceGetComputeRunningProcesses to get per-PID memory
               - Supports CUDA_VISIBLE_DEVICES with indices, UUIDs, or MIG IDs

            3. available_kv_cache = requested_memory - process_memory

        Fallback:
            If NVML is unavailable, falls back to profiling data:
            available = requested - (weights + activations + non_torch)
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            logger.info(
                "Using explicit kv_cache_memory_bytes: %s GiB",
                format_gib(kv_cache_memory_bytes),
            )
            return kv_cache_memory_bytes

        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        process_memory = get_process_gpu_memory(self.local_rank)

        if process_memory is not None:
            # NVML available: use per-process memory
            self.available_kv_cache_memory_bytes = max(0, self.requested_memory - process_memory)
            logger.debug(
                "Process-scoped memory (PID %d, GPU %d): requested=%s, used=%s, available=%s",
                os.getpid(),
                self.local_rank,
                format_gib(self.requested_memory),
                format_gib(process_memory),
                format_gib(self.available_kv_cache_memory_bytes),
            )
            logger.info_once(
                "Available KV cache memory: %s GiB (process-scoped)",
                format_gib(self.available_kv_cache_memory_bytes),
                scope="local",
            )
        else:
            # NVML unavailable: use profiling data as conservative fallback
            profiled_usage = (
                int(self.model_runner.model_memory_usage)
                + profile_result.torch_peak_increase
                + profile_result.non_torch_increase
            )
            self.available_kv_cache_memory_bytes = max(0, self.requested_memory - profiled_usage)
            logger.debug(
                "Profiling fallback (PID %d, GPU %d): requested=%s, profiled=%s, available=%s",
                os.getpid(),
                self.local_rank,
                format_gib(self.requested_memory),
                format_gib(profiled_usage),
                format_gib(self.available_kv_cache_memory_bytes),
            )
            logger.info_once(
                "Available KV cache memory: %s GiB (profiling fallback)",
                format_gib(self.available_kv_cache_memory_bytes),
                scope="local",
            )

        return int(self.available_kv_cache_memory_bytes)


    def load_model(self):
        """Identify LLM weights"""
        with self._maybe_get_memory_pool_context("weights"):
            super().load_model()

    # Provide memory pool context
    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # enable sleep mode
        if getattr(self.cache_config, "enable_sleep_mode", False) or \
           getattr(self.model_config, "enable_sleep_mode", False):
            from vllm.device_allocator.cumem import CuMemAllocator
            allocator = CuMemAllocator.get_instance()
            return allocator.use_memory_pool(tag=tag)
        return nullcontext()


    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep.
        Args:
            level: 1 (Offload weights to CPU), level: 2 (Total Discard).
        """
        from vllm.device_allocator.cumem import CuMemAllocator
        mem_before = current_omni_platform.get_current_memory_usage(self.device)
        offload_tags = ("weights") if level == 1 else tuple()
        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=offload_tags)
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()
        mem_after = current_omni_platform.get_current_memory_usage(self.device)
        freed = max(0, mem_before - mem_after)
        remaining_gb = mem_after / 1024**3
        logger.info(f"[LLM Worker {self.rank}] Level {level} Sleep: Freed {freed / 1024**3:.2f} GiB. {remaining_gb:.2f}GiB memory is still in use.")
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        "Physical video memory reloading logic"
        from vllm.device_allocator.cumem import CuMemAllocator
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)
        current_omni_platform.synchronize()
        logger.info(f"[LLM Worker {self.rank}] Wake-up complete.")
        return True

    def handle_sleep_task(self, task: OmniSleepTask) -> OmniACK:
        "Handle deterministic Sleep command from the main process"
        try:
            if isinstance(task, dict):
                task = OmniSleepTask(**task)

            logger.info(f"[LLM Worker {self.rank}] Handshake Received: Task {task.task_id}, Level {task.level}")
            if task.level >= 2:
                if hasattr(self.model_runner, "graph_runners"):
                    self.model_runner.graph_runners.clear()
                    logger.info(f"[LLM Worker {self.rank}] LLM CUDA Graphs cleared.")

            mem_before = current_omni_platform.get_current_memory_usage(self.device)
            self.sleep(level=task.level)
            mem_after = current_omni_platform.get_current_memory_usage(self.device)
            real_freed = max(0, mem_before - mem_after)
            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)

            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=current_stage_id,
                rank=self.rank,
                freed_bytes=real_freed,
                metadata={"vram_after": mem_after}
            )
            if hasattr(self, "result_mq") and self.result_mq:
                self.result_mq.put(ack)
            logger.info(f"[LLM Worker {self.rank}] ACK emitted for Task {task.task_id}")
            return ack

        except Exception as e:
            logger.error(f"[LLM Worker {self.rank}] Sleep Task Failed: {e}")
            return OmniACK(task_id=task.task_id, status="ERROR", error_msg=str(e))

    def handle_wake_task(self, task: OmniWakeTask) -> OmniACK:
        "Handle deterministic Wakeup command from the main process"
        try:
            if isinstance(task, dict):
                task = OmniWakeTask(**task)
            self.wake_up(tags=task.tags)
            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            ack = OmniACK(task_id=task.task_id, status="SUCCESS", stage_id=current_stage_id, rank=self.rank)
            if hasattr(self, "result_mq") and self.result_mq:
                self.result_mq.put(ack)
            logger.info(f"[LLM Worker {self.rank}] Wake-up ACK emitted.")
            return ack
        except Exception as e:
            tid = task.get("task_id", "unknown") if isinstance(task, dict) else getattr(task, "task_id", "unknown")
            logger.error(f"[LLM Worker {self.rank}] Wake-up Failed: {e}")
            return OmniACK(task_id=tid, status="ERROR", error_msg=str(e))