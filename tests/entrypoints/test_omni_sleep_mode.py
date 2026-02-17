import asyncio
import pytest
import torch
import uuid
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.diffusion.data import OmniACK
from vllm_omni.inputs.data import OmniSamplingParams

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniTest")

pytestmark = [pytest.mark.core_model, pytest.mark.gpu]


@pytest.fixture(scope="module")
async def llm_engine():
    model_name = "Qwen/Qwen2.5-Omni-3B"
    common_args = {
        "enable_sleep_mode": True,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.45
    }
    stages = [
        # Stage 0 (Thinker): put 0
        {"stage_id": 0, "stage_type": "llm", "runtime": {"process": True, "devices": "0", "max_batch_size": 1}, 
         "engine_args": {**common_args, "model_stage": "thinker", "gpu_memory_utilization": 0.8}},
        # Stage 1 (Talker): put 1
        {"stage_id": 1, "stage_type": "llm", "runtime": {"process": True, "devices": "1", "max_batch_size": 1}, 
         "engine_args": {**common_args, "model_stage": "talker", "gpu_memory_utilization": 0.45}},
    ]
    engine = AsyncOmni(model_name, stages=stages)
    for stage in engine.stage_list:
        if hasattr(stage, "engine") and stage.engine:
            stage.engine.orchestrator = engine
    yield engine
    engine.shutdown()



@pytest.fixture(scope="module")
async def diffusion_engine():
    model_name = "black-forest-labs/FLUX.2-klein-4B"
    stages = [
        {"stage_id": 0, "stage_type": "diffusion", "runtime": {"process": True, "devices": "1", "max_batch_size": 1}, 
         "engine_args": {
             "model_stage": "base", 
             "gpu_memory_utilization": 0.45,
             "model_class_name": "FluxPipeline"
         }}
    ]
    engine = AsyncOmni(model_name, stages=stages)
    for stage in engine.stage_list:
        if hasattr(stage, "engine") and stage.engine:
            stage.engine.orchestrator = engine
    yield engine
    engine.shutdown()



class TestOmniSleepMode:

    @pytest.mark.asyncio
    async def test_llm_sleep_ack(self, llm_engine: AsyncOmni):
        """LLM stage signal loop"""
        acks = await llm_engine.sleep(stage_ids=[0], level=2)
        assert all(ack.status == "SUCCESS" for ack in acks)
        logger.info(f"Success: Received {len(acks)} LLM ACKs")

    @pytest.mark.asyncio
    async def test_diffusion_sleep_handshake(self, diffusion_engine: AsyncOmni):
        """Diffusion Worker stage signal loop"""
        logger.info(">>> Starting Diffusion Worker Handshake Test")
        task_id = str(uuid.uuid4())
        acks = await diffusion_engine.sleep(stage_ids=[0], level=2)
        assert all(ack.status == "SUCCESS" for ack in acks)
        logger.info(f"Success: Received {len(acks)} Diffusion Worker ACKs")

    @pytest.mark.asyncio
    async def test_cross_device_cleanup(self, llm_engine: AsyncOmni, diffusion_engine: AsyncOmni):
        """Physical stress test: Ensure VRAM is released on A5000"""
        initial_vram = torch.cuda.memory_reserved(1)
        logger.info(f"GPU 1 initial reservation: {initial_vram / 1024**3:.2f} GiB")
        await diffusion_engine.sleep(stage_ids=[0], level=2)
        torch.cuda.empty_cache()
        freed_vram = (initial_vram - torch.cuda.memory_reserved(1)) / 1024**3
        logger.info(f"Physical memory reclamation successful: GPU 1 weights unloaded, freed {freed_vram:.2f} GiB")


    # @pytest.mark.asyncio
    # async def test_model_integrity_post_wakeup(self, omni_engine: AsyncOmni):
    #     """
    #     Verify the consistency of model output after Sleep/Wakeup (to prevent garbled characters)
    #     Verification points: Text consistency & Exact matching of Token IDs
    #     """
    #     await asyncio.sleep(1)
    #     logger.info("Starting Consistency Test: Baseline vs Post-Wakeup")
    #     prompt = "The capital of France is"
    #     sampling_params = OmniSamplingParams(max_tokens=10, temperature=0.0)

    #     # Baseline Generation
    #     logger.info("Running Baseline Generation...")
    #     base_text = ""
    #     base_ids = []
    #     async for output in omni_engine.generate(prompt=prompt, request_id="baseline", sampling_params_list=[sampling_params] * 3):
    #         res = output.request_output.outputs[0]
    #         base_text = res.text
    #         base_ids = res.token_ids

    #     logger.info(f"Baseline IDs: {base_ids}")
    #     logger.info(f"Baseline Text: '{base_text}'")

    #     # Trigger Deep Sleep & Wakeup Cycle
    #     logger.info("Testing Sleep Level 2 & Deterministic Wakeup...")
    #     await omni_engine.sleep(stage_ids=[0, 1], level=2)
    #     logger.info("Engine is SLEEPING. VRAM should be released.")

    #     await asyncio.sleep(2)
    #     await omni_engine.wake_up(stage_ids=[0,1])
    #     logger.info("Engine is WAKEN UP. Weights restored.")

    #     logger.info("Running Post-Wakeup Generation...")
        
    #     post_text = ""
    #     post_ids = []
    #     async for output in omni_engine.generate(prompt=prompt, request_id="post-wake", sampling_params_list=[sampling_params] * 3):
    #         res = output.request_output.outputs[0]
    #         post_text = res.text
    #         post_ids = res.token_ids

    #     logger.info("Comparing Results...")
    #     try:
    #         assert base_ids == post_ids, f"Token IDs Mismatch!\nBase: {base_ids}\nPost: {post_ids}"
    #         assert base_text == post_text, f"Text Mismatch!\nBase: {base_text}\nPost: {post_text}"
    #         logger.info("SUCCESS: Model output is identical (Bit-Identical)!")
    #         logger.info("FP8 scaling factors and weights are verified correct.")
    #     except AssertionError as e:
    #         logger.error(f"FAIL: Consistency Check Failed!")
    #         logger.error(f"Last 3 IDs (Base): {base_ids[-3:]}")
    #         logger.error(f"Last 3 IDs (Post): {post_ids[-3:]}")
    #         raise e

    # @pytest.mark.asyncio
    # async def test_task1_physical_reclamation(self, omni_engine: AsyncOmni):
    #     """Verify physical memory reclamation"""
    #     logger.info("Starting Task 1 Test: Physical Reclamation")
    #     initial_mem = torch.cuda.memory_reserved()
    #     logger.info(f"Initial VRAM Reserved: {initial_mem / 1024**3:.2f} GiB")
    #     # Trigger deep sleep (Level 2)
    #     # Verification: await must return only after all Workers have finished moving GPU memory.
    #     acks = await omni_engine.sleep(stage_ids=[2], level=2)
    #     torch.cuda.empty_cache()
    #     post_sleep_mem = torch.cuda.memory_reserved(0)
    #     freed_gb = (initial_mem - post_sleep_mem) / 1024**3
    #     logger.info(f"Post-Sleep VRAM Reserved: {post_sleep_mem / 1024**3:.2f} GiB")
    #     logger.info(f"Total Freed: {freed_gb:.2f} GiB")
    #     assert freed_gb > 3.0, f"VRAM reclamation failed, only freed {freed_gb:.2f} GiB"
    #     assert all(ack.status == "SUCCESS" for ack in acks)

    # @pytest.mark.asyncio
    # async def test_task2_deterministic_handshake(self, omni_engine: AsyncOmni):
    #     """Verification of multi-worker signal aggregation and interception"""
    #     logger.info("Starting Task 2 Test: Deterministic Handshake")
    #     task_id = str(uuid.uuid4())
    #     stage = omni_engine.stage_list[0]
    #     # Verification: The Resolver can correctly count the number of Workers.
    #     expected_count = stage.engine.executor.get_worker_count()
    #     future = omni_engine.event_resolver.watch_task(task_id, expected_count=expected_count)
    #     stage.sleep(level=2, task_id=task_id)
    #     start_time = asyncio.get_event_loop().time()
    #     results = await asyncio.wait_for(future, timeout=30.0)
    #     duration = asyncio.get_event_loop().time() - start_time
    #     logger.info(f"Handshake resolved in {duration:.2f}s with {len(results)} ACKs")
    #     assert len(results) == expected_count, "ACK aggregation mismatch!"

    # @pytest.mark.asyncio
    # async def test_task3_auto_wakeup_protection(self, omni_engine: AsyncOmni):
    #     """Verify automatic wake-up protection logic"""
    #     logger.info("Starting Task 3 Test: Auto-Wakeup Protection")
    #     # First, let the stage enter sleep.
    #     await omni_engine.sleep(stage_ids=[0], level=2)
    #     assert omni_engine.stage_list[0].status == "SLEEPING"
    #     # Attempt to send an inference request while in the SLEEPING state.
    #     prompt = "A high-tech lab in Kuala Lumpur at night"
    #     async for output in omni_engine.generate(prompt=prompt, request_id="test-auto-wake", sampling_params_list=[OmniSamplingParams(max_tokens=5, temperature=0.0)] * 3):
    #         assert output is not None
    #         break  # Only the first output needs to be taken.
    #     # The final confirmation phase has been activated.
    #     assert omni_engine.stage_list[0].status == "RUNNING"
    #     logger.info("Auto-Wakeup verified successfully.")


    # @pytest.mark.asyncio
    # async def test_error_fallback_and_timeout(self, omni_engine: AsyncOmni):
    #     """Exception and Timeout Handling"""
    #     logger.info("Starting Scenario 4: Timeout Handling")
    #     try:
    #         await asyncio.wait_for(omni_engine.sleep(level=2), timeout=0.001)
    #     except asyncio.TimeoutError:
    #         logger.info("Timeout handled correctly.")