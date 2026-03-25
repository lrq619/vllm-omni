# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from unittest.mock import Mock

import pytest
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.diffusion.stepwise_scheduler import StepwiseScheduler
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.omni_stage import _build_od_config

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def _make_serve_parser() -> argparse.ArgumentParser:
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)
    return parser


def test_cli_enable_stepwise_flag_builds_od_config():
    parser = _make_serve_parser()
    args = parser.parse_args(["serve", "dummy-model", "--omni", "--enable_step_wise"])

    assert args.enable_stepwise is True

    engine_args = {
        "enable_stepwise": args.enable_stepwise,
        "model_class_name": "QwenImagePipelineWithLogProbStep",
    }
    od_config_dict = _build_od_config(engine_args, model="dummy-model")
    assert od_config_dict["enable_stepwise"] is True
    assert od_config_dict["model_class_name"] == "QwenImagePipelineWithLogProbStep"


def test_python_enable_stepwise_selects_stepwise_scheduler(monkeypatch):
    def _fake_scheduler_init(self, od_config):
        self.num_workers = 0
        self.mq = Mock(enqueue=lambda *_: None)
        self.result_mq = Mock()
        self._started = True

    monkeypatch.setattr(StepwiseScheduler, "initialize", _fake_scheduler_init)
    monkeypatch.setattr(StepwiseScheduler, "get_broadcast_handle", lambda self: "broadcast-handle")
    monkeypatch.setattr(StepwiseScheduler, "initialize_result_queue", lambda self, handle: None)
    monkeypatch.setattr(StepwiseScheduler, "close", lambda self: None)
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_launch_workers", lambda self, handle: ([], "result-handle"))

    od_config = OmniDiffusionConfig(
        model="dummy-model",
        num_gpus=1,
        enable_stepwise=True,
        model_class_name="QwenImagePipelineWithLogProbStep",
    )

    executor = MultiprocDiffusionExecutor(od_config)
    assert isinstance(executor.scheduler, StepwiseScheduler)
    assert executor.od_config.model_class_name == "QwenImagePipelineWithLogProbStep"
    assert executor.od_config.enable_stepwise is True


def test_python_disable_stepwise_selects_default_scheduler(monkeypatch):
    def _fake_scheduler_init(self, od_config):
        self.num_workers = 0
        self.mq = Mock(enqueue=lambda *_: None)
        self.result_mq = Mock()

    monkeypatch.setattr(Scheduler, "initialize", _fake_scheduler_init)
    monkeypatch.setattr(Scheduler, "get_broadcast_handle", lambda self: "broadcast-handle")
    monkeypatch.setattr(Scheduler, "initialize_result_queue", lambda self, handle: None)
    monkeypatch.setattr(Scheduler, "close", lambda self: None)
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_launch_workers", lambda self, handle: ([], "result-handle"))

    od_config = OmniDiffusionConfig(model="dummy-model", num_gpus=1, enable_stepwise=False)
    executor = MultiprocDiffusionExecutor(od_config)
    assert isinstance(executor.scheduler, Scheduler)
    assert executor.od_config.enable_stepwise is False


def test_workerproc_create_worker_selects_stepwise_worker_class(monkeypatch):
    captured = {}

    class _FakeWrapper:
        def __init__(self, **kwargs):
            captured["base_worker_class"] = kwargs["base_worker_class"]
            self.worker = None

    monkeypatch.setattr("vllm_omni.diffusion.worker.diffusion_worker.WorkerWrapperBase", _FakeWrapper)

    proc = WorkerProc.__new__(WorkerProc)
    od_config = OmniDiffusionConfig(
        model="dummy-model",
        num_gpus=1,
        enable_stepwise=True,
        model_class_name="QwenImagePipelineWithLogProbStep",
    )
    proc._create_worker(
        gpu_id=0,
        od_config=od_config,
        worker_extension_cls=None,
        custom_pipeline_args=None,
    )
    assert captured["base_worker_class"].__name__ == "DiffusionStepwiseWorker"
