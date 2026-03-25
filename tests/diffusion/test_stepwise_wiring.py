# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import Mock

import pytest
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.diffusion.stepwise_scheduler import StepwiseScheduler
from vllm_omni.entrypoints import utils as utils_module
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.omni_stage import _build_od_config

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


@contextmanager
def _stub_modules(modules: dict[str, ModuleType]):
    prev_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, prev in prev_modules.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _make_openai_stub_modules() -> dict[str, ModuleType]:
    openai_pkg_stub = ModuleType("vllm_omni.entrypoints.openai")
    openai_pkg_stub.__path__ = []  # type: ignore[attr-defined]
    api_server_stub = ModuleType("vllm_omni.entrypoints.openai.api_server")
    api_server_stub.omni_run_server = lambda *args, **kwargs: None
    api_server_stub.build_async_omni = lambda *args, **kwargs: None
    api_server_stub.omni_init_app_state = lambda *args, **kwargs: None
    return {
        "vllm_omni.entrypoints.openai": openai_pkg_stub,
        "vllm_omni.entrypoints.openai.api_server": api_server_stub,
    }


def _make_vllm_openai_cli_args_stub_modules() -> dict[str, ModuleType]:
    cli_args_stub = ModuleType("vllm.entrypoints.openai.cli_args")

    def _make_arg_parser(parser):
        parser.add_argument("model_tag", nargs="?")
        return parser

    cli_args_stub.make_arg_parser = _make_arg_parser
    cli_args_stub.validate_parsed_serve_args = lambda args: None
    return {"vllm.entrypoints.openai.cli_args": cli_args_stub}


def _make_offloader_stub_modules() -> dict[str, ModuleType]:
    offloader_pkg_stub = ModuleType("vllm_omni.diffusion.offloader")
    offloader_pkg_stub.__path__ = []  # type: ignore[attr-defined]
    offloader_pkg_stub.get_offload_backend = lambda *args, **kwargs: None
    offloader_pkg_stub.LayerWiseOffloadBackend = object
    return {"vllm_omni.diffusion.offloader": offloader_pkg_stub}


def _make_serve_parser() -> argparse.ArgumentParser:
    with _stub_modules({**_make_openai_stub_modules(), **_make_vllm_openai_cli_args_stub_modules()}):
        from vllm_omni.entrypoints.cli.serve import OmniServeCommand

    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)
    return parser


def test_cli_enable_stepwise_flag_builds_od_config():
    parser = _make_serve_parser()
    args = parser.parse_args(["serve", "dummy-model", "--omni", "--enable_step_wise", "--max-step-batch-size", "12"])

    assert args.enable_stepwise is True
    assert args.max_step_batch_size == 12

    engine_args = {
        "enable_stepwise": args.enable_stepwise,
        "max_step_batch_size": args.max_step_batch_size,
        "model_class_name": "QwenImagePipelineWithLogProbStep",
    }
    od_config_dict = _build_od_config(engine_args, model="dummy-model")
    assert od_config_dict["enable_stepwise"] is True
    assert od_config_dict["max_step_batch_size"] == 12
    assert od_config_dict["model_class_name"] == "QwenImagePipelineWithLogProbStep"


def test_max_step_batch_size_wiring_paths(monkeypatch):
    with _stub_modules(_make_offloader_stub_modules()):
        from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

    parser = _make_serve_parser()
    args = parser.parse_args(["serve", "dummy-model", "--omni", "--enable_step_wise", "--max-step-batch-size", "12"])
    assert args.max_step_batch_size == 12

    monkeypatch.setattr(utils_module, "get_hf_file_to_dict", lambda filename, model, revision=None: {"_class_name": "QwenImagePipelineWithLogProbStep"} if filename == "model_index.json" else {})
    monkeypatch.setattr(utils_module, "load_stage_configs_from_model", lambda model, base_engine_args=None: [])
    monkeypatch.setattr(utils_module, "resolve_model_config_path", lambda model: None)
    monkeypatch.setattr(AsyncOmni, "_start_stages", lambda self, model: None)
    monkeypatch.setattr(AsyncOmni, "_wait_for_stages_ready", lambda self, timeout=0: None)

    omni = AsyncOmni(
        model="dummy-model",
        enable_stepwise=True,
        max_step_batch_size=12,
    )
    stage_cfg = omni.stage_configs[0]
    assert stage_cfg.engine_args["max_step_batch_size"] == 12

    captured = {}

    def _fake_scheduler_init(self, od_config):
        captured["max_step_batch_size"] = od_config.max_step_batch_size
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
        max_step_batch_size=12,
        model_class_name="QwenImagePipelineWithLogProbStep",
    )
    executor = MultiprocDiffusionExecutor(od_config)
    assert captured["max_step_batch_size"] == 12
    assert executor.od_config.max_step_batch_size == 12


def test_python_enable_stepwise_selects_stepwise_scheduler(monkeypatch):
    with _stub_modules(_make_offloader_stub_modules()):
        from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

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
    with _stub_modules(_make_offloader_stub_modules()):
        from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

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
    with _stub_modules(_make_offloader_stub_modules()):
        from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

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
