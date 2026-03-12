# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStage:
    def __init__(self, stage_type: str):
        self.stage_type = stage_type
        self.calls: list[dict] = []

    def update_weights_from_ipc(self, **kwargs) -> None:
        self.calls.append({"method": "update_weights_from_ipc", **kwargs})

    def add_lora(self, **kwargs) -> None:
        self.calls.append({"method": "add_lora", **kwargs})

    def remove_lora(self, **kwargs) -> None:
        self.calls.append({"method": "remove_lora", **kwargs})

    def list_loras(self, **kwargs) -> None:
        self.calls.append({"method": "list_loras", **kwargs})

    def pin_lora(self, **kwargs) -> None:
        self.calls.append({"method": "pin_lora", **kwargs})


class _FakeLoop:
    def __init__(self):
        self.calls: list[tuple] = []

    async def run_in_executor(self, executor, func, *args):
        self.calls.append((executor, func, args))
        return func(*args)


class _FakeEngine:
    def __init__(self):
        self.calls: list[tuple] = []

    def collective_rpc(self, *args):
        self.calls.append(args)
        return "rpc-ok"


@pytest.mark.asyncio
async def test_async_omni_update_weights_targets_diffusion_stages_only():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni._name = "test-omni"
    omni.stage_list = [_FakeStage("llm"), _FakeStage("diffusion"), _FakeStage("diffusion")]
    omni._run_output_handler = lambda: None

    captured: dict[str, object] = {}
    future = asyncio.get_running_loop().create_future()
    future.set_result(
        [
            {"stage_id": 1, "result": "stage-1"},
            {"stage_id": 2, "result": "stage-2"},
        ]
    )

    def _watch(task_id: str, expected_stage_ids: list[int]):
        captured["task_id"] = task_id
        captured["expected_stage_ids"] = expected_stage_ids
        return future

    omni._watch_stage_rpc_task = _watch

    result = await AsyncOmni.update_weights_from_ipc(
        omni,
        peft_config={"adapter": "demo"},
        base_sync_done=True,
        use_shm=True,
    )

    assert result == ["stage-1", "stage-2"]
    assert captured["expected_stage_ids"] == [1, 2]
    assert omni.stage_list[0].calls == []
    for stage in omni.stage_list[1:]:
        assert len(stage.calls) == 1
        assert stage.calls[0]["method"] == "update_weights_from_ipc"
        assert stage.calls[0]["peft_config"] == {"adapter": "demo"}
        assert stage.calls[0]["base_sync_done"] is True
        assert stage.calls[0]["use_shm"] is True
        assert stage.calls[0]["task_id"] == captured["task_id"]


@pytest.mark.asyncio
async def test_async_omni_update_weights_rejects_non_diffusion_stage_ids():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni._name = "test-omni"
    omni.stage_list = [_FakeStage("llm"), _FakeStage("diffusion")]
    omni._run_output_handler = lambda: None

    with pytest.raises(ValueError, match="only supports diffusion stages"):
        await AsyncOmni.update_weights_from_ipc(omni, stage_ids=[0])


@pytest.mark.asyncio
async def test_async_omni_list_loras_targets_diffusion_stages_only_and_merges_results():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni._name = "test-omni"
    omni.stage_list = [_FakeStage("llm"), _FakeStage("diffusion"), _FakeStage("diffusion")]
    omni._run_output_handler = lambda: None

    captured: dict[str, object] = {}
    future = asyncio.get_running_loop().create_future()
    future.set_result(
        [
            {"stage_id": 1, "result": [1, 2]},
            {"stage_id": 2, "result": [2, 3]},
        ]
    )

    def _watch(task_id: str, expected_stage_ids: list[int]):
        captured["task_id"] = task_id
        captured["expected_stage_ids"] = expected_stage_ids
        return future

    omni._watch_stage_rpc_task = _watch

    result = await AsyncOmni.list_loras(omni)

    assert result == [1, 2, 3]
    assert captured["expected_stage_ids"] == [1, 2]
    assert omni.stage_list[0].calls == []
    for stage in omni.stage_list[1:]:
        assert len(stage.calls) == 1
        assert stage.calls[0]["method"] == "list_loras"
        assert stage.calls[0]["task_id"] == captured["task_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "call", "expected_method"),
    [
        (
            "add_lora",
            lambda omni: AsyncOmni.add_lora(omni, lora_request=object(), lora_scale=0.5),
            "add_lora",
        ),
        (
            "remove_lora",
            lambda omni: AsyncOmni.remove_lora(omni, adapter_id=7),
            "remove_lora",
        ),
        (
            "pin_lora",
            lambda omni: AsyncOmni.pin_lora(omni, lora_id=9),
            "pin_lora",
        ),
    ],
)
async def test_async_omni_lora_rpcs_broadcast_to_diffusion_stages(method_name, call, expected_method):
    omni = AsyncOmni.__new__(AsyncOmni)
    omni._name = "test-omni"
    omni.stage_list = [_FakeStage("llm"), _FakeStage("diffusion"), _FakeStage("diffusion")]
    omni._run_output_handler = lambda: None

    future = asyncio.get_running_loop().create_future()
    future.set_result(
        [
            {"stage_id": 1, "result": True},
            {"stage_id": 2, "result": True},
        ]
    )
    omni._watch_stage_rpc_task = lambda task_id, expected_stage_ids: future

    result = await call(omni)

    assert result is True
    assert omni.stage_list[0].calls == []
    for stage in omni.stage_list[1:]:
        assert len(stage.calls) == 1
        assert stage.calls[0]["method"] == expected_method


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "method_name"),
    [
        (lambda omni: AsyncOmni.list_loras(omni, stage_ids=[0]), "list_loras"),
        (lambda omni: AsyncOmni.add_lora(omni, lora_request=object(), stage_ids=[0]), "add_lora"),
        (lambda omni: AsyncOmni.remove_lora(omni, adapter_id=1, stage_ids=[0]), "remove_lora"),
        (lambda omni: AsyncOmni.pin_lora(omni, lora_id=1, stage_ids=[0]), "pin_lora"),
    ],
)
async def test_async_omni_lora_rpcs_reject_non_diffusion_stage_ids(call, method_name):
    omni = AsyncOmni.__new__(AsyncOmni)
    omni._name = "test-omni"
    omni.stage_list = [_FakeStage("llm"), _FakeStage("diffusion")]
    omni._run_output_handler = lambda: None

    with pytest.raises(ValueError, match=f"{method_name} only supports diffusion stages"):
        await call(omni)


@pytest.mark.asyncio
async def test_async_omni_lora_rpcs_raise_on_stage_error():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni._name = "test-omni"
    omni.stage_list = [_FakeStage("diffusion")]
    omni._run_output_handler = lambda: None

    future = asyncio.get_running_loop().create_future()
    future.set_result([{"stage_id": 0, "error": "boom"}])
    omni._watch_stage_rpc_task = lambda task_id, expected_stage_ids: future

    with pytest.raises(RuntimeError, match="Stage 0 add_lora failed: boom"):
        await AsyncOmni.add_lora(omni, lora_request=object())


@pytest.mark.asyncio
async def test_async_omni_rpc_result_resolver_waits_for_all_stages():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni._name = "test-omni"
    omni._pending_stage_rpc_tasks = {}

    future = AsyncOmni._watch_stage_rpc_task(omni, "task-1", [3, 5])
    await AsyncOmni._resolve_stage_rpc_result(omni, {"task_id": "task-1", "stage_id": 3, "result": "a"})
    assert not future.done()

    await AsyncOmni._resolve_stage_rpc_result(omni, {"task_id": "task-1", "stage_id": 5, "result": "b"})
    assert future.done()
    assert future.result() == [
        {"task_id": "task-1", "stage_id": 3, "result": "a"},
        {"task_id": "task-1", "stage_id": 5, "result": "b"},
    ]


@pytest.mark.asyncio
async def test_async_omni_diffusion_update_weights_uses_single_reply_rpc(monkeypatch):
    fake_loop = _FakeLoop()
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: fake_loop)

    diffusion = AsyncOmniDiffusion.__new__(AsyncOmniDiffusion)
    diffusion._executor = object()
    diffusion.engine = _FakeEngine()

    result = await AsyncOmniDiffusion.update_weights_from_ipc(
        diffusion,
        peft_config={"adapter": "demo"},
        base_sync_done=True,
        use_shm=False,
    )

    assert result == "rpc-ok"
    assert len(fake_loop.calls) == 1
    assert diffusion.engine.calls == [
        (
            "update_weights_from_ipc",
            None,
            (),
            {"peft_config": {"adapter": "demo"}, "base_sync_done": True, "use_shm": False},
            0,
            True,
        )
    ]


@pytest.mark.asyncio
async def test_async_omni_diffusion_lora_task_handlers_forward_arguments(monkeypatch):
    diffusion = AsyncOmniDiffusion.__new__(AsyncOmniDiffusion)

    forwarded: list[tuple[str, tuple, dict]] = []

    async def _add_lora(*args, **kwargs):
        forwarded.append(("add_lora", args, kwargs))
        return True

    async def _remove_lora(*args, **kwargs):
        forwarded.append(("remove_lora", args, kwargs))
        return True

    async def _list_loras(*args, **kwargs):
        forwarded.append(("list_loras", args, kwargs))
        return [1, 3]

    async def _pin_lora(*args, **kwargs):
        forwarded.append(("pin_lora", args, kwargs))
        return True

    monkeypatch.setattr(diffusion, "add_lora", _add_lora)
    monkeypatch.setattr(diffusion, "remove_lora", _remove_lora)
    monkeypatch.setattr(diffusion, "list_loras", _list_loras)
    monkeypatch.setattr(diffusion, "pin_lora", _pin_lora)

    lora_request = object()
    assert await AsyncOmniDiffusion.handle_add_lora_task(
        diffusion, {"lora_request": lora_request, "lora_scale": 0.25}
    )
    assert await AsyncOmniDiffusion.handle_remove_lora_task(diffusion, {"adapter_id": 4})
    assert await AsyncOmniDiffusion.handle_list_loras_task(diffusion, {})
    assert await AsyncOmniDiffusion.handle_pin_lora_task(diffusion, {"lora_id": 5})

    assert forwarded == [
        ("add_lora", (), {"lora_request": lora_request, "lora_scale": 0.25}),
        ("remove_lora", (), {"adapter_id": 4}),
        ("list_loras", (), {}),
        ("pin_lora", (), {"lora_id": 5}),
    ]
