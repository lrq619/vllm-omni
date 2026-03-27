# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import math
import os
import pickle
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

from vllm_omni.diffusion.mooncake_store import MooncakeStore, _get_hostname_ip

pytestmark = [pytest.mark.diffusion]

_DEFAULT_STORAGE_PORT = int(os.getenv("MOONCAKE_STORAGE_PORT", "10619"))
_DEFAULT_MASTER_PORT = int(os.getenv("MOONCAKE_MASTER_PORT", "50051"))
_DEFAULT_METRICS_PORT = int(os.getenv("MOONCAKE_METRICS_PORT", "49999"))


def _require_mooncake_runtime() -> None:
    if importlib.util.find_spec("mooncake") is None:
        pytest.skip("Requires mooncake Python package.")
    if shutil.which("mooncake_master") is None:
        pytest.skip("Requires mooncake_master binary in PATH.")


def _wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


@contextmanager
def _ensure_mooncake_master(
    host: str,
    *,
    storage_port: int,
    master_port: int,
    metrics_port: int,
) -> Iterator[subprocess.Popen[str] | None]:
    rpc_ready = _port_is_open(host, master_port)
    http_ready = _port_is_open(host, storage_port)
    if rpc_ready and http_ready:
        yield None
        return
    if rpc_ready or http_ready:
        pytest.skip(
            "Mooncake default ports are partially occupied. "
            "Please free them or override MOONCAKE_STORAGE_PORT / MOONCAKE_MASTER_PORT."
        )

    proc = subprocess.Popen(
        [
            "mooncake_master",
            f"--rpc_port={master_port}",
            "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            f"--http_metadata_server_port={storage_port}",
            f"--metrics_port={metrics_port}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
        text=True,
    )
    try:
        assert _wait_for_port(host, master_port), "mooncake_master RPC port failed to come up."
        assert _wait_for_port(host, storage_port), "mooncake_master HTTP metadata port failed to come up."
        yield proc
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass


def _tensor_shape_from_env() -> tuple[int, ...]:
    shape_text = os.getenv("VLLM_OMNI_MOONCAKE_LATENTS_SHAPE", "1,16,1024,1024")
    dims = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())
    if not dims:
        raise ValueError("VLLM_OMNI_MOONCAKE_LATENTS_SHAPE must contain at least one dimension.")
    return dims


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_mooncake_store_gpu_put_get_across_devices():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 GPUs.")

    _require_mooncake_runtime()

    store_host = os.getenv("MOONCAKE_STORAGE_HOST", _get_hostname_ip())
    tensor_shape = _tensor_shape_from_env()
    request_id = "migration-req-0"
    state_key = f"{request_id}::state"
    tensor_key = f"{request_id}::latents"

    with _ensure_mooncake_master(
        store_host,
        storage_port=_DEFAULT_STORAGE_PORT,
        master_port=_DEFAULT_MASTER_PORT,
        metrics_port=_DEFAULT_METRICS_PORT,
    ):
        store = MooncakeStore()
        store.initialize(max_retries=30, retry_delay=1.0)

        try:
            state_payload = {
                "request_id": request_id,
                "tensor_pool_names": ["latents"],
                "shape": list(tensor_shape),
            }
            state_bytes = pickle.dumps(state_payload)

            put_state_ret = store.put(state_key, state_bytes)
            assert put_state_ret == 0, f"put(state) failed with code={put_state_ret}"

            src = torch.arange(
                math.prod(tensor_shape),
                dtype=torch.float16,
                device="cuda:0",
            ).reshape(tensor_shape)
            torch.cuda.synchronize(0)

            put_tensor_ret = store.put(tensor_key, src)
            assert put_tensor_ret == 0, f"put_tensor(latents) failed with code={put_tensor_ret}"

            restored_state = store.get(state_key, device="cpu")
            assert isinstance(restored_state, bytes)
            assert restored_state == state_bytes

            restored = store.get(tensor_key, device="cuda:1", non_blocking=False)
            torch.cuda.synchronize(1)

            assert isinstance(restored, torch.Tensor)
            assert restored.device.type == "cuda"
            assert restored.device.index == 1
            assert restored.shape == src.shape
            assert restored.dtype == src.dtype
            assert torch.equal(restored.cpu(), src.cpu())
        finally:
            store.shutdown()
