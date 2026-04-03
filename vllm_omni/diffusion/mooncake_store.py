# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_hostname_ip() -> str:
    try:
        process = subprocess.run(["hostname", "-i"], capture_output=True, text=True, check=True)
        return process.stdout.strip()
    except Exception:
        return "localhost"


class MooncakeStore:
    def __init__(
        self,
        storage_host: Optional[str] = None,
        storage_port: Optional[int] = None,
        master_host: Optional[str] = None,
        master_port: Optional[int] = None,
        protocol: str = "tcp",
        node_addr: Optional[str] = None,
        local_buffer_size: int = 17179869184 * 2,
    ) -> None:
        self.storage_host = storage_host or os.environ.get("MOONCAKE_STORAGE_HOST", _get_hostname_ip())
        self.storage_port = storage_port or int(os.environ.get("MOONCAKE_STORAGE_PORT", "10619"))
        self.master_host = master_host or os.environ.get("MOONCAKE_MASTER_HOST", self.storage_host)
        self.master_port = master_port or int(os.environ.get("MOONCAKE_MASTER_PORT", "50051"))
        self.protocol = protocol or os.environ.get("MOONCAKE_PROTOCOL", "tcp")
        self.node_addr = node_addr or os.environ.get("MOONCAKE_NODE_ADDR", _get_hostname_ip())
        self.local_buffer_size = local_buffer_size
        # self.segment_size_gb = int(os.environ.get("MOONCAKE_SEGMENT_SIZE_GB", "256"))
        self.segment_size_gb = 256
        if self.segment_size_gb <= 0:
            raise ValueError(
                f"Invalid MOONCAKE_SEGMENT_SIZE_GB={self.segment_size_gb}; expected a positive integer."
            )
        self.segment_size_bytes = self.segment_size_gb * (1024**3)
        self.initialized = False
        self.storage_client = None
        logger.info(
            "MooncakeStore initialized with storage=%s:%s master=%s:%s protocol=%s node=%s "
            "local_buffer_size=%s segment_size_gb=%s segment_size_bytes=%s",
            self.storage_host,
            self.storage_port,
            self.master_host,
            self.master_port,
            self.protocol,
            self.node_addr,
            self.local_buffer_size,
            self.segment_size_gb,
            self.segment_size_bytes,
        )

    def initialize(self, max_retries: int = 10, retry_delay: float = 2.0) -> None:
        if self.initialized:
            logger.warning("MooncakeStore already initialized")
            return

        from mooncake.store import MooncakeDistributedStore

        self.storage_client = MooncakeDistributedStore()
        retry_count = 0
        while retry_count < max_retries:
            status = self.storage_client.setup(
                self.node_addr,
                f"http://{self.storage_host}:{self.storage_port}/metadata",
                self.segment_size_bytes,
                self.local_buffer_size,
                self.protocol,
                "",
                f"{self.master_host}:{self.master_port}",
            )
            if status == 0:
                self.initialized = True
                try:
                    if os.environ.get("MOONCAKE_CLEAR_ON_INITIALIZE", "0") == "1":
                        self.storage_client.remove_all()
                        time.sleep(3)
                        logger.info("MooncakeStore cleared existing keys on initialize.")
                except Exception:
                    pass
                logger.info("MooncakeStore initialized successfully")
                return
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)

        raise RuntimeError(
            f"Failed to connect to Mooncake server after {max_retries} attempts. "
            "Please check if server is running and connection parameters are correct."
        )

    def put(self, key: str, data: bytes | bytearray | memoryview | Tensor, pin_memory: bool = True) -> int:
        if not self.initialized:
            raise RuntimeError("MooncakeStore not initialized. Call initialize() first.")

        if isinstance(data, (bytes, bytearray, memoryview)):
            payload = bytes(data)
            result = self.storage_client.put(key, payload)
            if result != 0:
                raise RuntimeError(f"Failed to store bytes payload '{key}', error code: {result}")
            return result

        tensor = data
        if tensor.device.type != "cpu":
            cpu_tensor = tensor.cpu()
            if pin_memory:
                cpu_tensor = cpu_tensor.pin_memory()
        else:
            cpu_tensor = tensor
            if pin_memory and not tensor.is_pinned():
                cpu_tensor = cpu_tensor.pin_memory()

        result = self.storage_client.put_tensor(key, cpu_tensor)
        if result != 0:
            raise RuntimeError(f"Failed to store tensor '{key}', error code: {result}")
        return result

    def get_bytes(
        self,
        key: str,
    ) -> bytes:
        if not self.initialized:
            raise RuntimeError("MooncakeStore not initialized. Call initialize() first.")

        payload = self.storage_client.get(key)
        if payload is None:
            raise RuntimeError(f"Missing bytes payload '{key}'")
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, bytearray):
            return bytes(payload)
        if isinstance(payload, memoryview):
            return payload.tobytes()
        raise RuntimeError(f"Expected bytes payload '{key}', got {type(payload).__name__}")

    def get_tensor(
        self,
        key: str,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
    ) -> Tensor:
        if not self.initialized:
            raise RuntimeError("MooncakeStore not initialized. Call initialize() first.")

        cpu_tensor = self.storage_client.get_tensor(key)
        if cpu_tensor is None:
            raise RuntimeError(f"Missing tensor payload '{key}'")
        if device is None:
            device = torch.device("cpu")
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cpu":
            return cpu_tensor
        return cpu_tensor.to(device, non_blocking=non_blocking)

    def delete(self, key: str) -> int:
        if not self.initialized:
            raise RuntimeError("MooncakeStore not initialized. Call initialize() first.")
        result = self.storage_client.remove(key)
        if result != 0 and result != -706:
            raise RuntimeError(f"Failed to delete key '{key}', error code: {result}")
        return result

    def has_key(self, key: str) -> bool:
        if not self.initialized:
            return False
        return self.storage_client.is_exist(key) == 1

    def clear(self) -> None:
        if not self.initialized:
            return
        result = self.storage_client.remove_all()
        if result != 0:
            raise RuntimeError(f"Failed to clear MooncakeStore, error code: {result}")

    def remove_all(self) -> None:
        if not self.initialized:
            return
        result = self.storage_client.remove_all()
        if result != 0:
            raise RuntimeError(f"Failed to remove all MooncakeStore keys, error code: {result}")

    def get_stats(self) -> Dict[str, Union[int, float]]:
        raise NotImplementedError("MooncakeStore.get_stats no longer tracks local key metadata.")

    def shutdown(self) -> None:
        if not self.initialized:
            return
        try:
            self.clear()
        except Exception:
            pass
        if self.storage_client is not None:
            try:
                self.storage_client.close()
            except Exception:
                pass
            self.storage_client = None
        self.initialized = False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
