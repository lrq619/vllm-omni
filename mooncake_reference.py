"""
Mooncake Store Manager for distributed KV cache and model parameter offloading.

Mooncake is a distributed key-value store optimized for GPU-CPU data transfers,
providing high-throughput offloading for model parameters and KV cache.

Uses the mooncake.store.MooncakeDistributedStore API.
"""
import os
import subprocess
import time
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from roll.utils.logging import get_logger
from roll.platforms import current_platform


def get_hostname_ip_subprocess_run():
    """Get hostname IP address using subprocess."""
    try:
        process = subprocess.run(
            ["hostname", "-i"],
            capture_output=True,
            text=True,
            check=True
        )
        ip_addresses = process.stdout.strip()
        return ip_addresses
    except FileNotFoundError:
        logger.warning("Error: 'hostname' Not Found, falling back to 'localhost'")
        return "localhost"
    except subprocess.CalledProcessError:
        logger.warning("Error: Run 'hostname -i' Failed, falling back to 'localhost'")
        return "localhost"
    except Exception as e:
        logger.warning(f"Error getting hostname: {e}, falling back to 'localhost'")
        return "localhost"




logger = get_logger()


class MooncakeStore:
    """
    Manager for Mooncake distributed store for model offloading.

    Provides high-performance offloading of:
    - Model parameters (weights, optimizer states)
    - KV cache (for inference engines like vLLM)
    - Gradient buffers
    """

    def __init__(
        self,
        storage_host: Optional[str] = None,
        storage_port: Optional[int] = None,
        master_host: Optional[str] = None,
        master_port: Optional[int] = None,
        protocol: str = "tcp",
        node_addr: Optional[str] = None,
        local_buffer_size: int = 17179869184 * 2,  # 32GB default
    ):
        """
        Initialize Mooncake store.

        Args:
            storage_host: Storage server hostname
            storage_port: Storage server metadata port (default: 12345)
            master_host: Master service hostname (default: same as storage_host)
            master_port: Master service port (default: 50051)
            protocol: Transfer protocol ('tcp' or 'rdma')
            node_addr: This node's address (default: 'localhost')
            local_buffer_size: Maximum size for read/write operations (default: 32GB)
        """
        self.storage_host = storage_host or os.environ.get("MOONCAKE_STORAGE_HOST", get_hostname_ip_subprocess_run())
        self.storage_port = storage_port or int(os.environ.get("MOONCAKE_STORAGE_PORT", "8083"))
        self.master_host = master_host or os.environ.get("MOONCAKE_MASTER_HOST", self.storage_host)
        self.master_port = master_port or int(os.environ.get("MOONCAKE_MASTER_PORT", "50051"))
        self.protocol = protocol or os.environ.get("MOONCAKE_PROTOCOL", "tcp")
        self.node_addr = node_addr or os.environ.get("MOONCAKE_NODE_ADDR", get_hostname_ip_subprocess_run())
        self.local_buffer_size = local_buffer_size
        logger.info(
            "MooncakeStore initialized with: "
            f"storage={self.storage_host}:{self.storage_port}, "
            f"master={self.master_host}:{self.master_port}, "
            f"protocol={self.protocol}, node={self.node_addr}, "
            f"local_buffer_size={self.local_buffer_size}"
            f"protocol={self.protocol}"
        )

        self.initialized = False
        self.storage_client = None
        self._cached_keys: Dict[str, Tuple] = {}  # key -> (shape, dtype)

    def initialize(self, max_retries: int = 10, retry_delay: float = 2.0) -> None:
        """
        Initialize Mooncake distributed store client.

        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
        """
        if self.initialized:
            logger.warning("MooncakeStore already initialized")
            return

        try:
            from mooncake.store import MooncakeDistributedStore

            logger.info(
                f"Initializing MooncakeStore: "
                f"storage={self.storage_host}:{self.storage_port}, "
                f"master={self.master_host}:{self.master_port}, "
                f"protocol={self.protocol}, node={self.node_addr}"
            )

            # Create Mooncake client
            self.storage_client = MooncakeDistributedStore()

            # Try to connect with retries
            retry_count = 0
            while retry_count < max_retries:
                status = self.storage_client.setup(
                    self.node_addr,                                      # Your node's address
                    f"http://{self.storage_host}:{self.storage_port}/metadata",  # HTTP metadata server
                    0,                                                   # segment size is zero means it is client
                    self.local_buffer_size,                             # local buffer: max size for read/write
                    self.protocol,                                       # Use TCP (or RDMA for high performance)
                    "",                                                  # Leave empty; Mooncake auto-picks RDMA devices
                    f"{self.master_host}:{self.master_port}"           # Master service
                )

                if status == 0:
                    self.initialized = True
                    logger.info("MooncakeStore initialized successfully")

                    # Clean up any existing data
                    try:
                        self.storage_client.remove_all()
                        self._cached_keys.clear()  # Clear cached keys after remove_all
                        logger.info("Cleaned up existing Mooncake data")
                        time.sleep(3)  # Sleep to ensure cleanup is fully processed before use
                    except:
                        pass  # Ignore cleanup errors

                    return

                retry_count += 1
                logger.warning(
                    f"Mooncake connection attempt {retry_count}/{max_retries} failed (status={status})"
                )
                if retry_count < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

            raise RuntimeError(
                f"Failed to connect to Mooncake server after {max_retries} attempts. "
                f"Please check if server is running and connection parameters are correct."
            )

        except ImportError as e:
            logger.error(
                f"Failed to import mooncake.store. "
                f"Please install Mooncake package. Error: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MooncakeStore: {e}")
            raise

    def put(
        self,
        key: str,
        tensor: Tensor,
        pin_memory: bool = True,
    ) -> int:
        """
        Store a tensor in Mooncake store.

        Args:
            key: Unique identifier for the tensor
            tensor: PyTorch tensor to store
            pin_memory: Pin memory for faster CPU-GPU transfers

        Returns:
            0 on success, non-zero on failure
        """
        if not self.initialized:
            raise RuntimeError("MooncakeStore not initialized. Call initialize() first.")

        # Convert to CPU tensor if needed
        if tensor.device.type != "cpu":
            cpu_tensor = tensor.cpu()
            if pin_memory:
                cpu_tensor = cpu_tensor.pin_memory()
        else:
            cpu_tensor = tensor
            if pin_memory and not tensor.is_pinned():
                cpu_tensor = cpu_tensor.pin_memory()

        # Store metadata for later retrieval
        self._cached_keys[key] = (tuple(tensor.shape), tensor.dtype)

        # Store torch.Tensor directly in Mooncake
        # The storage client can handle torch.Tensor natively
        result = self.storage_client.put_tensor(key, cpu_tensor)

        if result == 0:
            logger.info(f"Stored tensor '{key}' with shape {tensor.shape} and dtype {tensor.dtype}")
        else:
            logger.error(f"Failed to store tensor '{key}', error code: {result}")

        return result


    def get(
        self,
        key: str,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
    ) -> Optional[Tensor]:
        """
        Retrieve a tensor from Mooncake store.

        Args:
            key: Unique identifier for the tensor
            device: Target device (GPU or CPU)
            non_blocking: Enable non-blocking GPU transfer

        Returns:
            Retrieved tensor, or None if not found
        """
        if not self.initialized:
            raise RuntimeError("MooncakeStore not initialized. Call initialize() first.")

        if key not in self._cached_keys:
            raise KeyError(f"Key '{key}' not found in MooncakeStore")

        # Get torch.Tensor directly from Mooncake
        # The storage client returns torch.Tensor natively
        cpu_tensor = self.storage_client.get_tensor(key)

        if cpu_tensor is None:
            logger.warning(f"Key '{key}' not found in Mooncake store but in self._cached_keys. It may have been removed.")
            return None

        # Move to target device if specified
        if device is None:
            device = current_platform.device_type

        if isinstance(device, str):
            device = torch.device(device)

        if device.type == 'cpu':
            return cpu_tensor
        else:
            return cpu_tensor.to(device, non_blocking=non_blocking)

    def delete(self, key: str) -> int:
        """
        Delete a tensor from Mooncake store.

        Args:
            key: Unique identifier for the tensor

        Returns:
            0 on success, non-zero on failure
        """
        if not self.initialized:
            raise RuntimeError("MooncakeStore not initialized. Call initialize() first.")

        result = 0
        if key in self._cached_keys:
            result = self.storage_client.remove(key)
            # Error -706 means key not found, treat as success (idempotent delete)
            if result == 0 or result == -706:
                del self._cached_keys[key]
                logger.debug(f"Deleted tensor '{key}' from MooncakeStore")
                result = 0  # Normalize -706 to success
            else:
                logger.error(f"Failed to delete tensor '{key}', error code: {result}")

        return result

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the store."""
        if not self.initialized:
            return False
        # Use Mooncake's is_exist method
        result = self.storage_client.is_exist(key)
        return result == 1

    def clear(self) -> None:
        """Clear all stored tensors."""
        if not self.initialized:
            return

        # Remove all keys individually
        for key in list(self._cached_keys.keys()):
            result = self.storage_client.remove(key)
            # Ignore key not found errors (-706) - already deleted
            if result != 0 and result != -706:
                logger.warning(f"Failed to remove key '{key}' during clear, error code: {result}")

        self._cached_keys.clear()

        logger.info("Cleared all tensors from MooncakeStore")

    def remove_all(self) -> None:
        if not self.initialized:
            return
        self.storage_client.remove_all()
        self._cached_keys.clear()

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get storage statistics."""
        if not self.initialized or self.storage_client is None:
            return {}

        # Calculate total size
        total_size_bytes = 0
        for shape, dtype in self._cached_keys.values():
            numel = 1
            for dim in shape:
                numel *= dim
            total_size_bytes += numel * dtype.itemsize

        stats = {
            "num_keys": len(self._cached_keys),
            "total_size_mb": total_size_bytes / 1024 / 1024,
            "total_size_gb": total_size_bytes / 1024 / 1024 / 1024,
        }
        return stats

    def shutdown(self) -> None:
        """Shutdown Mooncake store and release resources."""
        if not self.initialized:
            return

        logger.info("Shutting down MooncakeStore")

        # Clear data and close connection
        try:
            self.clear()
        except:
            pass

        # Close the store connection
        if self.storage_client is not None:
            try:
                self.storage_client.close()
            except:
                pass
            self.storage_client = None

        self.initialized = False

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


# Global singleton instance
_global_mooncake_store: Optional[MooncakeStore] = None


def get_mooncake_store(
    storage_host: Optional[str] = None,
    storage_port: Optional[int] = None,
    master_host: Optional[str] = None,
    master_port: Optional[int] = None,
    protocol: str = "tcp",
    node_addr: Optional[str] = None,
    local_buffer_size: int = 17179869184 * 2,
    force_reinit: bool = False,
) -> MooncakeStore:
    """
    Get or create the global MooncakeStore instance.

    Args:
        storage_host: Storage server hostname
        storage_port: Storage server metadata port
        master_host: Master service hostname
        master_port: Master service port
        protocol: Transfer protocol ('tcp' or 'rdma')
        node_addr: This node's address
        local_buffer_size: Maximum size for read/write operations
        force_reinit: Force reinitialization

    Returns:
        Global MooncakeStore instance
    """
    global _global_mooncake_store

    if _global_mooncake_store is None or force_reinit:
        if _global_mooncake_store is not None:
            _global_mooncake_store.shutdown()

        _global_mooncake_store = MooncakeStore(
            storage_host=storage_host,
            storage_port=storage_port,
            master_host=master_host,
            master_port=master_port,
            protocol=protocol,
            node_addr=node_addr,
            local_buffer_size=local_buffer_size,
        )
        _global_mooncake_store.initialize()

    return _global_mooncake_store
