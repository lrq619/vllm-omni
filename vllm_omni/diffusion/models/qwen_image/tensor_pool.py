import logging
from collections.abc import Sequence

import torch

logger = logging.getLogger(__name__)


class TensorPool:
    def __init__(
        self,
        max_bsz: int,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if max_bsz <= 0:
            logger.error("Invalid `max_bsz`: expected > 0, got %d.", max_bsz)
            raise ValueError(f"Invalid `max_bsz`: expected > 0, got {max_bsz}.")
        if len(shape) == 0:
            logger.error("Invalid `shape`: expected non-empty shape, got %s.", tuple(shape))
            raise ValueError(f"Invalid `shape`: expected non-empty shape, got {tuple(shape)}.")
        if any(int(s) <= 0 for s in shape):
            logger.error("Invalid `shape`: all dims must be > 0, got %s.", tuple(shape))
            raise ValueError(f"Invalid `shape`: all dims must be > 0, got {tuple(shape)}.")

        self.max_bsz = int(max_bsz)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.data = torch.empty((self.max_bsz, *self.shape), dtype=self.dtype, device=self.device)

    def _validate_indicies(self, indicies: list[int]) -> list[int]:
        if not isinstance(indicies, list):
            logger.error(
                "Invalid `indicies`: expected list[int], got type=%s, value=%s.",
                type(indicies).__name__,
                indicies,
            )
            raise ValueError(
                f"Invalid `indicies`: expected list[int], got type={type(indicies).__name__}, value={indicies}."
            )
        for i in indicies:
            if not isinstance(i, int) or isinstance(i, bool):
                logger.error(
                    "Invalid `indicies`: expected list[int], got element type=%s in indicies=%s.",
                    type(i).__name__,
                    indicies,
                )
                raise ValueError(
                    f"Invalid `indicies`: expected list[int], got element type={type(i).__name__} in {indicies}."
                )

        if not indicies:
            return indicies

        min_idx = min(indicies)
        max_idx = max(indicies)
        if min_idx < 0 or max_idx >= self.max_bsz:
            logger.error(
                "Out-of-bounds `indicies`: valid range [0, %d), but got min=%d, max=%d, indicies=%s.",
                self.max_bsz,
                min_idx,
                max_idx,
                indicies,
            )
            raise IndexError(
                f"Out-of-bounds `indicies`: valid range [0, {self.max_bsz}), "
                f"but got min={min_idx}, max={max_idx}, indicies={indicies}."
            )
        return indicies

    def get(self, indicies: list[int]) -> torch.Tensor:
        idx = self._validate_indicies(indicies)
        idx_tensor = torch.tensor(idx, dtype=torch.long, device=self.device)
        # Advanced indexing creates a new physical tensor (not a view).
        return self.data[idx_tensor]

    def put(self, indicies: list[int], tensor: torch.Tensor) -> None:
        idx = self._validate_indicies(indicies)
        expected_shape = (len(idx), *self.shape)
        if tuple(tensor.shape) != expected_shape:
            logger.error(
                "Shape mismatch in `put`: expected tensor shape %s, got %s.",
                expected_shape,
                tuple(tensor.shape),
            )
            raise ValueError(
                f"Shape mismatch in `put`: expected tensor shape {expected_shape}, got {tuple(tensor.shape)}."
            )
        if tensor.dtype != self.dtype:
            logger.error(
                "Dtype mismatch in `put`: pool dtype=%s, input dtype=%s.",
                self.dtype,
                tensor.dtype,
            )
            raise TypeError(
                f"Dtype mismatch in `put`: pool dtype={self.dtype}, input dtype={tensor.dtype}."
            )
        if tensor.device != self.device:
            logger.error(
                "Device mismatch in `put`: pool device=%s, input device=%s.",
                self.device,
                tensor.device,
            )
            raise ValueError(
                f"Device mismatch in `put`: pool device={self.device}, input device={tensor.device}."
            )

        idx_tensor = torch.tensor(idx, dtype=torch.long, device=self.device)
        self.data[idx_tensor] = tensor


class TensorPoolManager:
    def __init__(self, max_bsz: int) -> None:
        if max_bsz <= 0:
            logger.error("Invalid manager `max_bsz`: expected > 0, got %d.", max_bsz)
            raise ValueError(f"Invalid manager `max_bsz`: expected > 0, got {max_bsz}.")

        self.max_bsz = int(max_bsz)
        self.pools: dict[str, TensorPool] = {}
        # True means this row has been allocated; False means free.
        self._allocated = torch.zeros(self.max_bsz, dtype=torch.bool)

    def _validate_indicies(self, indicies: list[int]) -> list[int]:
        if not isinstance(indicies, list):
            logger.error(
                "Invalid `indicies`: expected list[int], got type=%s, value=%s.",
                type(indicies).__name__,
                indicies,
            )
            raise ValueError(
                f"Invalid `indicies`: expected list[int], got type={type(indicies).__name__}, value={indicies}."
            )
        for i in indicies:
            if not isinstance(i, int) or isinstance(i, bool):
                logger.error(
                    "Invalid `indicies`: expected list[int], got element type=%s in indicies=%s.",
                    type(i).__name__,
                    indicies,
                )
                raise ValueError(
                    f"Invalid `indicies`: expected list[int], got element type={type(i).__name__} in {indicies}."
                )

        if not indicies:
            return indicies

        min_idx = min(indicies)
        max_idx = max(indicies)
        if min_idx < 0 or max_idx >= self.max_bsz:
            logger.error(
                "Out-of-bounds `indicies`: valid range [0, %d), but got min=%d, max=%d, indicies=%s.",
                self.max_bsz,
                min_idx,
                max_idx,
                indicies,
            )
            raise IndexError(
                f"Out-of-bounds `indicies`: valid range [0, {self.max_bsz}), "
                f"but got min={min_idx}, max={max_idx}, indicies={indicies}."
            )
        return indicies

    def _get_pool_or_raise(self, name: str) -> TensorPool:
        if name not in self.pools:
            logger.error("TensorPool `%s` does not exist.", name)
            raise KeyError(f"TensorPool `{name}` does not exist.")
        return self.pools[name]

    def _ensure_rows_allocated(self, idx: list[int]) -> None:
        if not idx:
            return
        idx_tensor = torch.tensor(idx, dtype=torch.long)
        not_allocated = idx_tensor[~self._allocated[idx_tensor]]
        if not_allocated.numel() > 0:
            rows = not_allocated.tolist()
            logger.error(
                "Access to unallocated rows is not allowed. rows=%s",
                rows,
            )
            raise RuntimeError(f"Access to unallocated rows is not allowed. rows={rows}")

    def add(
        self,
        name: str,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if name in self.pools:
            logger.error("TensorPool `%s` already exists.", name)
            raise ValueError(f"TensorPool `{name}` already exists.")

        self.pools[name] = TensorPool(
            max_bsz=self.max_bsz,
            shape=shape,
            dtype=dtype,
            device=device,
        )

    def alloc(self, num_rows: int) -> list[int]:
        if num_rows < 0:
            logger.error("Invalid `num_rows`: expected >= 0, got %d.", num_rows)
            raise ValueError(f"Invalid `num_rows`: expected >= 0, got {num_rows}.")
        if num_rows == 0:
            return []

        free_idx = torch.nonzero(~self._allocated, as_tuple=False).flatten()
        if free_idx.numel() < num_rows:
            logger.error(
                "Insufficient free rows for alloc(%d): free=%d, max_bsz=%d.",
                num_rows,
                int(free_idx.numel()),
                self.max_bsz,
            )
            raise RuntimeError(
                f"Insufficient free rows for alloc({num_rows}): free={int(free_idx.numel())}, max_bsz={self.max_bsz}."
            )

        selected = free_idx[:num_rows]
        self._allocated[selected] = True
        return selected.tolist()

    def reserve(self, indicies: list[int]) -> None:
        """
        Reserve specific rows explicitly.

        This is useful when an external coordinator (e.g., scheduler) assigns
        deterministic row indices and the local manager must mirror that state.
        """
        idx = self._validate_indicies(indicies)
        if not idx:
            return
        idx_tensor = torch.tensor(idx, dtype=torch.long)
        already_allocated = idx_tensor[self._allocated[idx_tensor]]
        if already_allocated.numel() > 0:
            rows = already_allocated.tolist()
            logger.error("Cannot reserve rows that are already allocated. rows=%s", rows)
            raise RuntimeError(f"Cannot reserve rows that are already allocated. rows={rows}")
        self._allocated[idx_tensor] = True

    def release(self, indicies: list[int]) -> None:
        idx = self._validate_indicies(indicies)
        if not idx:
            return

        idx_tensor = torch.tensor(idx, dtype=torch.long)
        already_free = idx_tensor[~self._allocated[idx_tensor]]
        if already_free.numel() > 0:
            rows = already_free.tolist()
            logger.error("Cannot release rows that are already free. rows=%s", rows)
            raise RuntimeError(f"Cannot release rows that are already free. rows={rows}")

        self._allocated[idx_tensor] = False

    def get(self, name: str, indicies: list[int]) -> torch.Tensor:
        pool = self._get_pool_or_raise(name)
        idx = self._validate_indicies(indicies)
        self._ensure_rows_allocated(idx)
        return pool.get(idx)

    def get_prompt_batch(
        self,
        prompt_name: str,
        mask_name: str,
        indicies: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        prompt_pool = self._get_pool_or_raise(prompt_name)
        mask_pool = self._get_pool_or_raise(mask_name)
        idx = self._validate_indicies(indicies)
        self._ensure_rows_allocated(idx)

        if not idx:
            prompt_shape = (0, 0, *prompt_pool.shape[1:])
            mask_shape = (0, 0, *mask_pool.shape[1:])
            return (
                torch.empty(prompt_shape, dtype=prompt_pool.dtype, device=prompt_pool.device),
                torch.empty(mask_shape, dtype=mask_pool.dtype, device=mask_pool.device),
                0,
            )

        max_seq_len = 0
        for row_idx in idx:
            row_mask = mask_pool.data[row_idx]
            seq_len = int(row_mask.sum().item())
            if seq_len > max_seq_len:
                max_seq_len = seq_len

        prompt_shape = (len(idx), max_seq_len, *prompt_pool.shape[1:])
        mask_shape = (len(idx), max_seq_len, *mask_pool.shape[1:])
        prompt_batch = torch.empty(prompt_shape, dtype=prompt_pool.dtype, device=prompt_pool.device)
        mask_batch = torch.empty(mask_shape, dtype=mask_pool.dtype, device=mask_pool.device)

        for out_idx, row_idx in enumerate(idx):
            prompt_batch[out_idx] = prompt_pool.data[row_idx, :max_seq_len]
            mask_batch[out_idx] = mask_pool.data[row_idx, :max_seq_len]

        return prompt_batch, mask_batch, max_seq_len

    def put(self, name: str, indicies: list[int], tensor: torch.Tensor) -> None:
        pool = self._get_pool_or_raise(name)
        idx = self._validate_indicies(indicies)
        self._ensure_rows_allocated(idx)
        pool.put(idx, tensor)
