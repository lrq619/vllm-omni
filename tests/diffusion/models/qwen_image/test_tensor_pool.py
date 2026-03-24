# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for qwen_image TensorPool."""

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.tensor_pool import TensorPool, TensorPoolManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_pool() -> TensorPool:
    return TensorPool(max_bsz=4, shape=(2, 3), dtype=torch.float32, device="cpu")


def test_init_creates_expected_shape_dtype_device():
    pool = _make_pool()
    assert tuple(pool.data.shape) == (4, 2, 3)
    assert pool.data.dtype == torch.float32
    assert pool.data.device.type == "cpu"


@pytest.mark.parametrize(
    "kwargs,exc_type,match",
    [
        ({"max_bsz": 0, "shape": (2, 3)}, ValueError, "max_bsz"),
        ({"max_bsz": -1, "shape": (2, 3)}, ValueError, "max_bsz"),
        ({"max_bsz": 2, "shape": ()}, ValueError, "shape"),
        ({"max_bsz": 2, "shape": (2, 0)}, ValueError, "shape"),
        ({"max_bsz": 2, "shape": (2, -1)}, ValueError, "shape"),
    ],
)
def test_init_invalid_args_raise(kwargs, exc_type, match):
    with pytest.raises(exc_type, match=match):
        TensorPool(dtype=torch.float32, device="cpu", **kwargs)


def test_get_returns_new_tensor_and_expected_rows():
    pool = _make_pool()
    src = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    pool.put([0, 1, 2, 3], src.clone())

    out = pool.get([0, 1, 3])
    assert tuple(out.shape) == (3, 2, 3)
    assert torch.equal(out, src[[0, 1, 3]])

    # get() uses advanced indexing and returns a new physical tensor.
    out[0, 0, 0] = -123.0
    assert pool.data[0, 0, 0].item() != -123.0


def test_get_empty_indices_returns_empty_batch():
    pool = _make_pool()
    out = pool.get([])
    assert tuple(out.shape) == (0, 2, 3)
    assert out.device.type == "cpu"


@pytest.mark.parametrize("indicies", [[-1], [4], [0, 1, 5]])
def test_get_out_of_bounds_indices_raise(indicies):
    pool = _make_pool()
    with pytest.raises(IndexError, match="Out-of-bounds `indicies`"):
        pool.get(indicies)


@pytest.mark.parametrize("indicies", [0, [[0, 1]], torch.tensor([[0, 1]], dtype=torch.long)])
def test_get_non_1d_indices_raise(indicies):
    pool = _make_pool()
    with pytest.raises(ValueError, match="expected list\\[int\\]"):
        pool.get(indicies)


def test_put_writes_expected_rows():
    pool = _make_pool()
    values = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=torch.float32,
    )
    pool.put([1, 3], values)

    assert torch.equal(pool.data[1], values[0])
    assert torch.equal(pool.data[3], values[1])


def test_put_shape_mismatch_raises():
    pool = _make_pool()
    bad = torch.zeros((1, 2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        pool.put([0, 1], bad)


def test_put_dtype_mismatch_raises():
    pool = _make_pool()
    bad = torch.zeros((2, 2, 3), dtype=torch.float16)
    with pytest.raises(TypeError, match="Dtype mismatch"):
        pool.put([0, 1], bad)


def test_put_device_mismatch_raises():
    pool = _make_pool()
    bad = torch.zeros((2, 2, 3), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="Device mismatch"):
        pool.put([0, 1], bad)


def test_put_out_of_bounds_indices_raises():
    pool = _make_pool()
    values = torch.zeros((2, 2, 3), dtype=torch.float32)
    with pytest.raises(IndexError, match="Out-of-bounds `indicies`"):
        pool.put([0, 4], values)


def _make_manager() -> TensorPoolManager:
    mgr = TensorPoolManager(max_bsz=4)
    mgr.add("a", shape=(2, 3), dtype=torch.float32, device="cpu")
    mgr.add("b", shape=(1,), dtype=torch.float32, device="cpu")
    return mgr


def test_manager_add_duplicate_name_raises():
    mgr = TensorPoolManager(max_bsz=4)
    mgr.add("x", shape=(2, 3), dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="already exists"):
        mgr.add("x", shape=(2, 3), dtype=torch.float32, device="cpu")


def test_manager_alloc_and_release_happy_path():
    mgr = _make_manager()
    idx = mgr.alloc(3)
    assert idx == [0, 1, 2]
    assert mgr._allocated.tolist() == [True, True, True, False]

    mgr.release([1, 2])
    assert mgr._allocated.tolist() == [True, False, False, False]


def test_manager_alloc_insufficient_rows_raises_without_partial_allocation():
    mgr = _make_manager()
    first = mgr.alloc(3)
    assert first == [0, 1, 2]
    before = mgr._allocated.clone()

    with pytest.raises(RuntimeError, match="Insufficient free rows"):
        mgr.alloc(2)
    assert torch.equal(mgr._allocated, before)


def test_manager_release_free_row_raises():
    mgr = _make_manager()
    mgr.alloc(1)
    with pytest.raises(RuntimeError, match="already free"):
        mgr.release([0, 2])


def test_manager_put_get_require_allocated_rows():
    mgr = _make_manager()
    vals = torch.zeros((1, 2, 3), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="unallocated rows"):
        mgr.put("a", [0], vals)
    with pytest.raises(RuntimeError, match="unallocated rows"):
        mgr.get("a", [0])

    idx = mgr.alloc(1)
    mgr.put("a", idx, vals + 3.0)
    out = mgr.get("a", idx)
    assert torch.equal(out, vals + 3.0)


def test_manager_put_get_unknown_pool_raises():
    mgr = _make_manager()
    idx = mgr.alloc(1)
    vals = torch.zeros((1, 2, 3), dtype=torch.float32)
    with pytest.raises(KeyError, match="does not exist"):
        mgr.put("missing", idx, vals)
    with pytest.raises(KeyError, match="does not exist"):
        mgr.get("missing", idx)


def test_manager_put_checks_tensor_shape_via_pool():
    mgr = _make_manager()
    idx = mgr.alloc(2)
    bad = torch.zeros((1, 2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        mgr.put("a", idx, bad)
