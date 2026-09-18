# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""CPU contract tests; load the helper without importing TE's native extension."""

import importlib.util
import sys
import weakref
from dataclasses import dataclass, replace
from pathlib import Path

import torch

path = (
    Path(__file__).resolve().parents[2]
    / "transformer_engine/pytorch/ops/fused/grouped_mlp_cache.py"
)
spec = importlib.util.spec_from_file_location("grouped_mlp_cache_under_test", path)
cache_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cache_module
spec.loader.exec_module(cache_module)


@dataclass
class Weight:
    rowwise_data: torch.Tensor
    scale_inv: torch.Tensor
    num_tensors: int = 2
    logical_shape: tuple = (4, 4)
    quantizer: object = None
    _with_gemm_swizzled_scales: bool = False

    def copy(self):
        return replace(self)


def make_weight():
    return Weight(torch.arange(16, dtype=torch.uint8), torch.arange(16, dtype=torch.uint8))


def test_interval_and_invalidation():
    owner = object()
    weight = make_weight()
    calls = []

    def swizzle(prepared):
        calls.append(weakref.ref(prepared))
        prepared.scale_inv = prepared.scale_inv.flip(-1)

    def get(enabled=True):
        return cache_module.packed_grouped_weight_scales(
            owner, weight, enabled=enabled, scale_dtype=torch.uint8, swizzle=swizzle
        )

    assert get().data_ptr() != get().data_ptr()
    with cache_module.grouped_mlp_weight_cache() as cache:
        first = get()
        assert get() is first
        assert get(False) is not first
        weight.scale_inv.add_(1)
        cache.invalidate(owner)
        second = get()
        assert second is not first
        torch.testing.assert_close(second, weight.scale_inv.flip(-1))
        assert second is get()
        with cache_module.grouped_mlp_weight_cache() as nested:
            assert get() is not second
            assert len(nested.entries) == 1
        assert not nested.entries
        assert get() is second
        cache.invalidate()
        assert not cache.entries
        assert get() is not second
    assert not cache.entries
    assert all(ref() is None for ref in calls)
    assert cache_module.weight_cache_context.get() is None


def test_replacement_and_layout():
    owners = [object(), object()]
    weight = make_weight()

    def swizzle(prepared):
        prepared.scale_inv = prepared.scale_inv.clone()

    def get(owner=owners[0], layout_key=(), dtype=torch.uint8):
        return cache_module.packed_grouped_weight_scales(
            owner, weight, enabled=True, scale_dtype=dtype, layout_key=layout_key, swizzle=swizzle
        )

    with cache_module.grouped_mlp_weight_cache() as cache:
        previous = get()
        assert get(owners[1]) is not previous
        cache.invalidate(owners[1])
        assert get() is previous
        weight.scale_inv = weight.scale_inv.clone()
        current = get()
        assert current is not previous
        previous = current
        weight.rowwise_data = weight.rowwise_data.clone()
        current = get()
        assert current is not previous
        previous = current
        weight.scale_inv = weight.scale_inv.view(4, 4).t()
        current = get()
        assert current is not previous
        assert tuple(current.stride()) == tuple(weight.scale_inv.stride())
        previous = current
        weight.quantizer = object()
        current = get()
        assert current is not previous
        assert get(layout_key=(32,)) is not current
        assert get(dtype=torch.int8).dtype == torch.int8
        cache.invalidate()
        assert not cache.entries


def test_exception_closes_interval():
    try:
        with cache_module.grouped_mlp_weight_cache() as cache:
            cache.entries[1] = object()
            raise ValueError("test")
    except ValueError:
        pass
    assert not cache.entries
    assert cache_module.weight_cache_context.get() is None
