# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Explicit unchanged-weight intervals for grouped MLP packed MXFP8 scales."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch


@dataclass
class GroupedMLPWeightCache:
    """Packed scales owned by one interval; invalidate after every weight update.

    Raw extension writes and writes through quantized aliases are not automatically
    detected. Call ``invalidate(owner)`` after optimizer updates, requantization,
    or other in-place scale changes, before the next forward. ``owner`` is the
    corresponding GroupedLinear operation; omit it to invalidate all weights.
    """

    entries: dict = field(default_factory=dict)

    def invalidate(self, owner=None):
        """Forget one GroupedLinear's entries, or every entry when owner is None."""
        if owner is None:
            self.entries.clear()
        else:
            for key in tuple(self.entries):
                if key[0] == id(owner):
                    del self.entries[key]


weight_cache_context = ContextVar("grouped_mlp_weight_cache", default=None)


@contextmanager
def grouped_mlp_weight_cache():
    """Reuse packed weight scales until the context exits.

    Only dense, primary MXFP8 weights opt in at the call site. Source buffers
    must remain unchanged inside the interval unless explicitly invalidated.
    Nested intervals are independent. CUDA streams have separate entries;
    normal producer-to-consumer synchronization remains the caller's duty.

    Example::

        with grouped_mlp_weight_cache() as cache:
            for microbatch in microbatches:
                loss = model(microbatch)
                loss.backward()
            optimizer.step()
            cache.invalidate()

    The default, outside this context, is to prepare scales on every forward.
    """
    cache = GroupedMLPWeightCache()
    token = weight_cache_context.set(cache)
    try:
        yield cache
    finally:
        weight_cache_context.reset(token)
        cache.invalidate()


def weight_buffer_signature(tensor):
    """Describe the current source allocation and its interpretation."""
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


def swizzle_grouped_weight(weight):
    """Prepare only rowwise scales on a shallow storage copy."""
    import transformer_engine_torch as tex

    tex.grouped_swizzle_for_gemm(weight, rowwise=True, columnwise=False)


def packed_grouped_weight_scales(
    owner,
    weight,
    *,
    enabled,
    scale_dtype,
    layout_key=(),
    swizzle=swizzle_grouped_weight,
):
    """Return packed rowwise scales viewed as ``scale_dtype``.

    ``enabled`` must only be true for dense primary MXFP8 grouped weights using
    the grouped swizzle (not the single-group special path). ``layout_key``
    contains any caller-specific quantization/layout configuration. Callers
    always obtain B data from the current weight, independently of this cache.
    ``swizzle`` mutates the supplied shallow copy's ``scale_inv`` only.
    """
    cache = weight_cache_context.get() if enabled else None
    if cache is not None:
        data = weight.rowwise_data
        scales = weight.scale_inv
        stream = torch.cuda.current_stream(scales.device) if scales.is_cuda else None
        key = (id(owner), scales.device, None if stream is None else stream.cuda_stream)
        signature = (
            weight_buffer_signature(data),
            weight_buffer_signature(scales),
            weight.num_tensors,
            tuple(weight.logical_shape),
            id(weight.quantizer),
            weight._with_gemm_swizzled_scales,
            scale_dtype,
            layout_key,
        )
        entry = cache.entries.get(key)
        if entry is not None and entry[0] == signature:
            return entry[-1]
    prepared = weight.copy()
    swizzle(prepared)
    packed = prepared.scale_inv.view(dtype=scale_dtype)
    if cache is not None:
        cache.entries[key] = (signature, owner, data, scales, weight.quantizer, stream, packed)
    return packed
