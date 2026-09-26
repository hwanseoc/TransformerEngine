# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""CPU-only dispatch contract checks; imports no Transformer Engine or CUDA code."""

import importlib.util
import math
from pathlib import Path
import sys
from unittest.mock import Mock


import pytest


class Tensor:
    def __init__(self, shape, dtype="float8_e4m3fn", contiguous=True):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = "cuda:0"
        self.is_cuda = True
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return math.prod(self.shape)

    def stride(self, dim=None):
        strides = tuple(math.prod(self.shape[index + 1 :]) for index in range(self.ndim))
        if not self.contiguous:
            strides = tuple(2 * stride for stride in strides)
        return strides if dim is None else strides[dim]

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]


def arguments(m=512):
    return dict(
        a_tensor=Tensor((m, 256)),
        b_tensor=Tensor((4, 512, 256)),
        sfa_tensor=Tensor((m * 8,), "float8_e8m0fnu"),
        sfb_tensor=Tensor((4 * 512 * 8,), "float8_e8m0fnu"),
        alpha_tensor=Tensor((4,), "bfloat16"),
        padded_offsets=Tensor((4,), "int32"),
        prob_tensor=Tensor((m,), "bfloat16"),
        bias_tensor=None,
        scheduler_counter_tensor=Tensor((4,), "int32"),
        current_stream=7,
        sf_vec_size=32,
        d_dtype="float8_e4m3fn",
        use_dynamic_sched=True,
    )


def load_module(root):
    name = "execution_" + str(len(sys.modules))
    spec = importlib.util.spec_from_file_location(name, root / "grouped_mlp_execution.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def harness(module):
    kernel = Mock(return_value="wrapper")
    kernel.supports_canonical_layouts = True
    plans = []

    def prepare(kind, **kwargs):
        plan = Mock()
        plan.run.return_value = ("plan", len(plans))
        plans.append(plan)
        return plan

    factory = Mock(side_effect=prepare)
    execution = module.GroupedMLPExecution(True, True, True, factory)
    return execution, kernel, factory, plans


def test_fallbacks(module):
    for case in ("empty", "legacy_b", "strided_prob", "short_prob"):
        execution, kernel, factory, plans = harness(module)
        kwargs = arguments()
        if case == "empty":
            kwargs = arguments(0)
        elif case == "legacy_b":
            kwargs["b_tensor"] = Tensor((512, 256, 1))
        elif case == "strided_prob":
            kwargs["prob_tensor"] = Tensor((512,), "bfloat16", contiguous=False)
        else:
            kwargs["prob_tensor"] = Tensor((511,), "bfloat16")
        assert execution.run("glu", kernel, kwargs) == "wrapper", case
        kernel.assert_called_once_with(**kwargs)
        factory.assert_not_called()
        assert not execution.plans and not plans, case


def test_plan_reuse(module):
    execution, kernel, factory, plans = harness(module)
    first = arguments()
    second = arguments()
    assert execution.run("glu", kernel, first) == ("plan", 0)
    assert execution.run("glu", kernel, second) == ("plan", 0)
    assert factory.call_count == 1
    assert plans[0].run.call_count == 2
    plans[0].run.assert_called_with(check=False, **second)
    assert execution.run("glu", kernel, arguments(768)) == ("plan", 0), "Plans accept any M"
    assert execution.run("glu", kernel, dict(first, use_dynamic_sched=False)) == ("plan", 1)
    assert factory.call_count == 2 and len(execution.plans) == 2
    kernel.assert_not_called()


def test_cache_bound(module):
    execution, kernel, factory, plans = harness(module)
    for index in range(12):
        kwargs = dict(arguments(), b_tensor=Tensor((4, 512 * (index + 1), 256)))
        execution.run("glu", kernel, kwargs)
        assert len(execution.plans) <= 4, (index, len(execution.plans))
    assert factory.call_count == 12
    execution.run("glu", kernel, kwargs)
    assert factory.call_count == 12, "Most recent configuration should remain cached"
    kernel.assert_not_called()


@pytest.fixture
def module():
    root = Path(__file__).resolve().parents[2] / "transformer_engine/pytorch/ops/fused"
    return load_module(root)
