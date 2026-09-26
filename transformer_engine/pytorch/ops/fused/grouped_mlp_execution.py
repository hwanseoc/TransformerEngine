# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Per-fused-operation execution plans for canonical dense MXFP8 calls."""

from dataclasses import dataclass, field
import os

NON_SCALAR_ARGUMENTS = frozenset(("padded_offsets", "current_stream", "b_ptrs", "sfb_ptrs"))


@dataclass
class GroupedMLPExecution:
    prepared: bool
    reuse_rows: bool
    counters: bool
    prepare: object
    plans: dict = field(default_factory=dict)

    def run(self, kind, kernel, kwargs):
        if (
            not self.prepared
            or not getattr(kernel, "supports_canonical_layouts", False)
            or kwargs.get("b_tensor") is None
            or kwargs["a_tensor"].ndim != 2
            or kwargs.get("sf_vec_size") != 32
        ):
            return kernel(**kwargs)
        a, b = kwargs["a_tensor"], kwargs["b_tensor"]
        prob = kwargs.get("prob_tensor")
        if (
            a.shape[0] == 0
            or b.ndim != 3
            or b.shape[-1] != a.shape[-1]
            or b.shape[0] != kwargs["padded_offsets"].numel()
            or not b.is_contiguous()
            or (prob is not None and (prob.shape != (a.shape[0],) or not prob.is_contiguous()))
        ):
            return kernel(**kwargs)
        # Plans accept any routed row count, so M is not part of the key.
        key = (
            kind,
            int(kwargs["current_stream"]),
            a.shape[1],
            a.dtype,
            a.device,
            b.shape,
            b.dtype,
            kwargs["alpha_tensor"].dtype,
            None if prob is None else prob.dtype,
            None if kwargs.get("bias_tensor") is None else kwargs["bias_tensor"].dtype,
            kwargs.get("scheduler_counter_tensor") is not None,
            *(
                item
                for item in kwargs.items()
                if item[0] not in NON_SCALAR_ARGUMENTS and not item[0].endswith("_tensor")
            ),
        )
        plan = self.plans.get(key)
        if plan is None:
            plan = self.prepare(kind, reuse_row_outputs=self.reuse_rows and kind == "glu", **kwargs)
            if len(self.plans) >= 4:
                del self.plans[next(iter(self.plans))]
            self.plans[key] = plan
        return plan.run(check=False, **kwargs)


def make_grouped_mlp_execution(glu_kernel, quant_kernel):
    try:
        from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm
    except ImportError:
        prepare_grouped_gemm = None
    canonical = getattr(glu_kernel, "supports_canonical_layouts", False) and getattr(
        quant_kernel, "supports_canonical_layouts", False
    )
    return GroupedMLPExecution(
        prepared=canonical
        and prepare_grouped_gemm is not None
        and os.getenv("NVTE_GROUPED_MLP_PREPARED", "1") != "0",
        reuse_rows=os.getenv("NVTE_GROUPED_MLP_REUSE_ROWS", "1") != "0",
        counters=getattr(glu_kernel, "supports_external_scheduler_counter", False)
        and getattr(quant_kernel, "supports_external_scheduler_counter", False)
        and os.getenv("NVTE_GROUPED_MLP_EXTERNAL_COUNTERS", "1") != "0",
        prepare=prepare_grouped_gemm,
    )
