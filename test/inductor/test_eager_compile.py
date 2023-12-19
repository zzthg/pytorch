import cProfile, io, pstats
import time
from pstats import SortKey
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from torch import nn
from torch._inductor import config
from torch._dynamo import config as dym_config

ShapeType = Union[torch.Size, List[int], Tuple[int, ...]]

dynamic = True
config.fx_graph_cache = True
dym_config.cache_size_limit = 2000

res_call_count = 0
ref_call_count = 0
device = "CUDA"
dispatch_key = "CUDA"
namespace_name = "aten"

def make_elementwise(op_name):
    class WrapperFn:
        def __init__(self, op_name) -> None:
            self.op_name = op_name

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            global res_call_count
            res_call_count = res_call_count + 1
            with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
                opt_fn = torch.compile(getattr(torch, self.op_name), dynamic=dynamic)
                res = opt_fn(*args, **kwargs)
                return res

    return WrapperFn(op_name)

def wrapper_fn_layer_norm(
    input: torch.Tensor,
    normalized_shape: ShapeType,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_native_layer_norm = torch.compile(
            torch.ops.aten.native_layer_norm, dynamic=dynamic
        )
        res = opt_fn_native_layer_norm(input, normalized_shape, weight, bias, eps)
        return res


# N, C, H, W = 2, 3, 4, 5
# input = torch.randn(N, C, H, W)
# layer_norm = nn.LayerNorm([C, H, W])
# weight = torch.randn([C, H, W])
# layer_norm.weight.data = weight

# beg = time.time()
# ref = layer_norm(torch.exp(x + y - y * y + x / y) + bias)
# end = time.time()
# print(end - beg)

custom_op_lib_xpu_impl = torch.library.Library("aten", "IMPL")
custom_op_lib_xpu_impl.impl_compile("native_layer_norm", wrapper_fn_layer_norm, dispatch_key)

unary_op_set = [
  "abs",
  "acos",
  "acosh",
  "asin",
  "asinh",
  "atan",
  "atanh",
  "cos",
  "cosh",
#   "bessel_j0",
#   "bessel_j1",
#   "bessel_i0",
#   "bessel_i0e",
#   "bessel_i1",
#   "bessel_i1e",
#   "bitwise_not",
#   "cbrt",
  "ceil",
  "conj_physical",
#   "digamma",
  "erf",
#   "erf_inv",
  "erfc",
#   "erfcx",
  "exp",
  "expm1",
  "exp2",
#   "fill",
  "floor",
#   "imag",
  "isfinite",
  "lgamma",
  "log",
  "log1p",
  "log2",
  "log10",
  "real",
  "reciprocal",
#   "ndtri",
  "neg",
  "round",
  "rsqrt",
  "sign",
  "signbit",
  "sin",
  "sinh",
#   "spherical_bessel_j0",
  "sqrt",
  "tan",
  "tanh",
  "trunc"
]
for unary_op_name in unary_op_set:
    custom_op_lib_xpu_impl.impl_compile(unary_op_name, make_elementwise(unary_op_name), dispatch_key)

binary_op_set = [
  "add",
#   "atan2",
#   "bitwise_and",
#   "bitwise_or",
#   "bitwise_xor",
  "div",
  "eq",
#   "fmax",
#   "fmin",
#   "fmod",
#   "gcd",
  "ge",
  "gt",
#   "hypot",
#   "igamma",
#   "igammac",
  "le",
  "lt",
#   "maximum",
#   "minimum",
  "mul",
  "ne",
#   "nextafter",
  "pow",
  "remainder",
#   "shift_left",
#   "shift_right_arithmetic",
  "sub",
#   "zeta"
]
for binary_op_name in binary_op_set:
    qualified_op_name = f"{namespace_name}::{binary_op_name}"
    op, overload_names = torch._C._jit_get_operation(qualified_op_name)
    for overload_name in overload_names:

        _overload_name = overload_name if overload_name else 'default'
        try:
            schema = torch._C._get_schema(qualified_op_name, _overload_name)
            reg_name = f"{schema.name}.{schema.overload_name}"
            custom_op_lib_xpu_impl.impl_compile(reg_name, make_elementwise(binary_op_name), dispatch_key)
        except:
            continue


def demo_eager_run():
    device = torch.device("cuda")
    x = torch.empty(2, 3, 4, 5, device=device).fill_(1)
    y = torch.empty(2, 3, 4, 5, device=device).fill_(2)

    for unary_op_name in unary_op_set:
        res = getattr(torch, unary_op_name)(x)
        ref_call_count = ref_call_count + 1
        assert ref_call_count == res_call_count

    for binary_op_name in binary_op_set:
        res = getattr(torch, binary_op_name)(x, y)
        ref_call_count = ref_call_count + 1
        assert ref_call_count == res_call_count, f"binary_op_name:{binary_op_name} ref_call_count: {ref_call_count}, res_call_count: {res_call_count}"

def demon_perf_profiling():
    device = torch.device("cuda")
    x = torch.empty(2, 3, 4, 5, device=device).fill_(1)
    y = torch.empty(2, 3, 4, 5, device=device).fill_(2)

    print("High Overhead - Intialization + Compilation")
    print("+" * 100)
    sortby = SortKey.CUMULATIVE
    with cProfile.Profile() as pr:
        # First time run - The overhead is the high
        x + y

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    print("Medium Overhead - Compilation")
    print("+" * 100)
    with cProfile.Profile() as pr:
        # Second time run - The overhead is the medium
        x - y

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    print("Low Overhead")
    print("+" * 100)
    with cProfile.Profile() as pr:
        # Third time run - The overhead is the low
        x - y

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    x = torch.empty(2, 3, 5, 5, device=device).fill_(1)
    y = torch.empty(2, 3, 5, 5, device=device).fill_(2)

    print("High Overhead - Re-Compilation")
    print("+" * 100)
    with cProfile.Profile() as pr:
        # Third time run - The overhead is the low
        x - y

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

demon_perf_profiling()
