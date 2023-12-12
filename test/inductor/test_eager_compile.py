from typing import List, Optional, Tuple, Union
import torch
import time
import numpy as np

from torch import nn

ShapeType = Union[torch.Size, List[int], Tuple[int, ...]]

device = torch.device("cpu")
x = torch.empty(2, 3, 4, 5).fill_(1)
y = torch.empty(2, 3, 4, 5).fill_(2)
bias = torch.randn(2, 3, 4, 5)

dynamic = True

def wrapper_fn_sub(*args, **kwargs):
    a, b = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_sub = torch.compile(torch.ops.aten.sub, dynamic=dynamic)
        res = opt_fn_sub(a, b)
        return res

def wrapper_fn_add(*args, **kwargs):
    a, b = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_add = torch.compile(torch.ops.aten.add, dynamic=dynamic)
        res = opt_fn_add(a, b)
        return res

def wrapper_fn_mul(*args, **kwargs):
    a, b = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_mul = torch.compile(torch.ops.aten.mul, dynamic=dynamic)
        res = opt_fn_mul(a, b)
        return res

def wrapper_fn_div(*args, **kwargs):
    a, b = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_div = torch.compile(torch.ops.aten.div, dynamic=dynamic)
        res = opt_fn_div(a, b)
        return res

def wrapper_fn_exp(*args, **kwargs):
    a, = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_exp = torch.compile(torch.ops.aten.exp, dynamic=dynamic)
        res = opt_fn_exp(a)
        return res

def wrapper_fn_layer_norm(
    input: torch.Tensor,
    normalized_shape: ShapeType,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_native_layer_norm = torch.compile(torch.ops.aten.native_layer_norm, dynamic=dynamic)
        res = opt_fn_native_layer_norm(input, normalized_shape, weight, bias, eps)
        print("l" * 100)
        return res

N, C, H, W = 2, 3, 4, 5
input = torch.randn(N, C, H, W)
layer_norm = nn.LayerNorm([C, H, W])
weight = torch.randn([C, H, W])
layer_norm.weight.data = weight

beg = time.time()
ref = layer_norm(torch.exp(x + y - y * y + x / y) + bias)
end = time.time()
print(end - beg)

custom_op_lib_xpu_impl = torch.library.Library("aten", "IMPL")
custom_op_lib_xpu_impl.impl_compile("sub.Tensor", wrapper_fn_sub, "CPU")
custom_op_lib_xpu_impl.impl_compile("add.Tensor", wrapper_fn_add, "CPU")
custom_op_lib_xpu_impl.impl_compile("mul.Tensor", wrapper_fn_mul, "CPU")
custom_op_lib_xpu_impl.impl_compile("div.Tensor", wrapper_fn_div, "CPU")
custom_op_lib_xpu_impl.impl_compile("exp", wrapper_fn_exp, "CPU")
custom_op_lib_xpu_impl.impl_compile("native_layer_norm", wrapper_fn_layer_norm, "CPU")

beg = time.time()
res = layer_norm(torch.exp(x + y - y * y + x / y) + bias)
end = time.time()
print(end - beg)

print(res)
print(ref)
print(res == ref)
assert np.allclose(res.detach().numpy(), ref.detach().numpy(), rtol=1e-3, atol=1e-3)
