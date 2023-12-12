import torch
import time

device = torch.device("cpu")
x = torch.empty(10).to(device=device).fill_(1).requires_grad_()
y = torch.empty(10).to(device=device).fill_(2).requires_grad_()

def wrapper_fn_sub(*args, **kwargs):
    a, b = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_sub = torch.compile(torch.ops.aten.sub)
        res = opt_fn_sub(a, b)
        return res

def wrapper_fn_add(*args, **kwargs):
    a, b = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_add = torch.compile(torch.ops.aten.add)
        res = opt_fn_add(a, b)
        return res

def wrapper_fn_mul(*args, **kwargs):
    a, b = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_mul = torch.compile(torch.ops.aten.mul)
        res = opt_fn_mul(a, b)
        return res

def wrapper_fn_div(*args, **kwargs):
    a, b = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_div = torch.compile(torch.ops.aten.div)
        res = opt_fn_div(a, b)
        return res

def wrapper_fn_exp(*args, **kwargs):
    a, = args
    with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False):
        opt_fn_exp = torch.compile(torch.ops.aten.exp)
        res = opt_fn_exp(a)
        return res

beg = time.time()
ref = torch.exp(x + y - y * y + x / y)
end = time.time()
print(end - beg)

custom_op_lib_xpu_impl = torch.library.Library("aten", "IMPL")
custom_op_lib_xpu_impl.impl_compile("sub.Tensor", wrapper_fn_sub, "CPU")
custom_op_lib_xpu_impl.impl_compile("add.Tensor", wrapper_fn_add, "CPU")
custom_op_lib_xpu_impl.impl_compile("mul.Tensor", wrapper_fn_mul, "CPU")
custom_op_lib_xpu_impl.impl_compile("div.Tensor", wrapper_fn_div, "CPU")
custom_op_lib_xpu_impl.impl_compile("exp", wrapper_fn_exp, "CPU")

beg = time.time()
res = torch.exp(x + y - y * y + x / y)
end = time.time()
print(end - beg)

assert all(res == ref)
