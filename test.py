import torch
from functorch import make_fx
from custom_ops import my_sin, my_sin_cos, MySinInplace

# =====================================================================
# Test cases 
# =====================================================================

# =====================================================================
# my_sin Basic
x = torch.randn(3)
y = my_sin(x)
assert torch.allclose(y, x.sin())

# =====================================================================
# my_sin Autograd
x = torch.randn(3, requires_grad=True)
y = my_sin(x)
y.sum().backward()
assert torch.allclose(x.grad, x.cos())

# =====================================================================
# my_sin make_fx
def f(x):
    return my_sin(x)


gm = make_fx(f, tracing_mode="fake")(x)
result = gm.code.strip()
expected = """
def forward(self, x_1):
    my_sin = torch.ops.mangled2__custom_ops.MySin.default(x_1);  x_1 = None
    return my_sin
""".strip()
assert result == expected

# =====================================================================
# my_sin_cos Basic
x = torch.randn(3)
y = my_sin_cos(x)
assert torch.allclose(y, x.sin().cos())

# =====================================================================
# my_sin_cos make_fx
def f(x):
    return my_sin_cos(x)


gm = make_fx(f, tracing_mode="fake")(x)
result = gm.code.strip()
expected = """
def forward(self, x_1):
    my_sin_cos = torch.ops.mangled2__custom_ops.MySinCos.default(x_1);  x_1 = None
    return my_sin_cos
""".strip()
assert result == expected

x = torch.randn(3)
x_version = x._version
MySinInplace.call(x)
new_x_version = x._version
# TODO: need to fix.
# assert x_version < new_x_version, (x_version, new_x_version)


from library import OP_TO_TRACEABLE_IMPL

import torch._inductor
from torch._inductor.decomposition import register_decomposition
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

@register_decomposition([torch.ops.mangled2__custom_ops.MySinCos.default])
def decomp(*args, **kwargs):
    # return OP_TO_TRACEABLE_IMPL[0](*args, **kwargs)
    with disable_proxy_modes_tracing():
        class MyMod(torch.nn.Module):
            def forward(self, x):
                return OP_TO_TRACEABLE_IMPL[0]

        gm = torch.export.export(MyMod(), args, kwargs)
    return gm.module(*args, **kwargs)

@torch.compile(backend="inductor")
def f(x):
    return my_sin_cos(x)

x = torch.randn(3)
y = f(x)
assert torch.allclose(y, x.sin().cos())

