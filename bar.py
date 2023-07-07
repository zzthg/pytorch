import torch
from torch._higher_order_ops.wrap import wrap

def g(*, x):
    return x.sin()
# def g(x):
#     return x.sin()

@torch.compile(backend='aot_eager', fullgraph=True)
def f(x):
    # return g(x=x)
    return wrap(g, x=x)

x = torch.randn([])
f(x)

# import torch._dynamo
# gm, _ = torch._dynamo.export(f, x)
# gm.print_readable()
