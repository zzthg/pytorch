import torch

# sample for issue#123374

def foo(x):
    return x.sum()

torch.compile(foo)(torch.ones(10))
