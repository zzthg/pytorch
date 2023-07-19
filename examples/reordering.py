import torch

@torch.compile
def fn(x, y):
    b = torch.sin(y)

    # b has 2 fanouts
    c = torch.cos(b)
    d = torch.tan(b)

    a = torch.matmul(x, b)
    a = torch.matmul(a, c)
    a = torch.matmul(a, d)
    return a


x = torch.randn(4, 4, device="cuda")
y = torch.randn(4, 4, device="cuda")
fn(x, y)
