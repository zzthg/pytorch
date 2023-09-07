import torch


A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).to(torch.int8).cuda()
B = (10 * torch.ones(128, 128)).cuda().to(torch.int8)
alpha = 0.01 * torch.rand(128).cuda()

# A_compressed = torch.load("w_vals.pt")
# B = torch.load("tmp.pt")
# alpha = torch.load("w_scales.pt")

print(A)
print(B)
print(alpha)

A_compressed = torch._cslt_compress(A)
print(torch._cslt_sparse_mm(A_compressed, B.t(), alpha).t())
print(torch._int_mm(A, B))
