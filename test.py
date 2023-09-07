import torch


A = torch.ones(128, 128).cuda().to(torch.int8)
B = (10 * torch.ones(128, 128)).cuda().to(torch.int8).t()

print(A)
print(B)

A_compressed = torch._cslt_compress(A)
print(torch._cslt_sparse_mm(A_compressed, B))
print(torch._int_mm(A, B))
