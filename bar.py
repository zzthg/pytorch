import torch
from torch.custom_op import triton_op

##################################################
# TEST
##################################################
import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def mul_kernel(arr1_ptr, arr2_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Step 1: Get range of axis
    pid = tl.program_id(axis=0)

    # Step 2: Define the offsets and mask
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Step 3: Load the data from RAM
    arr1 = tl.load(arr1_ptr + offsets, mask=mask)
    arr2 = tl.load(arr2_ptr + offsets, mask=mask)

    # Step 4: Do the computation
    output = arr1 * arr2

    # Step 5: Store the result in RAM
    tl.store(output_ptr + offsets, output, mask=mask)


def mul_grid(arr1, arr2, N, BLOCK_SIZE):
    # TODO: not 3d grid
    return lambda meta: (N, meta['BLOCK_SIZE'], 1)


def mul_abstract(arr1: torch.Tensor, arr2: torch.Tensor, N, BLOCK_SIZE):
    output = torch.empty_like(arr1)
    return output


triton_op(
    "namespace::mul",
    "Tensor arr1, Tensor arr2, Output out, int N, int BLOCK_SIZE",
    mul_kernel,
    mul_grid,
    mul_abstract)

def f(x, y):
    return torch.ops.namespace.mul(x, y, x.numel(), 1024)

x = torch.randn(3, device='cuda')
y = torch.randn(3, device='cuda')

# Eager Mode
expected = x * y
result = f(x, y)
assert torch.allclose(result, expected)
print("passed eager mode test")

AOT = True

if not AOT:
    # With torch.compile
    f_opt = torch.compile(f, fullgraph=True)
    result = f_opt(x, y)
    assert torch.allclose(result, expected)
    print("passed torch.compile test")
else:
    # Using AOTInductor
    import torch._inductor
    from functorch import make_fx
    gm = make_fx(f)(x, y)
    torch._inductor.aot_compile(gm, (x, y))
    print("passed AOTInductor test")

