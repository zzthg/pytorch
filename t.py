import torch
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import counters
from torch import _inductor as inductor

def compiler_fn(gm):
    """Same as torch.compile() but counts number of compiles"""

    def inner_compiler(gm_, example_inputs_):
        counters["compiled_autograd"]["compiles"] += 1
        return inductor.compile(gm_, example_inputs_)

    return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)

def check_output_and_recompiles(fn, count=1):
    with torch.autograd.set_multithreading_enabled(False):
        torch._dynamo.reset()
        counters["compiled_autograd"].clear()
        torch.manual_seed(123)
        # expected = list(fn())
        torch.manual_seed(123)
        with compiled_autograd.enable(compiler_fn):
            actual = list(fn())
        # self.assertEqual(expected, actual)
        # self.assertEqual(counters["compiled_autograd"]["captures"], count)
        # self.assertEqual(counters["compiled_autograd"]["compiles"], count)

def test_retain_grad_test1():
    input = torch.rand(1, 3, requires_grad=True)
    h1 = input * 3
    out = (h1 * h1).sum()
    print(f"input={input}")
    print(f"out={out}")

    # It should be possible to call retain_grad() multiple times
    print("t.py calling retain_grad")
    h1.retain_grad()
    h1.retain_grad()

    print("t.py calling backward")
    import pdb
    pdb.set_trace()
    # Gradient should be accumulated
    out.backward(retain_graph=True)
    import pdb
    pdb.set_trace()

    print(h1 * 2 == h1.grad) # h1 grad is None
    yield h1.grad
    out.backward(retain_graph=True)
    print(h1 * 4, h1.grad)
    yield h1.grad

check_output_and_recompiles(test_retain_grad_test1, 1)
