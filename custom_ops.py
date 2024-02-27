import torch
import numpy as np
# aka torch.library
from library import Operator, traceable

# =====================================================================
# This is example user library code. It defines
# my_sin and my_sin_cos operators.
# =====================================================================

# User provides their custom op schema and implementations
class MySin(Operator):
    schema = "(Tensor x) -> Tensor"

    # the black-box cpu kernel
    @staticmethod
    def impl_cpu(x):
        return torch.from_numpy(np.sin(x.detach().cpu().numpy()))

    # the black-box cuda kernel
    @staticmethod
    def impl_cuda(x):
        return torch.from_numpy(np.sin(x.detach().cpu().numpy())).to(x.device)

    # the abstract impl. Must be "traceable". User must use opcheck to test.
    @staticmethod
    def abstract(x):
        return torch.empty_like(x)

    # autograd: provide us setup_backward() and backward() methods.
    # these must be "traceable". User must use opcheck to test.
    @staticmethod
    def setup_backward(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors 
        return grad_output * x.cos()


# The user may optionally wrap in a function that provides a docstring and type hints.
def my_sin(x: torch.Tensor) -> torch.Tensor:
    """my_sin(x: Tensor) -> Tensor

    Returns the sin of x.
    """
    return MySin.call(x)


# Example of an operator that is implemented with pytorch operations
# We automatically generate an abstract impl for it.
class MySinCos(Operator):
    schema = "(Tensor x) -> Tensor"

    # Instead of specifying separate per-device impls, the user may give us a 
    # single `impl` staticmethod that we will apply to all backends,
    # CompositeExplicitAutograd-style.
    @staticmethod
    # Specifies that the impl is make_fx traceable. We will autogenerate rules
    # (e.g. abstract, autograd, vmap). The user may override these by declaring
    # those methods.
    # This decorator may only be applied to `impl`.
    @traceable
    def impl(x):
        return x.sin().cos()


def my_sin_cos(x):
    """my_sin_cos(x: Tensor) -> Tensor

    Returns x.sin().cos()
    """
    return MySinCos.call(x)


# Mutable op example
class MySinInplace(Operator):
    schema = "(Tensor(a!) x) -> ()"

    # the black-box cpu kernel
    @staticmethod
    def impl_cpu(x):
        x_np = x.detach().numpy()
        np.sin(x_np, out=x_np)

    # the abstract impl. Must be "traceable". User must use opcheck to test.
    @staticmethod
    def abstract(x):
        return None
