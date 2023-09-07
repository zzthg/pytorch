import torch
import torch.utils._pytree as pytree

from torch._ops import HigherOrderOperator
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._higher_order_ops.utils import autograd_not_implemented


def invoke(fn):
    return invoke_op(fn)

invoke_op = HigherOrderOperator("invoke")

@invoke_op.py_impl(ProxyTorchDispatchMode)
def inner(fn):
    mode = _get_current_dispatch_mode()
    out_proxy = mode.tracer.create_proxy(
        "call_function", fn, (), {}, name="invocation"
    )
    out = fn()
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

@invoke_op.py_impl(FakeTensorMode)
def inner(fn):
    return ()

@invoke_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_op_dense(fn):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return fn()

invoke_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(invoke_op, deferred_error=True)
)

@invoke_op.py_impl(DispatchKey.Functionalize)
def invoke_functionalized(fn):
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        return invoke_op(fn)


# TODO(voz): Make this automatic for keys, this is very ugly atm
invoke_op.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
invoke_op.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
invoke_op.fallthrough(DispatchKey.ADInplaceOrView)
invoke_op.fallthrough(DispatchKey.BackendSelect)
invoke_op.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
invoke_op.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
