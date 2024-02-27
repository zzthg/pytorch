import torch
import inspect
import functools
import re
import sys
from typing import Dict, List

"""
These are changes that we'll add to PyTorch
"""


API = "torch.Operator"
# We support the following impl_<device_type> methods. Feel free to add more.
SUPPORTED_DEVICE_TYPES = {"cpu", "cuda"}
# Decorators (like @traceable) set this field on methods
ANNOTATION_FIELD = "__torch_properties__"
# Maps qualname to the Library initially used to def/impl the Operator
OP_TO_LIB: Dict[str, torch.library.Library] = {}
OP_TO_TRACEABLE_IMPL =  []


def register(opdef: "Operator"):
    name = opdef.__name__

    frame = sys._getframe(2)
    mod = inspect.getmodule(frame)
    if mod is None:
        raise RuntimeError("{API}: can't infer module")

    assert mod is not None
    ns = mangle_module(mod.__name__)
    qualname = f"{ns}::{name}"

    check_nameless_schema(opdef.schema)
    check_supported_schema(opdef, qualname)
    check_allowed_attrs(opdef)
    lib = get_library_allowing_overwrite(ns, name)
    lib.define(f"{name}{opdef.schema}")
    op = torch._library.utils.lookup_op(qualname)

    impl_method = register_device_impls(lib, name, opdef)
    properties = get_properties(impl_method)
    if properties.get("traceable", False):
        OP_TO_TRACEABLE_IMPL.append(impl_method)

    if getattr(opdef, "abstract", None):
        torch.library.impl_abstract(qualname, opdef.abstract, lib=lib)
    elif impl_method and properties.get('traceable', False):
        torch.library.impl_abstract(qualname, impl_method, lib=lib)

    register_autograd(lib, name, opdef, op)


    ophandle = torch._C._dispatch_find_schema_or_throw(qualname, "")
    torch._C._dispatch_set_report_error_callback(
        ophandle, functools.partial(report_error_callback, name)
    )

    return op


def get_properties(impl_method):
    properties = getattr(impl_method, ANNOTATION_FIELD, {})
    return properties


def register_device_impls(lib, name, opdef):
    check_either_single_or_split_impl(opdef)
    impl_method = getattr(opdef, "impl", None)
    if impl_method is not None:
        # TODO: Don't allow the meta to be reused for FakeTensor.
        register_backend(lib, name, impl_method, "CompositeExplicitAutograd")
        return impl_method

    for device_type in SUPPORTED_DEVICE_TYPES:
        impl_device_method = getattr(opdef, f"impl_{device_type}", None)
        if impl_device_method is not None:
            dk = torch._C._dispatch_key_for_device(device_type)
            register_backend(lib, name, impl_device_method, dk)


def check_either_single_or_split_impl(opdef):
    has_impl = getattr(opdef, "impl", None)
    device_impls: List[str] = []
    for device_type in SUPPORTED_DEVICE_TYPES:
        device_impl = getattr(opdef, "impl_{device_type}", None)
        if device_impl is not None:
            device_impls.append(device_impl)

    if has_impl and len(device_impls) > 0:
        raise ValueError(
            f"{API}: Expected there to be either a single `impl` method or "
            f"any number of `impl_<device>` methods. Found both an `impl` method "
            f"and {device_impls} methods.")


def check_nameless_schema(schema):
    """E.g. "(Tensor x) -> Tensor" instead of sin(Tensor x) -> Tensor"""
    match = re.match(r'\(.*\) -> .*$', schema)
    if match is not None:
        return
    raise ValueError(
        f"{API}: expected .schema to look like \"(<args>) -> <rets>\" "
        f"but got {schema}")


def check_allowed_attrs(op):
    return
    attrs = set(dir(op)) - set(dir(object))
    allowed_attrs = {
        "namespace",
        "schema",
        "impl_cpu",
        "impl_cuda",
        "abstract"
        "setup_backward",
        "backward",
    }
    if attrs.issubset(allowed_attrs):
        return
    raise ValueError(
        f"{API}: Subclasses are only allowed to have the following attributes: "
        f"attrs. Got unknown attribute {attrs - allowed_attrs}; please delete "
        f"them.")


def get_library_allowing_overwrite(ns, name):
    qualname = f"{ns}::{name}"

    if qualname in OP_TO_LIB:
        OP_TO_LIB[qualname]._destroy()
        del OP_TO_LIB[qualname]

    lib = torch.library.Library(ns, "FRAGMENT")
    OP_TO_LIB[qualname] = lib
    return lib


def traceable(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    setattr(wrapper, ANNOTATION_FIELD, {**getattr(f, ANNOTATION_FIELD, {})})
    getattr(wrapper, ANNOTATION_FIELD)['traceable'] = True
    return wrapper


def dispatch_keyset_before(dk):
    result = torch._C._dispatch_keyset_full()
    result = result - torch._C._dispatch_keyset_full_after(dk)
    result = result.remove(dk)
    return result


def register_backend(lib, name, kernel, key):
    if kernel is None:
        def wrapped(*args, **kwargs):
            raise RuntimeError(
                "{name}: was passed {key} Tensors, but "
                "{name}.{key.lowercase}_impl was not defined")
    else:
        def wrapped(*args, **kwargs):
            before_dense = dispatch_keyset_before(torch._C.DispatchKey.Dense)
            with torch._C._ExcludeDispatchKeyGuard(before_dense):
                return kernel(*args, **kwargs)

    lib.impl(name, wrapped, key)


def report_error_callback(op, key: str) -> None:
    if key == "Undefined":
        raise NotImplementedError(
            f"{op}: There were no Tensor inputs to this operator "
            f"(e.g. you passed an empty list of Tensors). If your operator is a "
            f"factory function (that is, it takes no Tensors and constructs "
            f"a new one), then please file an issue on GitHub."
        )
    if key == "Meta":
        raise NotImplementedError(
            f"{op}: when running with device='Meta' tensors: there is no "
            f"abstract impl registered for this op. Please register one by "
            f"defining the {op}.abstract staticmethod."
        )
    if key in ("CPU", "CUDA"):
        device = key.lower()
        raise NotImplementedError(
            f"{op}: when running with device='{device}' tensors: there is no "
            f"{device} impl registered for this {API}. Please register one by "
            f"defining the {op}.impl_{device} staticmethod."
        )
    raise NotImplementedError(
        f"{op}: No implementation for dispatch key {key}. It is likely "
        f"that we have not added this functionality yet, please either open an "
        f"issue or use the low-level torch.library APIs."
    )


def mangle_module(module):
    """Mangles the module name.

    The scheme is replacing dots with some number of underscores
    (specified as mangledN where N is the number of underscores).

    Examples:
    foo.bar.baz -> mangled1_foo_bar_baz
    foo_bar.baz -> mangled2__foo_bar__baz
    foo.__baz__ -> mangled3___foo_____baz__

    Don't parse the mangled string directly; use mangle_module and demangle_module
    """
    sep = unique_underscoring(module)
    prefix = f"mangled{len(sep)}"
    splits = module.split(".")
    return sep.join([prefix, *splits])


def demangle_module(mangled_module):
    pass


def unique_underscoring(s: str):
    i = 1
    while True:
        result = "_" * i
        if result not in s:
            return result
        i += 1


def check_supported_schema(opdef, qualname):
    # We only support the following schemas, for now.
    # - functional
    # - auto_functionalizable.
    # For all others, we ask the user to go use the raw torch.library API.
    schema = qualname + opdef.schema
    import torch
    if torch._library.utils.is_functional_schema(schema):
        return
    # TODO(rzou):a put in final version
    import torch._higher_order_ops.auto_functionalize
    if torch._higher_order_ops.auto_functionalize.auto_functionalizable_schema(torch._C.parse_schema(schema)):
        return
    raise NotImplementedError(
        f"{API}: Tried to create an operator with unsupported schema "
        f"'{str(schema)}'. We support functional ops and mutable ops "
        f"where the outputs do not alias the inputs.")


def register_autograd(lib, name, opdef, op):
    # TODO: (1) autograd not found, (2) only support autograd for functional ops
    class MyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            # TODO: mark-dirty things
            with torch._C._AutoDispatchBelowAutograd():
                output = op(*inputs)
            if hasattr(opdef, "setup_backward"):
                opdef.setup_backward(ctx, inputs, output)
            return output

        @staticmethod
        def backward(ctx, *grads):
            return opdef.backward(ctx, *grads)

    lib.impl(name, MyFunction.apply, "Autograd")


class RegistersSubclassOnDefinition(type):
    def __new__(cls, name, bases, dct):
        result = super().__new__(cls, name, bases, dct)
        # This is the base class
        if name == "Operator":
            return result
        opoverload = register(result)
        result.opoverload = opoverload
        return result


class Operator(metaclass=RegistersSubclassOnDefinition):
    @classmethod
    def call(cls, *args, **kwargs):
        return cls.opoverload(*args, **kwargs)
