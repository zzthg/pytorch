import contextlib
import dataclasses
import functools
import tempfile
from typing import Dict, List

import torch
import torch.fx._pytree as fx_pytree
from torch._inductor.utils import aot_inductor_launcher, cache_dir
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


@contextlib.contextmanager
def enable_python_dispatch():
    return torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Python, False)


class EagerCompileRegistry:
    def __init__(self):
        self.libraries = {}

    def register(self, op, dispatch_key):
        def _op_impl(op, *args, **kwargs):
            op_aot_compiled = aot_cache.lookup(op, *args, **kwargs)
            if op_aot_compiled is not None:
                return op_aot_compiled(*args, **kwargs)
            else:
                op_compiled = torch.compile(op)
                with enable_python_dispatch():
                    return op_compiled(*args, **kwargs)

        name = (
            op.__name__
            if op._overloadname != "default"
            else op.__name__[: -len(".default")]
        )
        print(f"register {name}")
        lib = self.libraries.get(op.namespace, None)
        if lib is None:
            lib = torch.library.Library(op.namespace, "IMPL", dispatch_key)
            self.libraries[op.namespace] = lib
        lib.impl(name, functools.partial(_op_impl, op))


class AotCache:
    def __init__(self):
        self.cache = {}

    def lookup(self, op, *args, **kwargs):
        return self.cache.get(op.name(), None)

    def add(self, op, *args, **kwargs):
        def load(device, so_path):
            module = torch.utils.cpp_extension.load_inline(
                name="aot_inductor",
                cpp_sources=[aot_inductor_launcher(so_path, device)],
                # use a unique build directory to avoid test interference
                build_directory=tempfile.mkdtemp(dir=cache_dir()),
                functions=["run", "get_call_spec"],
                with_cuda=(device == "cuda"),
                use_pch=True,
            )

            call_spec = module.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])

            def optimized(*args):
                flat_inputs = fx_pytree.tree_flatten_spec((*args, {}), in_spec)
                flat_outputs = module.run(flat_inputs)
                return pytree.tree_unflatten(flat_outputs, out_spec)

            return optimized

        if op.name() not in self.cache:
            so_path = torch._export.aot_compile(op, args, kwargs)
            # TODO: support other devices than cpu
            self.cache[op.name()] = load("cpu", so_path)
            return self.cache[op.name()]


registry = EagerCompileRegistry()
aot_cache = AotCache()


def register_eager_compile_for_fn(
    dispatch_key, fn, args=(), kwargs=None, ignore_op_fn=None
):
    @dataclasses.dataclass
    class Record:
        op: torch._ops.OpOverload
        args: tuple
        kwargs: dict

    class Recorder(TorchDispatchMode):
        def __init__(self):
            self.op_records: Dict[str, List[Record]] = {}

        def __torch_dispatch__(self, op, types, args=(), kwargs=None):
            kwargs = kwargs if kwargs else {}
            record = Record(op, args, kwargs)
            records = self.op_records.get(op.name(), [])
            records.append(record)
            self.op_records[op.name()] = records
            return op(*args, **kwargs)

    kwargs = kwargs if kwargs else {}
    recorder = Recorder()
    with recorder:
        fn(*args, **kwargs)
    for op_name, records in recorder.op_records.items():
        assert records
        if ignore_op_fn is not None and ignore_op_fn(records[0].op):
            continue
        for record in records:
            aot_cache.add(record.op, *record.args, **record.kwargs)
        registry.register(records[0].op, dispatch_key)
