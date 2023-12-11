import torch
import torch.fx
import torch.utils._pytree as pytree
import dataclasses
from torch.fx.experimental.proxy_tensor import make_fx

def test_mm():
    x = torch.randn(2, 3)
    y = torch.randn(3, 5)
    y = y + 1
    y.zero_()
    result = torch.ops.aten.mm(x, y)
    assert torch.allclose(result, x @ y)

import torch._custom_ops as custom_ops
import torch.testing._internal.custom_op_db
import torch.testing._internal.optests as optests
from functorch import make_fx
from torch import Tensor
from torch._custom_op.impl import custom_op, CustomOp
from torch._utils_internal import get_file_path_2
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.custom_op_db import custom_op_db
from typing import *  # noqa: F403
from torch.testing._internal.common_utils import *  # noqa: F403
from torch.testing._internal.common_device_type import *  # noqa: F403
import unittest


class CustomOpTestCaseBase(TestCase):
    test_ns = "_test_custom_op"

    def setUp(self):
        self.libraries = []

    def tearDown(self):
        import torch._custom_op

        keys = list(torch._custom_op.impl.global_registry.keys())
        for key in keys:
            if not key.startswith(f"{self.test_ns}::"):
                continue
            torch._custom_op.impl.global_registry[key]._destroy()
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        for lib in self.libraries:
            lib._destroy()
        del self.libraries

    def ns(self):
        return getattr(torch.ops, self.test_ns)

    def lib(self):
        result = torch.library.Library(self.test_ns, "FRAGMENT")
        self.libraries.append(result)
        return result

    def get_op(self, qualname):
        return torch._custom_op.impl.get_op(qualname)

class MiniOpTest(CustomOpTestCaseBase):
    test_ns = "mini_op_test"

    def _init_op_delayed_backward_error(self):
        name = "delayed_error"
        qualname = f"{self.test_ns}::{name}"
        lib = self.lib()
        lib.define(f"{name}(Tensor x) -> Tensor")
        lib.impl(name, lambda x: x.clone(), "CompositeExplicitAutograd")
        op = self.get_op(qualname)

        class Op(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, grad):
                raise NotImplementedError()

        def autograd_impl(x):
            return Op.apply(x)

        lib.impl(name, autograd_impl, "Autograd")
        return op

    def _init_op_with_no_abstract_impl(self):
        name = "no_abstract"
        qualname = f"{self.test_ns}::{name}"
        lib = self.lib()
        lib.define(f"{name}(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,))
        lib.impl(name, lambda x: x.clone(), "CPU")
        return torch._library.utils.lookup_op(qualname)

    def setUp(self):
        super().setUp()
        self._op_with_no_abstract_impl = self._init_op_with_no_abstract_impl()
        self._op_delayed_backward_error = self._init_op_delayed_backward_error()

    @optests.dontGenerateOpCheckTests("Testing this API")
    def test_dont_generate(self):
        op = op_with_incorrect_schema(self, "incorrect_schema")
        x = torch.randn(3)
        op(x)

    def test_mm(self):
        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(3, 5)
        result = torch.ops.aten.mm.default(x, y)
        self.assertEqual(result, x @ y)

    def test_mm_meta(self):
        x = torch.randn(2, 3, requires_grad=True, device="meta")
        y = torch.randn(3, 5, device="meta")
        result = torch.ops.aten.mm.default(x, y)
        self.assertEqual(result.shape, (x @ y).shape)

    def test_mm_fake(self):
        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.randn(2, 3, requires_grad=True, device="cpu")
            y = torch.randn(3, 5, device="cpu")
            result = torch.ops.aten.mm.default(x, y)
            self.assertEqual(result.shape, (x @ y).shape)

    def test_mm_errors(self):
        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(4, 5)
        with self.assertRaisesRegex(RuntimeError, "cannot be multiplied"):
            result = torch.ops.aten.mm.default(x, y)

    def test_nonzero(self):
        x = torch.tensor([0, 1, 2, 0, 0])
        y = torch.ops.aten.nonzero.default(x)
        self.assertEqual(y, torch.tensor([[1], [2]]))

    def test_inplace(self):
        x = torch.randn(3)
        x_clone = x.clone()
        y = torch.ops.aten.sin_(x)
        self.assertEqual(x, x_clone.sin())

    def test_incorrect_schema(self):
        op = op_with_incorrect_schema(self, "incorrect_schema")
        x = torch.randn(3)
        op(x)

    def test_no_abstract(self):
        op = self._op_with_no_abstract_impl
        x = torch.randn(3)
        op(x)

    def test_delayed_error(self):
        op = self._op_delayed_backward_error
        x = torch.randn([], requires_grad=True)
        y = op(x)
        with self.assertRaises(NotImplementedError):
            y.sum().backward()

    def test_delayed_error_no_requires_grad(self):
        op = self._op_delayed_backward_error
        x = torch.randn([])
        y = op(x)


def extract_init_graph(graph, arg_nodes, kwarg_nodes=None):
    if kwarg_nodes is None:
        kwarg_nodes = {}

    # Step 1: find all nodes in the graph that we want to
    # copy over to a new graph
    
    # This is an "ordered set". Ordering matters when we copy the
    # nodes over (they must be copied in topologically sorted order).
    all_nodes_to_copy = {}
    
    def visit_node(maybe_node):
        if not isinstance(maybe_node, torch.fx.Node):
            return
        node = maybe_node
        if node in all_nodes_to_copy:
            return
        for arg in node.args:
            visit_node(arg)
        for _, kwarg in node.kwargs.items():
            visit_node(kwarg)
        all_nodes_to_copy[maybe_node] = None

    for node in arg_nodes:
        visit_node(node)
    for node in kwarg_nodes.values():
        visit_node(node)

    # Step 2: actually copy them over to the new graph
    new_graph = torch.fx.Graph()
    env = {}
    for node in all_nodes_to_copy.keys():
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    output_args = []
    for node in arg_nodes:
        output_args.append(env[node] if isinstance(node, torch.fx.Node) else node)
    output_kwargs = {}
    for name, node in output_kwargs:
        output_kwargs[name] = env[node] if isinstance(node, torch.fx.Node) else node
    new_graph.output((tuple(output_args), output_kwargs))
    new_graph.lint()
    return new_graph


def extract_sample_input_graphs(fn, op_filter):
    gm = make_fx(fn, tracing_mode="real", pre_dispatch=True)()
    extracted_graphs = []
    for node in gm.graph.nodes: 
        if node.op != "call_function":
            continue
        if not isinstance(node.target, torch._ops.OpOverload):
            continue
        if not op_filter(node.target):
            continue
        init_graph = extract_init_graph(gm, node.args, node.kwargs)
        extracted_graphs.append((node.target, init_graph))
    return extracted_graphs


def print_graph(graph):
    print(graph.python_code("self").src)


def graph_to_python(graph):
    result = [
        "import torch",
        "import torch.fx",
    ] 
    python_code = graph.python_code("self")
    _custom_builtins = torch.fx.graph._custom_builtins
    for symbol, value in python_code.globals.items():
        if symbol in _custom_builtins and value is _custom_builtins[symbol].obj:
            result.append(_custom_builtins[symbol].import_str)
        else:
            raise NotImplementedError("how to reconstruct symbol?")
    result.append(python_code.src)
    return "\n".join(result)


import unittest
attrs = dir(MiniOpTest)
attrs = [attr for attr in attrs if attr.startswith("test_")]

def get_test(name):
    def f():
        singletest = unittest.TestSuite()
        singletest.addTest(MiniOpTest(name))
        unittest.TextTestRunner().run(singletest)
    return f

results = {}
for attr in attrs:
    graphs = extract_sample_input_graphs(get_test(attr), lambda op: True)
    for op, graph in graphs:
        if op not in results:
            results[op] = []
        results[op].append(graph_to_python(graph))


breakpoint()


