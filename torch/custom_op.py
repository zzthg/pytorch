import torch
from torch.library import Library

KEEP_ALIVE = []
op_to_kernel = {}

def triton_op(
    name,
    triton_schema,
    triton_kernel,
    grid_fn,
    abstract_impl,
):
    ns, name = name.split("::")
    lib = Library(ns, 'FRAGMENT')
    KEEP_ALIVE.append(lib)

    functional_schema, mappings = process_schema(triton_schema)
    functional_schema = f"{name}{functional_schema}"
    lib.define(functional_schema)

    # Define abstract impl
    # TODO: should really use CustomOp.impl_abstract, but we need to refactor that
    # a bit...
    lib.impl(name, abstract_impl, "Meta")

    op_to_kernel

    # Define backend implementation
    backend_impl = construct_backend_impl(abstract_impl, triton_kernel, grid_fn, mappings)
    lib.impl(name, backend_impl, "CompositeExplicitAutograd")

    result = getattr(getattr(torch.ops, ns), name).default
    op_to_kernel[f"{ns}::{name}"] = (triton_kernel, grid_fn, abstract_impl)
    return result


def process_schema(triton_schema):
    # 1) construct a functional schema
    # 2) Construct a mapping from:
    #    inputs of the functional schema -> inputs of the triton_schema
    #    outputs of the functional schema -> inputs of the triton_schema
    args = [arg.strip().split(" ") for arg in triton_schema.split(',')]
    functional_args = [arg for arg in args if arg[0] != 'Output']
    num_outputs = len(args) - len(functional_args)

    functional_schema = ", ".join([" ".join(arg) for arg in functional_args])
    functional_schema = f"({functional_schema}) -> ({','.join(['Tensor'] * num_outputs)})"

    inputs_mapping = []
    outputs_mapping = []
    for idx, (typ, name) in enumerate(args):
        if typ == "Output":
            outputs_mapping.append(idx)
        else:
            inputs_mapping.append(idx)
    return functional_schema, (inputs_mapping, outputs_mapping)


def construct_backend_impl(abstract_impl, triton_kernel, grid_fn, mappings):
    def inner(*args):
        orig_outputs = abstract_impl(*args)
        if isinstance(orig_outputs, torch.Tensor):
            outputs = (orig_outputs,)
        triton_args = [None] * (len(args) + len(outputs))

        inputs_mapping, outputs_mapping = mappings
        for arg, idx in zip(args, inputs_mapping):
            triton_args[idx] = arg
        for out, idx in zip(outputs, outputs_mapping):
            triton_args[idx] = out

        grid = grid_fn(*args)
        triton_kernel[grid](*triton_args)
        return orig_outputs
    return inner
