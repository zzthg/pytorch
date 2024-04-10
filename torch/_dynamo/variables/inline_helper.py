import torch.fx
from .constant import ConstantVariable


def dummy_user_function_to_inline_gm(gm, args):
    return gm(*args)


def dummy_user_function_to_inline_wrapped_gm(wrapped_gm, args, kwargs):
    return wrapped_gm(args, kwargs)


def vt_to_fake_helper(vt, tx):
    from ..utils import get_fake_value

    proxy_ = vt.as_proxy()

    def proxy_to_fake_helper(p):
        if type(p) is torch.fx.proxy.Proxy:
            return get_fake_value(p.node, tx)
        elif type(p) is tuple:
            return tuple(map(proxy_to_fake_helper, p))
        else:
            # mostly handle scalar
            # check return type is a fake tensor
            assert type(p) != torch.fx.proxy.Proxy
            return p

    return proxy_to_fake_helper(proxy_)


tracer_to_used_names = {}


def reconstruct_node_meta_data(module_vt, tx, num_nodes_need_update_metadata):
    for node in tx.output.graph.nodes.__reversed__():
        num_nodes_need_update_metadata -= 1
        if num_nodes_need_update_metadata < 0:
            break
        # restore the source_fn_stack to be nn module.
        if "source_fn_stack" in node.meta and len(node.meta["source_fn_stack"]) > 0:
            # below logic to get a unique name for source_fn_stack is mimic from
            # the _Namespace.create_name() which is used to get a unique name for
            # the fx node.
            if tx.output.current_tracer not in tracer_to_used_names.keys():
                # TODO(JackCaoG): use weakref here?
                tracer_to_used_names[tx.output.current_tracer] = {}

            base_module_key = module_vt.module_key.lower()

            if (
                base_module_key
                not in tracer_to_used_names[tx.output.current_tracer].keys()
            ):
                tracer_to_used_names[tx.output.current_tracer][base_module_key] = 0

            count = tracer_to_used_names[tx.output.current_tracer][base_module_key]
            tracer_to_used_names[tx.output.current_tracer][base_module_key] += 1
            unique_module_key = (
                base_module_key if count == 0 else f"{base_module_key}_{count}"
            )
            node.meta["source_fn_stack"][-1] = (
                unique_module_key,
                type(module_vt.module),
            )
        # remove the additional stack trace caused by fwd inlining
        if "stack_trace" in node.meta and len(node.meta["stack_trace"]) > 0:
            splited = node.meta["stack_trace"].split("\n")
            # handle the cases where make_fx is called.
            if len(splited) > 7 and "_dynamo/variables/inline_helper.py" in splited[-5]:
                node.meta["stack_trace"] = "\n".join(splited[:-7]) + "\n"
            # handle the case for lifted parameters.
            elif (
                len(splited) > 4
                and "return forward_call(*args, **kwargs)" in splited[-2]
            ):
                node.meta["stack_trace"] = "\n".join(splited[:-3]) + "\n"


code_to_Fx = {}


def decompose_and_inline_function_with_makefx(tx, fn, args, kwargs):
    from functorch import make_fx

    from torch._dispatch.python import enable_python_dispatcher
    from torch._guards import detect_fake_mode
    from .base import MutableLocal
    from .builder import SourcelessBuilder
    from .dicts import ConstDictVariable
    from .lists import BaseListVariable

    # convert arguments from VariableTracker to fake tensors + constants again
    fake_value_args = []
    for arg in args:
        fake_value_args.append(vt_to_fake_helper(arg, tx))

    fake_value_kwargs = {}
    for key, value in kwargs.items():
        fake_value_kwargs[key] = vt_to_fake_helper(value, tx)

    # Wrap the function before calling make_fx to avoid make_fx modify the kwargs's key.
    def wrapper_fn(fn):
        def inner(arg, kwargs):
            return fn(*arg, **kwargs)

        return inner

    wrapped_fn = wrapper_fn(fn)
    fake_mode = detect_fake_mode(fake_value_args)

    with fake_mode:
        with enable_python_dispatcher():
            fx_g = make_fx(wrapped_fn, pre_dispatch=True)(
                fake_value_args, fake_value_kwargs
            )

    # this is a hack, we want to access `.code` here to trigger the `real_recompile`
    # in case this is `_lazy_graph_module`. This will aovid us trying to inline the
    # `_LazyGraphModule._lazy_forward`(in the skip list) below.
    code = fx_g.code

    # make_fx on the same nn_module we will create function with different names.
    # SpeculationLog will replay the dynamo tracing upon graph break and it expects
    # to see the same functon name. It is safer to rerun the `make_Fx` and use the
    # cached fx only if code is the same.
    if code in code_to_Fx:
        fx_g = code_to_Fx[code]
    else:
        code_to_Fx[code] = fx_g

    # now inline this fx graph and return the output
    user_fn_variable_with_kwargs = SourcelessBuilder.create(
        tx, dummy_user_function_to_inline_wrapped_gm
    )
    gm_variable = SourcelessBuilder.create(tx, fx_g)
    cls = BaseListVariable.cls_for(list)
    input_args_variable = cls(
        args,
        mutable_local=MutableLocal(),
    )

    # kwarg's key needs to be turn into VariableTracker before passing
    # to ConstDictVariable.
    updated_kwargs = {}
    for k, v in kwargs.items():
        updated_kwargs[ConstantVariable.create(k)] = v

    input_kwargs_variable = ConstDictVariable(
        updated_kwargs,
        dict,
        mutable_local=MutableLocal(),
    )
    res = tx.inline_user_function_return(
        user_fn_variable_with_kwargs,
        (gm_variable, input_args_variable, input_kwargs_variable),
        {},
    )
    return res
