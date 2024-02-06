import contextlib
import inspect
import logging

import torch
import torch._dynamo as torchdynamo
import torch._inductor
import torch._inductor.config


# Include optimizer code for tracing
optim_filenames = set(
    [
        inspect.getfile(obj)
        for obj in torch.optim.__dict__.values()
        if inspect.isclass(obj)
    ]
)


optim_filenames |= {torch.optim._functional.__file__}


@contextlib.contextmanager
def enable_optimizer_tracing():
    try:
        old = set(torch._dynamo.skipfiles.FILENAME_ALLOWLIST)
        torch._dynamo.skipfiles.FILENAME_ALLOWLIST.update(optim_filenames)
        yield
    finally:
        torch._dynamo.skipfiles.FILENAME_ALLOWLIST.clear()
        torch._dynamo.skipfiles.FILENAME_ALLOWLIST.update(old)


def get_hf_model(name):
    import importlib

    module = importlib.import_module("transformers")
    model_cls = getattr(module, name)
    config_cls = model_cls.config_class
    config = config_cls()
    if "auto" in model_cls.__module__:
        # Handle auto classes
        model = model_cls.from_config(config).to(device="cuda", dtype=torch.float32)
    else:
        model = model_cls(config).to(device="cuda", dtype=torch.float32)

    return model


# torchdynamo.config.log_level = logging.DEBUG
# torch._inductor.config.trace.enabled = False
# torch._inductor.config.trace.graph_diagram = False
# torch._inductor.config.aggressive_fusion = True

# torch._logging.set_logs(fusion=True)


def opt_trace_benchmark(opt_ctor, **kwargs):
    print(f"Running tracing benchmark on {opt_ctor.__name__}")
    for param_count in [1000]:
        torchdynamo.reset()
        model = torch.nn.Sequential(
            *[
                torch.nn.Linear(100, 100, False, device="cuda")
                for _ in range(param_count)
            ]
        )
        # model = get_hf_model("MobileBertForMaskedLM")
        # model = get_hf_model("DebertaV2ForQuestionAnswering")
        optimizer = opt_ctor(model.parameters(), **kwargs)
        # x = torch.ones(100, dtype=torch.float, device="cuda:0")
        # y = model(x)
        # torch.sum(y).backward()
        for p in model.parameters():
            p.grad = torch.rand_like(p)

        import time

        @torchdynamo.optimize("inductor")
        def repro():
            optimizer.step()

        print(f"param count: {len(list(model.parameters()))}")
        t0 = time.perf_counter()
        # with torch.profiler.profile() as prof:
        #    repro()
        repro()
        t1 = time.perf_counter()
        # prof.export_chrome_trace("opt_trace_bench.json")
        print(f"param count: {len(list(model.parameters()))}, trace latency: {t1 - t0}")


opt_trace_benchmark(
    torch.optim.Adam,
    lr=0.001,
    weight_decay=0.01,
    foreach=True,
    capturable=True,
    # fused=True,
)
