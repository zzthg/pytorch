import contextlib
import inspect
import itertools
import logging

import torch
import torch._dynamo as torchdynamo
import torch._inductor


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


torchdynamo.config.log_level = logging.DEBUG
torchdynamo.config.print_graph_breaks = False


def param_seq():
    return itertools.cycle([100, 200, 300, 400, 300, 200, 100])


def pairwise(itr):
    a, b = itertools.tee(itr)
    next(b, None)
    return zip(a, b)


def model_seq():
    for in_f, out_f in pairwise(param_seq()):
        yield torch.nn.Linear(in_f, out_f, dtype=torch.float32, device="cuda:0")


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / 10**9


# torch.cuda.reset_peak_memory_stats()
# torch.cuda.empty_cache()


def mem_bench(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    for _ in range(5):
        fn()
    return get_peak_memory()


def opt_trace_benchmark(opt_ctor, **kwargs):
    print(f"Running tracing benchmark on {opt_ctor.__name__}")
    for param_count in [500]:
        torchdynamo.reset()
        model = torch.nn.Linear(1000, 1000, device="cuda:0")
        optimizer = opt_ctor(model.parameters())
        x = torch.ones(1000, dtype=torch.float, device="cuda:0")

        def model_step(x):
            optimizer.zero_grad(True)
            y = model(x)
            torch.sum(y).backward()
            optimizer.step()

        model_opt = torchdynamo.optimize("eager")(model_step)


# opt_trace_benchmark(lambda params: torch.optim.SGD(params, lr=0.01))


def small_fn(x, y, lr):
    x.add_(y, alpha=-lr)


opt_small_fn = torchdynamo.optimize("inductor")(small_fn)

opt_res = mem_bench(
    lambda: opt_small_fn(
        torch.ones(37, 23, 3, 3, dtype=torch.float32, device="cuda:0"),
        torch.ones(37, 23, 3, 3, dtype=torch.float32, device="cuda:0"),
        0.01,
    )
)
eager_res = mem_bench(
    lambda: small_fn(
        torch.ones(37, 23, 3, 3, dtype=torch.float32, device="cuda:0"),
        torch.ones(37, 23, 3, 3, dtype=torch.float32, device="cuda:0"),
        0.01,
    )
)
print(f"optimized: {opt_res}")
print(f"eager: {eager_res}")


# 80 zigzag params
# foreach False Eager param count: 1000, latency: 7.036989330779761
# foreach True Eager param count: 1000, latency: 1.9389240988530219

# 30 zigzag params
# foreach False Inductor param count: 1000, latency: 0.9155371068045497
# foreach True Inductor param count: 1000, latency: 0.23188304575160146
# foreach True Eager param count: 1000, latency: 0.9454238400794566
# foreach False Eager param count: 1000, latency: 1.9389240988530219

# foreach True Inductor param count: 1000, latency: 0.05797715578228235
# foreach True Eager param count: 1000, latency: 0.2671662042848766
# foreach False Inductor param count: 1000, latency: 0.15395703399553895
# foreach False Eager param count: 1000, latency: 0.4496450489386916
