import argparse
import contextlib

import os
from typing import List, Optional, Tuple

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import _functional_collectives as collectives
import torchbenchmark
import subprocess
from torch.nn.parallel import DistributedDataParallel as DDP

from torchbench import TorchBenchmarkRunner
from common import parse_args, patch_torch_manual_seed, cast_to_fp16

import contextlib

def bench(
    iter_func,
    iters=8,
    warmup=5,
    profile=False,
    device=None,
    model_name=None,
):
    assert device is not None

    dynamo_config.suppress_errors = False

    f_ = iter_func

    repeat = 5
    f = lambda: [(f_(), torch.cuda.synchronize()) for _ in range(repeat)]
    import time

    # measure memory on cold run
    torch.cuda.reset_peak_memory_stats(device)
    f()
    torch.cuda.synchronize()
    f_gb = torch.cuda.max_memory_allocated(device) / 1e9

    for _ in range(warmup):
        f()

    if profile:
        if dist.get_rank() == 0:
            prof = torch.profiler.profile()
        else:
            prof = contextlib.nullcontext()
        with prof:
            f()
        if dist.get_rank() == 0:
            prof.export_chrome_trace(f"{model_name}.json")
    f_times = []

    for _ in range(iters):
        # Calculate the elapsed time
        torch.cuda.synchronize(device)
        begin = time.time()
        f()
        torch.cuda.synchronize(device)
        f_times.append(time.time() - begin)

    # avg_time = sum(f_times)*1000/repeat/len(f_times)
    # print(f"{model_name}: avg_time    : {avg_time} ms \t{f_gb}GB")

    return f_times, f_gb


def run_one_rank(
    my_rank,
    args,
    runner,
    compile
):
    global print
    if my_rank != 0:
        print = lambda *args, **kwargs: None

    torch.cuda.set_device(my_rank)
    device = torch.device(f"cuda:{my_rank}")

    os.environ["RANK"] = f"{my_rank}"
    os.environ["WORLD_SIZE"] = f"{args.world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    nccl_options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", pg_options=nccl_options
    )

    (
        _,
        name,
        model,
        example_inputs,
        batch_size,
    ) = runner.load_model("cuda", args.only, batch_size=args.batch_size)

    model, example_inputs = cast_to_fp16(model, example_inputs)

    if args.accuracy:
        torch._inductor.config.fallback_random = True
        if args.only not in {
            "alexnet",
            "Background_Matting",
            "pytorch_CycleGAN_and_pix2pix",
            "pytorch_unet",
            "Super_SloMo",
            "vgg16",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForPreTraining",
            "sam",
        }:
            # some of the models do not support use_deterministic_algorithms
            torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

        # Remove randomeness when torch manual seed is called
        patch_torch_manual_seed()

    def run_eager():
        model_eager = model
        if args.ddp:
            model_eager = DDP(
                model,
                device_ids=[my_rank],
                output_device=my_rank,
                # bucket_cap_mb=25,  # DDP default value
            )

        runner.optimizer = torch.optim.SGD(model_eager.parameters(), lr=0.01, foreach=True)

        maybe_amp = contextlib.nullcontext
        if args.amp:
            maybe_amp = torch.cuda.amp.autocast

        with maybe_amp():
            eager_times, f_gb = bench(
                lambda: runner.model_iter_fn(model_eager, example_inputs, collect_outputs=False),
                profile=args.export_profiler_trace,
                device=device,
                model_name=f"{args.only}_eager"
            )

        assert len(eager_times) == 8
        print(f"eager    : {(sum(eager_times) / len(eager_times) * 1000):.3f} ms \t{f_gb} GB")

    def run_compile():
        # NOTE: throws `daemonic processes are not allowed to have children` error at `AsyncCompile.warm_pool() -> pool._adjust_process_count()` if we don't set this to 1.
        inductor_config.compile_threads = 1
        torch._inductor.config.triton.cudagraphs = not args.disable_cudagraphs
        if not args.disable_cudagraphs:
            torch.profiler._utils._init_for_cuda_graphs()

        if args.ddp:
            model_compiled = DDP(
                torch.compile(model),# mode="reduce-overhead"),
                device_ids=[my_rank],
                output_device=my_rank,
                # bucket_cap_mb=args.ddp_bucket_cap_mb_for_compiled
            )
        else:
            model_compiled = torch.compile(model),# mode="reduce-overhead")

        runner.optimizer = torch.optim.SGD(model_compiled.parameters(), lr=0.01, foreach=True)

        maybe_amp = contextlib.nullcontext
        if args.amp:
            maybe_amp = torch.cuda.amp.autocast

        with maybe_amp():
            compiled_times, g_gb = bench(
                lambda: runner.model_iter_fn(model_compiled, example_inputs),
                profile=args.export_profiler_trace,
                device=device,
                model_name=f"{args.only}_compiled"
            )

        assert len(compiled_times) == 8
        print(f"compiled : {(sum(compiled_times) / len(compiled_times) * 1000):.3f} ms \t{g_gb} GB")

    run_compile() if compile else run_eager()

def main(compile: bool):
    args = parse_args()
    args.world_size = 1

    runner = TorchBenchmarkRunner()
    runner.args = args
    runner.model_iter_fn = runner.forward_and_backward_pass

    processes = []
    for rank in range(args.world_size):
        p = torch.multiprocessing.get_context("spawn").Process(
            target=run_one_rank,
            args=(
                rank,
                args,
                runner,
                compile
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    for rank, p in enumerate(processes):
        p.join()

if __name__ == "__main__":
    main(compile=False)
    main(compile=True)
