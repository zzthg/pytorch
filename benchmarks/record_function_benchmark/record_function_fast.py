import time

import torch
from torch.autograd.profiler import profile as tprofile
profiler_events = []
is_enabled = False

from memory_profiler import profile

def record_time(fn, name):
    for _ in range(100):
        fn()
    start = time.time()
    for _ in range(1000):
        fn()
    end = time.time()
    print(f"{name}:: {(end-start)} us")

x, y = (torch.rand((4, 4)) for _ in range(2))
def test_record_function_no_args():
        with tprofile() as p:
             with torch._C._profiler._RecordFunctionFast("add_test_fast_rf1"):
                 x.view_as(x)

record_time(test_record_function_no_args, "No shape or args")

def test_record_function_rec_shape():
        with tprofile(record_shapes=True) as p:
             with torch._C._profiler._RecordFunctionFast("add_test_fast_rf1", [x]):
                 x.view_as(x)
record_time(test_record_function_rec_shape, "With shape")
def test_record_function_args_no_shape():
        with tprofile(record_shapes=False) as p:
             with torch._C._profiler._RecordFunctionFast("add_test_fast_rf1", [x]):
                 x.view_as(x)
record_time(test_record_function_args_no_shape, "Args no shape")

@profile
def test_rec_function_profile_no_args():
    for _ in range(1000):
        with tprofile() as p:
             with torch._C._profiler._RecordFunctionFast("add_test_fast_rf1"):
                 x.view_as(x)

test_rec_function_profile_no_args()

@profile
def test_rec_function_profile_args():
    for _ in range(1000):
        with tprofile() as p:
             with torch._C._profiler._RecordFunctionFast("add_test_fast_rf1", [x]):
                 x.view_as(x)

test_rec_function_profile_args()



