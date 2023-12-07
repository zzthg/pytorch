#pragma once
#include <torch/csrc/utils/python_stub.h>

// see [Note: Compiled Autograd]
namespace torch::dynamo::autograd {
PyObject* torch_c_dynamo_compiled_autograd_init();
bool is_eager_compile();
} // namespace torch::dynamo::autograd
