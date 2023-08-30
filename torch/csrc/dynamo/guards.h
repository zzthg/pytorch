#pragma once
#include <torch/csrc/python_headers.h>

PyObject* torch_c_dynamo_guards_init();
void torch_c_dynamo_new_guards_init(PyObject*);
