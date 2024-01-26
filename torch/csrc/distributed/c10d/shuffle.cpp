#include <ATen/ATen.h>
#include <torch/library.h>
// TODO: Think about file name. Should it be shuffle.cpp or something else?

#ifdef USE_CUDA
void resize_cat_cuda(
    std::vector<at::Tensor> tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor out
);
#endif

namespace {
void resize_cat(
  std::vector<at::Tensor> tensors,
  int64_t dim,
  int64_t num_chunks,
  at::Tensor out
) {
#ifdef USE_CUDA
  return resize_cat_cuda(tensors, dim, num_chunks, out);
#else
  C10_THROW_ERROR(NotImplementedError, "Not implemented for CPU");
#endif
}
} // namespace

TORCH_LIBRARY_FRAGMENT(c10d, m) {
  m.def(
    "resize_cat("
    "Tensor[] tensors, int dim, int num_chunks, Tensor out) -> ()",
    torch::dispatch(
      c10::DispatchKey::CompositeExplicitAutograd,
      ::resize_cat),
      {at::Tag::pt2_compliant_tag});
}
