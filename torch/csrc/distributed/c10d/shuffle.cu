#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

constexpr int64_t BYTES_PER_THREAD = 16;
constexpr int64_t BLOCK_SIZE = 512;
constexpr int64_t TILE_SIZE = BYTES_PER_THREAD * BLOCK_SIZE;

namespace detail {
template <typename T>
__device__ inline void stream_load128(uint4& val, const T* addr) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
#else
  unsigned long long int low, high;
  asm("ld.global.nc.v2.u64 {%0, %1}, [%2];"
      : "=l"(low), "=l"(high)
      : "l"(addr));
  reinterpret_cast<unsigned long long int*>(&val)[0] = low;
  reinterpret_cast<unsigned long long int*>(&val)[1] = high;
#endif
}

template <typename T>
__device__ inline void stream_store128(T* addr, const uint4& val) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
#else
  unsigned long long int low, high;
  low = reinterpret_cast<const unsigned long long int*>(&val)[0];
  high = reinterpret_cast<const unsigned long long int*>(&val)[1];
  asm("st.global.cs.v2.u64 [%0], {%1, %2};" : : "l"(addr), "l"(low), "l"(high));
#endif
}

static __host__ __device__ inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

static __host__ __device__ inline int64_t minInt64(int64_t a, int64_t b) {
  return a < b ? a : b;
}

static __host__ __device__ inline int64_t maxInt64(int64_t a, int64_t b) {
  return a < b ? b : a;
}

static __device__ inline bool is_aligned(const void* ptr, size_t alignment) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  return addr % alignment == 0;
}

__device__ inline uint4 initialize() {
  uint4 zero;
  reinterpret_cast<uint64_t*>(&zero)[0] = 0;
  reinterpret_cast<uint64_t*>(&zero)[1] = 0;
  return zero;
}

std::pair<at::Tensor, std::vector<int64_t*>> pack(
    std::vector<std::vector<int64_t>> vecs,
    const at::Device& device) {
  int64_t numel = 0;
  for (const auto& vec : vecs) {
    numel += vec.size();
  }
  auto packed = at::empty(
      {numel}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  size_t offset = 0;
  for (const auto& vec : vecs) {
    memcpy(
        packed.data_ptr<int64_t>() + offset,
        vec.data(),
        sizeof(int64_t) * vec.size());
    offset += vec.size();
  }
  packed = packed.to(device, /*non_blocking=*/true);
  std::vector<int64_t*> ptrs;
  offset = 0;
  for (const auto& vec : vecs) {
    ptrs.push_back(packed.data_ptr<int64_t>() + offset);
    offset += vec.size();
  }
  return std::make_pair(packed, ptrs);
}

static __device__ __inline__ void get_aligned_region(
    char* ptr,
    const int64_t chunk_size,
    const int64_t alignment,
    int64_t& align_off,
    int64_t& aligned_size) {
  const int64_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  align_off = div_up(ptr_val, alignment) * alignment - ptr_val;
  aligned_size = (chunk_size - align_off) / alignment * alignment;
}

static __device__ __inline__ void copy_chunk_with_pad(
  char* dst,
  const char* src,
  int64_t max_chunk_size,
  int64_t actual_chunk_size,
  int64_t thread_idx,
  int64_t num_threads
) {
  if (max_chunk_size < num_threads) {
    if (thread_idx < actual_chunk_size) {
      dst[thread_idx] = src[thread_idx];
    } else {
      dst[thread_idx] = (char) 0;
    }
    return;
  }
  uint4 zero = initialize();
  int64_t align_off, aligned_size;
  get_aligned_region(dst, actual_chunk_size, BYTES_PER_THREAD, align_off, aligned_size);
  int64_t align_end = align_off + aligned_size;
  for (
    int64_t i = align_off + thread_idx * BYTES_PER_THREAD;
    i < align_end;
    i += num_threads * BYTES_PER_THREAD
  ) {
    uint4 val = zero;
    if(detail::is_aligned(src + i, BYTES_PER_THREAD)) {
      stream_load128(val, src + i);
    } else {
      for (size_t j = 0; j < BYTES_PER_THREAD; ++j) {
        reinterpret_cast<char*>(&val)[j] = src[i + j];
      }
    }
    stream_store128(&dst[i], val);
  }
  if(thread_idx < align_off && thread_idx < max_chunk_size) {
    char val = (char) 0;
    if (thread_idx < actual_chunk_size) {
      val = src[thread_idx];
    }
    dst[thread_idx] = val;
  }
  if(align_end + thread_idx < max_chunk_size) {
    char val = (char) 0;
    if (align_end + thread_idx < actual_chunk_size) {
      val = src[align_end + thread_idx];
    }
    dst[align_end + thread_idx] = val;
  }
}

static __global__ void resize_cat_cuda_kernel(
  char** src,
  char* dst,
  int64_t* block_idx_to_tensor_idx,
  int64_t* block_idx_to_start_tensor_bytes,
  int64_t* start_block_idx_per_tensor_chunk,
  int64_t* actual_tensor_sizes,
  int64_t* pad_tensor_chunk_sizes,
  int64_t* num_blocks_per_tensor_chunk,
  int64_t slice_size,
  int64_t chunk_size) {
  const int64_t slice_idx = blockIdx.z;
  const int64_t chunk_idx = blockIdx.y;
  const int64_t tensor_idx = block_idx_to_tensor_idx[blockIdx.x];
  const int64_t tile_idx = blockIdx.x - start_block_idx_per_tensor_chunk[tensor_idx];
  // Number of threads for the `tensor_idx`-th tensor chunk.
  const int64_t num_threads = num_blocks_per_tensor_chunk[tensor_idx] * BLOCK_SIZE;
  const char* src_addr = src[tensor_idx]
      + slice_idx * actual_tensor_sizes[tensor_idx]
      + chunk_idx * pad_tensor_chunk_sizes[tensor_idx]
      + tile_idx * TILE_SIZE;
  char* dst_addr = dst
      + slice_idx * slice_size
      + chunk_idx  * chunk_size
      + block_idx_to_start_tensor_bytes[tensor_idx]
      + tile_idx * TILE_SIZE;
  const int64_t actual_copy_size = minInt64(
    TILE_SIZE,
    maxInt64(
      actual_tensor_sizes[tensor_idx]
        -(chunk_idx * pad_tensor_chunk_sizes[tensor_idx] + tile_idx * TILE_SIZE),
      0)
  );
  if (actual_copy_size == 0) {
    return;
  }
  copy_chunk_with_pad(
    dst_addr,
    src_addr,
    pad_tensor_chunk_sizes[tensor_idx],
    actual_copy_size,
    threadIdx.x,
    num_threads
  );
}
} // namespace detail

void assert_leading_dimension_matches(
  std::vector<at::Tensor> tensors,
  uint64_t dim
) {
  const auto num_tensors = tensors.size();
  TORCH_CHECK(
    num_tensors > 0,
    "assert_leading_dimension_matches() has invalid args: should have at least 1 input tensors"
  );
  std::vector<c10::SymInt> leading_dim_sizes;
  for (const auto i : c10::irange(dim)) {
    leading_dim_sizes.push_back(tensors[0].size(i));
  }
  for (const auto i : c10::irange(num_tensors)) {
    at::Tensor tensor = tensors[i];
    TORCH_CHECK(tensor.numel() > 0, "assert_leading_dimension_matches() error: tensor should have at least 1 element");
    auto sizes = tensor.sizes();
    TORCH_CHECK(sizes.size() >= dim, "assert_leading_dimension_matches() error: invalid dim");
    for(const auto j : c10::irange(dim)) {
      TORCH_CHECK(
        tensor.size(j) == leading_dim_sizes[j],
        "resize_cat_cuda() has invalid args: tensors should have same sizes in the first dim dimensions"
      );
    }
  }
}

void resize_cat_cuda(
  std::vector<at::Tensor> tensors,
  int64_t dim,
  int64_t num_chunks,
  at::Tensor out
) {
  const auto device = out.device();
  auto num_tensors = tensors.size();
  TORCH_CHECK(
    out.is_cuda() && out.is_non_overlapping_and_dense(),
    "resize_cat_cuda() error: invalid out tensor"
  );
  assert_leading_dimension_matches(tensors, (uint64_t)dim);
  int64_t leading_dim = 1;
  if (dim > 0) {
    leading_dim = c10::multiply_integers(tensors[0].sizes().slice(0, dim));
  }
  std::vector<int64_t> pad_tensor_chunk_sizes;
  std::vector<int64_t> num_blocks_per_tensor_chunk;
  std::vector<int64_t> block_idx_to_tensor_idx;
  std::vector<int64_t> start_block_idx_per_tensor_chunk{0};
  std::vector<int64_t> actual_tensor_sizes;
  std::vector<int64_t> block_idx_to_start_tensor_bytes{0};
  std::vector<int64_t> srcs;
  for (const auto i : c10::irange(num_tensors)) {
    at::Tensor tensor = tensors[i];
    srcs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
    TORCH_CHECK(
      tensor.is_cuda() && tensor.is_non_overlapping_and_dense(),
      "resize_cat_cuda() error: invalid out tensor"
    );
    auto sizes = tensor.sizes();
    const int64_t size_along_dim = sizes[dim];
    int64_t tailing_numel = 1;
    if(sizes.size() > (uint64_t)dim + 1) {
      tailing_numel = c10::multiply_integers(sizes.slice(dim+1, sizes.size()-dim-1));
    }
    const int64_t pad_size_along_dim = detail::div_up(size_along_dim, num_chunks) * num_chunks;
    const int64_t pad_tensor_chunk_size = pad_size_along_dim * tailing_numel * tensor.element_size() / num_chunks;
    pad_tensor_chunk_sizes.push_back(pad_tensor_chunk_size);
    // Number of blocks required to process this tensor chunk.
    const int64_t num_blocks = detail::div_up(pad_tensor_chunk_size, TILE_SIZE);
    num_blocks_per_tensor_chunk.push_back(num_blocks);
    start_block_idx_per_tensor_chunk.push_back(start_block_idx_per_tensor_chunk.back() + num_blocks);
    block_idx_to_tensor_idx.insert(block_idx_to_tensor_idx.end(), num_blocks, i);
    actual_tensor_sizes.push_back(size_along_dim * tailing_numel * tensor.element_size());
    block_idx_to_start_tensor_bytes.push_back(block_idx_to_start_tensor_bytes.back() + pad_tensor_chunk_size);
  }
  const int64_t num_blocks_per_chunk = start_block_idx_per_tensor_chunk.back();
  const int64_t chunk_size = pad_tensor_chunk_sizes.back();
  const int64_t slice_size = num_chunks * chunk_size;
  auto packed = detail::pack(
    {srcs, block_idx_to_tensor_idx, block_idx_to_start_tensor_bytes, start_block_idx_per_tensor_chunk, actual_tensor_sizes, pad_tensor_chunk_sizes, num_blocks_per_tensor_chunk}, device
  );
  dim3 blocks(num_blocks_per_chunk, num_chunks, leading_dim);
  dim3 threads(BLOCK_SIZE, 1, 1);
  detail::resize_cat_cuda_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    /*srcs=*/reinterpret_cast<char**>(packed.second[0]),
    reinterpret_cast<char*>(out.data_ptr()),
    /*block_idx_to_tensor_idx=*/packed.second[1],
    /*block_idx_to_start_tensor_bytes=*/packed.second[2],
    /*start_block_idx_per_tensor_chunk=*/packed.second[3],
    /*actual_tensor_sizes=*/packed.second[4],
    /*pad_tensor_chunk_sizes=*/packed.second[5],
    /*num_blocks_per_tensor_chunk=*/packed.second[6],
    slice_size,
    chunk_size
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
