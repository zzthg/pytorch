#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

constexpr int64_t BYTES_PER_THREAD = 16;
constexpr int64_t MAX_NUM_THREADS = 1024;
constexpr int64_t MIN_NUM_THREADS = 128;
constexpr int64_t WARP_SIZE = 32;
constexpr int64_t BLOCK_SIZE = 512;

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
  int64_t num_threads,
  uint4 default_value
) {
  if (max_chunk_size < num_threads) {
    if (thread_idx < actual_chunk_size) {
      dst[thread_idx] = src[thread_idx];
    } else {
      dst[thread_idx] = (char) 0;
    }
    return;
  }
  int64_t align_off, aligned_size;
  get_aligned_region(dst, actual_chunk_size, BYTES_PER_THREAD, align_off, aligned_size);
  int64_t align_end = align_off + aligned_size;
  for (
    int64_t i = align_off + thread_idx * BYTES_PER_THREAD;
    i < align_end;
    i += num_threads * BYTES_PER_THREAD
  ) {
    uint4 val = default_value;
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

static __global__ void unflatten_cat_with_pad_kernel(
  char** src,
  int64_t num_chunks,
  char* dst,
  int64_t* chunk_offset_to_tensor_idx,
  int64_t* cum_sum_blocks_per_chunk,
  // int64_t* cum_sum_num_bytes_per_chunk,
  // int64_t num_bytes_per_chunk,
  int64_t num_blocks_per_chunk,
  int64_t* tensor_bytes,
  int64_t chunk_stride
  // int64_t num_slice
) {



  tensors = tensors + blockIdx.y * num_slice;
  // out = out + blockIdx.y * num_bytes_per_chunk * num_chunks;
  // const int64_t slice_offset =
  const int64_t chunk_offset = blockIdx.x % num_blocks_per_chunk;
  const int64_t tensor_idx = chunk_offset_to_tensor_idx[chunk_offset];
  const uint4 zero = initialize();
  for (int64_t chunk_idx = blockIdx.x / num_blocks_per_chunk; chunk_idx < num_chunks; chunk_idx += chunk_stride) {
    // const int64_t chunk_begin = cum_sum_num_bytes_per_chunk[tensor_idx];
    // const int64_t theory_chunk_num_bytes = cum_sum_num_bytes_per_chunk[tensor_idx+1] - chunk_begin;
    const int64_t actual_num_bytes = minInt64(
      theory_chunk_num_bytes,
      maxInt64(tensor_bytes[tensor_idx] - chunk_idx * theory_chunk_num_bytes, 0)
    );
    if (actual_num_bytes == 0) {
      return;
    }
    const int64_t chunk_block_count = cum_sum_blocks_per_chunk[tensor_idx + 1] - cum_sum_blocks_per_chunk[tensor_idx];
    const int64_t num_threads = chunk_block_count * blockDim.x;
    const int64_t thread_idx = (chunk_offset - cum_sum_blocks_per_chunk[tensor_idx]) * blockDim.x + threadIdx.x;
    const int64_t dst_off = chunk_idx * num_bytes_per_chunk + chunk_begin;
    const int64_t src_off = chunk_idx * theory_chunk_num_bytes;
    char* dst_pointer = reinterpret_cast<char*>(out) + dst_off;
    const char* src_pointer = reinterpret_cast<char*>(tensors[tensor_idx]) + src_off;
    detail::copy_chunk_with_pad(
      dst_pointer,
      src_pointer,
      theory_chunk_num_bytes,
      actual_num_bytes,
      thread_idx,
      num_threads,
      zero
    );
  }
}
} // namespace detail

void assert_leading_dimension_matches(
  std::vector<at::Tensor> tensors,
  int64_t dim
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
        "unflatten_cat_with_pad() has invalid args: tensors should have same sizes in the first dim dimensions"
      );
    }
  }
}




// First, tensors can be viewed as a list of 3-d tensor:
//  For a given tensor, we can split dimensions as [:dim], [dim], [dim+1:].
//  These three parts can be treated as leading_numel, dim, tailing_numel, respectively.
//  For all tensors, we have num_tensors, leading_numel, dim, tailing_numel
// Second,
//  input_tensors = []
//  for i in range(num_tensors):
//    for j in range(leading_numel):
//      input_tensors[j*num_tensors + i] = tensors[i][j] // input_tensors[j][i] have shape [dim x tailing_numel]
// Third,
//  For each j in range(leading_numel), there are num_tensors tensors and num_tensors * dim * tailing_numel (slice_numel) elements in total
//  We can compute num_blocks_per_slice according to slice_numel.
//  blockIdx.x =>
//    blockIdx.y = blockIdx.x / num_blocks_per_slice
//    blockIdx.x = blockIdx.x % num_blocks_per_slice
//    chunk_idx = blockIdx.x / num_blocks_per_chunk
//    chunk_offset = blockIdx.x % num_blocks_per_chunk
//    tensor_idx = chunk_offset_to_tensor_idx[chunk_offset]
// TODO: Rename as pad_reshape_cat
void unflatten_cat_with_pad_cuda(
  std::vector<at::Tensor> tensors,
  int64_t dim,
  int64_t factor,
  at::Tensor out
) {
  const auto device = out.device();
  auto num_tensors = tensors.size();
  TORCH_CHECK(
    out.is_cuda() && out.is_non_overlapping_and_dense(),
    "unflatten_cat_with_pad_cuda() error: invalid out tensor"
  );
  assert_leading_dimension_matches(tensors, dim);
  auto leading_numel = c10::multiply_integers(tensors[0].sizes().slice(0, dim));
  // tensor_pointers has layout of (leading_numel, num_tensors) where each element is a pointer to a tensor of layout size_along_dim x tailing_numel
  std::vector<int64_t> tensor_pointers(num_tensors * leading_numel, 0);
  std::vector<int64_t> tensor_bytes_per_slice;
  std::vector<int64_t> tensor_idx_to_num_bytes_per_chunk;
  std::vector<int64_t> cum_sum_num_bytes_per_chunk{0};
  for (const auto i : c10::irange(num_tensors)) {
    at::Tensor tensor = tensors[i];
    TORCH_CHECK(
      tensor.is_cuda() && tensor.is_non_overlapping_and_dense(),
      "unflatten_cat_with_pad_cuda() error: invalid out tensor"
    );
    auto sizes = tensor.sizes();
    const int64_t size_along_dim = sizes[dim];
    int64_t tailing_numel = 1;
    if(sizes.size() > dim + 1) {
      tailing_numel = c10::multiply_integers(sizes.slice(dim+1, sizes.size()-dim-1));
    }
    const int64_t pad_size_along_dim = div_up(size_along_dim, factor) * factor;
    const int64_t num_bytes_per_chunk = pad_size_along_dim * tailing_numel * tensor.element_size() / factor;
    const int64_t base_pointer = reinterpret_cast<int64_t>(tensor.data_ptr());
    tensor_bytes_per_slice.push_back(tailing_numel * size_along_dim * tensor.element_size());
    for (const auto j : c10::irange(leading_numel)) {
      tensor_pointers[j*num_tensors + i] = base_pointer + j * tensor_bytes_per_slice.back();
    }
    tensor_idx_to_num_bytes_per_chunk.push_back(num_bytes_per_chunk);
    cum_sum_num_bytes_per_chunk.push_back(cum_sum_num_bytes_per_chunk[i] + num_bytes_per_chunk);
  }
  constexpr int64_t max_active_blocks = 32 * 132;
  constexpr int64_t sm_oversub = 2;
  std::vector<int64_t> chunk_offset_to_tensor_idx;
  std::vector<int64_t> cum_sum_blocks_per_chunk{0};
  for (const auto i : c10::irange(num_tensors)) {
    int64_t num_blocks_per_chunk = div_up(tensor_idx_to_num_bytes_per_chunk[i], BLOCK_SIZE * BYTES_PER_THREAD);
    chunk_offset_to_tensor_idx.insert(chunk_offset_to_tensor_idx.end(), num_blocks_per_chunk, i);
    cum_sum_blocks_per_chunk.push_back(cum_sum_blocks_per_chunk.back() + num_blocks_per_chunk);
  }
  const auto num_blocks_per_chunk = cum_sum_blocks_per_chunk.back();
  auto packed = pack(
    {tensor_pointers, chunk_offset_to_tensor_idx, cum_sum_num_bytes_per_chunk, cum_sum_num_bytes_per_chunk, tensor_bytes_per_slice}, device
  );
  int64_t chunks_per_block = 1;
  while (num_blocks_per_chunk * (factor / chunks_per_block) * leading_numel >
          max_active_blocks * sm_oversub &&
        chunks_per_block < factor) {
    ++chunks_per_block;
  }
  dim3 blocks(num_blocks_per_chunk * (factor / chunks_per_block), leading_numel, 1);
  dim3 threads(BLOCK_SIZE, 1, 1);
  unflatten_cat_with_pad_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<void**>(packed.second[0]),
      factor,
      out.data_ptr(),
      /*chunk_offset_to_tensor_idx=*/packed.second[1],
      /*cum_sum_blocks_per_chunk=*/packed.second[2],
      /*cum_sum_num_bytes_per_chunk=*/packed.second[3],
      cum_sum_num_bytes_per_chunk.back(),
      num_blocks_per_chunk,
      /*tensor_bytes_per_slice=*/packed.second[4],
      factor / chunks_per_block,
      leading_numel
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
