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

template <typename T>
__device__ inline void streamLoad128(uint4& val, const T* addr) {
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
__device__ inline void streamStore128(T* addr, const uint4& val) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
#else
  unsigned long long int low, high;
  low = reinterpret_cast<const unsigned long long int*>(&val)[0];
  high = reinterpret_cast<const unsigned long long int*>(&val)[1];
  asm("st.global.cs.v2.u64 [%0], {%1, %2};" : : "l"(addr), "l"(low), "l"(high));
#endif
}

static __host__ __device__ inline int64_t divUp(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

static __host__ __device__ inline int64_t minInt64(int64_t a, int64_t b) {
  return a < b ? a : b;
}

static __host__ __device__ inline int64_t maxInt64(int64_t a, int64_t b) {
  return a < b ? b : a;
}

static __device__ inline bool isAligned(const void* ptr, size_t alignment) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  return addr % alignment == 0;
}

__device__ inline uint4 initialize() {
  uint4 zero;
  reinterpret_cast<uint64_t*>(&zero)[0] = 0;
  reinterpret_cast<uint64_t*>(&zero)[1] = 0;
  return zero;
}

static __global__ void fsdpAllGatherCopyOutKernel(
    void** paramPtrs,
    void* allGatherResPtr,
    int64_t* blockOffsetToParamIdx,
    int64_t* blockCumSums,
    int64_t* shardDimCumSums,
    int64_t numBytesPerRank,
    int64_t numBlocksPerRank,
    int64_t rankStride,
    int64_t worldSize) {
  const int64_t blockOffset = blockIdx.x % numBlocksPerRank;
  const int64_t paramIdx = blockOffsetToParamIdx[blockOffset];
  for (int64_t rank = blockIdx.x / numBlocksPerRank; rank < worldSize;
       rank += rankStride) {
    const int64_t shardBlockCount =
        blockCumSums[paramIdx + 1] - blockCumSums[paramIdx];
    const int64_t groupSize = shardBlockCount * blockDim.x;
    const int64_t localTid =
        (blockOffset - blockCumSums[paramIdx]) * blockDim.x + threadIdx.x;

    const int64_t shardBegin = shardDimCumSums[paramIdx];
    const int64_t shardEnd = shardDimCumSums[paramIdx + 1];
    const int64_t shardLen = shardEnd - shardBegin;
    const int64_t srcOff = rank * numBytesPerRank + shardBegin;
    const int64_t dstOff = rank * shardLen;

    const char* srcPtr = reinterpret_cast<char*>(allGatherResPtr) + srcOff;
    char* dstPtr = reinterpret_cast<char*>(paramPtrs[paramIdx]) + dstOff;

    const int64_t alignOff =
        divUp(dstOff, BYTES_PER_THREAD) * BYTES_PER_THREAD - dstOff;
    const int64_t begin = alignOff + localTid * BYTES_PER_THREAD;
    const int64_t end =
        alignOff + (shardLen - alignOff) / BYTES_PER_THREAD * BYTES_PER_THREAD;
    const int64_t stride = groupSize * BYTES_PER_THREAD;

    for (size_t i = begin; i < end; i += stride) {
      uint4 val;
      if (isAligned(srcPtr + i, BYTES_PER_THREAD)) {
        streamLoad128(val, srcPtr + i);
      } else {
        for (size_t j = 0; j < BYTES_PER_THREAD; ++j) {
          reinterpret_cast<char*>(&val)[j] = srcPtr[i + j];
        }
      }
      streamStore128(&dstPtr[i], val);
    }
    if (localTid < alignOff && localTid < shardLen) {
      dstPtr[localTid] = srcPtr[localTid];
    }
    if (end + localTid < shardLen) {
      dstPtr[end + localTid] = srcPtr[end + localTid];
    }
  }
}

static int64_t geometricMean(const std::vector<int64_t>& numbers) {
  TORCH_CHECK(numbers.size() > 0);
  double logSum = 0.0;
  for (double num : numbers) {
    TORCH_CHECK(num > 0);
    logSum += log(num);
  }
  double avgLog = logSum / numbers.size();
  return exp(avgLog);
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

void fsdpAllGatherCopyOut(
    std::vector<at::Tensor> params,
    at::Tensor allGatherRes,
    int64_t worldSize) {
  const auto device = allGatherRes.device();
  const auto totalSize = allGatherRes.numel() * allGatherRes.element_size();

  TORCH_CHECK(allGatherRes.is_cuda());
  TORCH_CHECK(allGatherRes.is_non_overlapping_and_dense());

  std::vector<int64_t> paramPtrs;
  std::vector<int64_t> shardDims; // In bytes
  std::vector<int64_t> dimCumSums{0}; // In bytes
  for (size_t i = 0; i < params.size(); ++i) {
    const auto& param = params[i];
    TORCH_CHECK(param.is_non_overlapping_and_dense());
    TORCH_CHECK(param.device() == device);
    TORCH_CHECK(param.numel() > 0);
    // All params are expected to be aligned at worldSize.
    TORCH_CHECK(param.numel() % worldSize == 0);
    const auto shardDim = param.numel() * param.element_size() / worldSize;
    paramPtrs.push_back(reinterpret_cast<int64_t>(param.data_ptr()));
    shardDims.push_back(shardDim);
    dimCumSums.push_back(dimCumSums[i] + shardDim);
  }

  TORCH_CHECK(
      dimCumSums.back() * worldSize == totalSize,
      "The total byte size must be identical between params and allGatherRes");

  // To balance the throughput larger shards and waste on smaller shards, we
  // use the geometric mean of the shard dims to determine the block size.
  int64_t meanShardDim = geometricMean(shardDims);
  int64_t blockSize = divUp(meanShardDim, BYTES_PER_THREAD);
  blockSize = divUp(blockSize, WARP_SIZE) * WARP_SIZE;
  blockSize = std::min(std::max(blockSize, MIN_NUM_THREADS), MAX_NUM_THREADS);

  // TODO: this is only for A100
  constexpr int64_t maxActiveBlocks = 32 * 108;
  constexpr double smOverSubFactor = 1.75;

  // Roughly estimate the amount of blocks needed for each rank, and calculate
  // an iter factor to regularize SM over-subscription.
  int64_t iterFactor = 1;
  while (divUp(totalSize, blockSize * BYTES_PER_THREAD * iterFactor) >
         (maxActiveBlocks * smOverSubFactor)) {
    iterFactor += 1;
  }

  std::vector<int64_t> blockOffsetToParamIdx;
  std::vector<int64_t> blockCumSums{0};
  for (int64_t paramIdx = 0; paramIdx < static_cast<int64_t>(params.size());
       ++paramIdx) {
    int64_t numBlocks =
        divUp(shardDims[paramIdx], blockSize * BYTES_PER_THREAD * iterFactor);
    blockOffsetToParamIdx.insert(
        blockOffsetToParamIdx.end(), numBlocks, paramIdx);
    blockCumSums.push_back(blockCumSums.back() + numBlocks);
  }
  const auto numBlocks = blockCumSums.back();

  auto packed = pack(
      {paramPtrs, blockOffsetToParamIdx, blockCumSums, dimCumSums}, device);

  int64_t ranksPerBlock = 1;
  while (numBlocks * (worldSize / ranksPerBlock) >
             maxActiveBlocks * smOverSubFactor &&
         ranksPerBlock < worldSize) {
    ++ranksPerBlock;
  }

  dim3 blocks(numBlocks * (worldSize / ranksPerBlock), 1, 1);
  dim3 threads(blockSize, 1, 1);

  LOG(INFO) << "meanShardDim: " << meanShardDim
            << ", iterFactor: " << iterFactor
            << ", ranksPerBlock: " << ranksPerBlock << ", blocks: " << blocks.x
            << ", threads: " << threads.x;

  fsdpAllGatherCopyOutKernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<void**>(packed.second[0]),
      allGatherRes.data_ptr(),
      /*blockOffsetToParamIdx=*/packed.second[1],
      /*blockCumSums=*/packed.second[2],
      /*shardDimCumSums=*/packed.second[3],
      dimCumSums.back(),
      blockCumSums.back(),
      worldSize / ranksPerBlock,
      worldSize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// static __global__ void unflatten_cat_with_pad_dim0_cuda(
//   void **tensors,
//   int64_t factor,
//   void* out,
//   int64_t* blockOffsetToTensorIdx,
//   int64_t* cumSumBlocksPerShard,
//   int64_t* cumSumNumBytesPerShard,
//   int64_t numBytesPerRank,
//   int64_t numBlocksPerRank,
//   int64_t* tensorBytes,
//   int64_t rankStride
// ) {
//   const int64_t blockOffset = blockIdx.x % numBlocksPerRank;
//   const int64_t tensorIdx = blockOffsetToTensorIdx[blockOffset];
//   for (int64_t rank = blockIdx.x / numBlocksPerRank; rank < factor; rank += rankStride) {
//     const int64_t shardBegin = cumSumNumBytesPerShard[tensorIdx];
//     const int64_t shardEnd = cumSumNumBytesPerShard[tensorIdx+1];
//     const int64_t theoryShardNumBytes = shardEnd - shardBegin;
//     const int64_t actualNumBytes = minInt64(theoryShardNumBytes, maxInt64(tensorBytes[tensorIdx] - rank * theoryShardNumBytes, 0));
//     if (actualNumBytes == 0) {
//       return;
//     }
//     const int64_t shardBlockCount = cumSumBlocksPerShard[tensorIdx + 1] - cumSumBlocksPerShard[tensorIdx];
//     const int64_t groupSize = shardBlockCount * blockDim.x;
//     const int64_t localTid = (blockOffset - cumSumBlocksPerShard[tensorIdx]) * blockDim.x + threadIdx.x;
//     const int64_t dstOff = rank * numBytesPerRank + shardBegin;
//     const int64_t srcOff = rank * theoryShardNumBytes;
//     char* dstPtr = reinterpret_cast<char*>(out) + dstOff;
//     const char* srcPtr = reinterpret_cast<char*>(tensors[tensorIdx]) + srcOff;
//     const int64_t alignOff =
//       divUp(dstOff, BYTES_PER_THREAD) * BYTES_PER_THREAD - dstOff;
//     const int64_t begin = alignOff + localTid * BYTES_PER_THREAD;
//     const int64_t end = alignOff + (actualNumBytes - alignOff) / BYTES_PER_THREAD * BYTES_PER_THREAD;
//     const int64_t stride = groupSize * BYTES_PER_THREAD;
//     const uint4 zero = initialize();
//     for (size_t i = begin; i < end; i += stride) {
//       uint4 val = zero;
//       if(isAligned(srcPtr + i, BYTES_PER_THREAD)) {
//         streamLoad128(val, srcPtr + i);
//       } else {
//         for (size_t j = 0; j < BYTES_PER_THREAD; ++j) {
//           reinterpret_cast<char*>(&val)[j] = srcPtr[i + j];
//         }
//       }
//       streamStore128(&dstPtr[i], val);
//     }
//     if(localTid < alignOff && localTid < theoryShardNumBytes) {
//       char val = (char) 0;
//       if (localTid < actualNumBytes) {
//         val = srcPtr[localTid];
//       }
//       dstPtr[localTid] = val;
//     }
//     if(end + localTid < theoryShardNumBytes) {
//       char val = (char) 0;
//       if (end + localTid < actualNumBytes) {
//         val = srcPtr[end + localTid];
//       }
//       dstPtr[end + localTid] = val;
//     }
//   }
// }

// // Pad and cat along 0-th dimension. We do not assume that param.numel() % factor == 0.
// // TODO: Add more doc.
// void unflatten_cat_with_pad_dim0(
//   std::vector<at::Tensor> tensors,
//   int64_t factor,
//   at::Tensor out
// ) {
//   const auto device = out.device();
//   TORCH_CHECK(out.is_cuda());
//   TORCH_CHECK(out.is_non_overlapping_and_dense());
//   std::vector<int64_t> tensorPtrs;
//   std::vector<int64_t> tensorBytes;
//   std::vector<int64_t> tensorIdxToNumBytesPerShard;
//   std::vector<int64_t> cumSumNumBytesPerShard{0};
//   // TODO: We may only do boundary check once.
//   // There are three types of blocks: within boundary, outside boundary, or on the boundary. We do not need to check boundary many times.
//   for (size_t i = 0; i < tensors.size(); ++i) {
//     const auto& tensor = tensors[i];
//     TORCH_CHECK(tensor.is_non_overlapping_and_dense());
//     TORCH_CHECK(tensor.device() == device);
//     TORCH_CHECK(tensor.numel() > 0);
//     const auto sizes = tensor.sizes();
//     const int64_t sizeRemainingDims = tensor.numel() / sizes[0];
//     const int64_t padSizeAlongDim = divUp(sizes[0], factor) * factor;
//     const int64_t numBytesPerShard = padSizeAlongDim * sizeRemainingDims * tensor.element_size() / factor;
//     tensorPtrs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
//     tensorBytes.push_back(tensor.numel() * tensor.element_size());
//     tensorIdxToNumBytesPerShard.push_back(numBytesPerShard);
//     cumSumNumBytesPerShard.push_back(cumSumNumBytesPerShard[i] + numBytesPerShard);
//   }
//   constexpr int64_t maxActiveBlocks = 32 * 132;
//   constexpr int64_t smOverSubFactor = 1.75;
//   std::vector<int64_t> blockOffsetToTensorIdx;
//   std::vector<int64_t> cumSumBlocksPerShard{0};
//   for (int64_t tensorIdx = 0; tensorIdx < static_cast<int64_t>(tensors.size()); ++tensorIdx) {
//     int64_t numBlocksPerShard = divUp(tensorIdxToNumBytesPerShard[tensorIdx], BLOCK_SIZE * BYTES_PER_THREAD);
//     blockOffsetToTensorIdx.insert(blockOffsetToTensorIdx.end(), numBlocksPerShard, tensorIdx);
//     cumSumBlocksPerShard.push_back(cumSumBlocksPerShard.back() + numBlocksPerShard);
//   }
//   const auto numBlocksPerRank = cumSumBlocksPerShard.back();
//   auto packed = pack(
//     {tensorPtrs, blockOffsetToTensorIdx, cumSumBlocksPerShard, cumSumNumBytesPerShard, tensorBytes}, device
//   );
//   int64_t ranksPerBlock = 1;
//   while (numBlocksPerRank * (factor / ranksPerBlock) >
//           maxActiveBlocks * smOverSubFactor &&
//         ranksPerBlock < factor) {
//     ++ranksPerBlock;
//   }
//   dim3 blocks(numBlocksPerRank * (factor / ranksPerBlock), 1, 1);
//   dim3 threads(BLOCK_SIZE, 1, 1);
//   unflatten_cat_with_pad_dim0_kernel<<<
//     blocks,
//     threads,
//     0,
//     at::cuda::getCurrentCUDAStream()>>>(
//       reinterpret_cast<void**>(packed.second[0]),
//       factor,
//       out.data_ptr(),
//       /*blockOffsetToTensorIdx=*/packed.second[1],
//       /*cumSumBlocksPerShard=*/packed.second[2],
//       /*cumSumNumBytesPerShard=*/packed.second[3],
//       cumSumNumBytesPerShard.back(),
//       numBlocksPerRank,
//       /*tensorBytes=*/packed.second[4],
//       factor / ranksPerBlock
//   );
// }


// // Pad and cat along 0-th dimension. We do not assume that param.numel() % factor == 0.
// // TODO: Add more doc.
// void unflatten_cat_with_pad_dim0_temp(
//   std::vector<at::Tensor> tensors,
//   int64_t factor,
//   at::Tensor out
// ) {
//   // const auto device = out.device();
//   // TORCH_CHECK(out.is_cuda());
//   // TORCH_CHECK(out.is_non_overlapping_and_dense());
//   // std::vector<int64_t> tensor_pointers;
//   // std::vector<int64_t> tensor_bytes;
//   // std::vector<int64_t> tensor_idx_to_num_bytes_per_chunk;
//   // std::vector<int64_t> cum_sum_num_bytes_per_chunk{0};
//   // TODO: We may only do boundary check once.
//   // There are three types of blocks: within boundary, outside boundary, or on the boundary. We do not need to check boundary many times.
//   // for (size_t i = 0; i < tensors.size(); ++i) {
//     // const auto& tensor = tensors[i];
//     // TORCH_CHECK(tensor.is_non_overlapping_and_dense());
//     // TORCH_CHECK(tensor.device() == device);
//     // TORCH_CHECK(tensor.numel() > 0);
//     // const auto sizes = tensor.sizes();
//     // const int64_t tailing_numel = tensor.numel() / sizes[0];
//     // const int64_t pad_size_along_dim = divUp(sizes[0], factor) * factor;
//     // const int64_t num_bytes_per_chunk = pad_size_along_dim * tailing_numel * tensor.element_size() / factor;
//     // tensor_pointers.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
//     // tensor_bytes.push_back(tensor.numel() * tensor.element_size());
//     // tensor_idx_to_num_bytes_per_chunk.push_back(num_bytes_per_chunk);
//     // cum_sum_num_bytes_per_chunk.push_back(cum_sum_num_bytes_per_chunk[i] + num_bytes_per_chunk);
//   // }
//   // constexpr int64_t max_active_blocks = 32 * 132;
//   // constexpr int64_t smOverSubFactor = 1.75;
//   // std::vector<int64_t> chunk_offset_to_tensor_idx;
//   // std::vector<int64_t> cum_sum_blocks_per_chunk{0};
//   // for (int64_t i = 0; i < static_cast<int64_t>(tensors.size()); ++i) {
//   //   int64_t num_blocks_per_chunk = divUp(tensor_idx_to_num_bytes_per_chunk[i], BLOCK_SIZE * BYTES_PER_THREAD);
//   //   chunk_offset_to_tensor_idx.insert(chunk_offset_to_tensor_idx.end(), num_blocks_per_chunk, i);
//   //   cum_sum_blocks_per_chunk.push_back(cum_sum_blocks_per_chunk.back() + num_blocks_per_chunk);
//   // }
//   // const auto num_blocks_per_chunk = cum_sum_blocks_per_chunk.back();
//   // auto packed = pack(
//   //   {tensor_pointers, chunk_offset_to_tensor_idx, cum_sum_blocks_per_chunk, cum_sum_num_bytes_per_chunk, tensorBytes}, device
//   // );
//   // int64_t chunks_per_block = 1;
//   // while (num_blocks_per_chunk * (factor / chunks_per_block) >
//   //         max_active_blocks * smOverSubFactor &&
//   //       chunks_per_block < factor) {
//   //   ++chunks_per_block;
//   // }
//   // dim3 blocks(num_blocks_per_chunk * (factor / chunks_per_block), 1, 1);
//   // dim3 threads(BLOCK_SIZE, 1, 1);
//   unflatten_cat_with_pad_dim0_kernel<<<
//     blocks,
//     threads,
//     0,
//     at::cuda::getCurrentCUDAStream()>>>(
//       reinterpret_cast<void**>(packed.second[0]),
//       factor,
//       out.data_ptr(),
//       /*chunk_offset_to_tensor_idx=*/packed.second[1],
//       /*cum_sum_blocks_per_chunk=*/packed.second[2],
//       /*cum_sum_num_bytes_per_chunk=*/packed.second[3],
//       cum_sum_num_bytes_per_chunk.back(),
//       num_blocks_per_chunk,
//       /*tensorBytes=*/packed.second[4],
//       factor / chunks_per_block
//   );
// }

static __global__ void unflatten_cat_with_pad_kernel(
  void **tensors,
  int64_t factor,
  void* out,
  int64_t* chunk_offset_to_tensor_idx,
  int64_t* cum_sum_blocks_per_chunk,
  int64_t* cum_sum_num_bytes_per_chunk,
  int64_t num_bytes_per_chunk,
  int64_t num_blocks_per_chunk,
  int64_t* tensor_bytes,
  int64_t chunk_stride,
  int64_t num_slice
) {
  tensors = tensors + blockIdx.y * num_slice;
  out = out + blockIdx.y * num_bytes_per_chunk * factor;
  const int64_t chunk_offset = blockIdx.x % num_blocks_per_chunk;
  const int64_t tensor_idx = chunk_offset_to_tensor_idx[chunk_offset];
  for (int64_t chunk_idx = blockIdx.x / num_blocks_per_chunk; chunk_idx < factor; chunk_idx += chunk_stride) {
    const int64_t chunk_begin = cum_sum_num_bytes_per_chunk[tensor_idx];
    const int64_t chunk_end = cum_sum_num_bytes_per_chunk[tensor_idx+1];
    const int64_t theory_chunk_num_bytes = chunk_end - chunk_begin;
    const int64_t actual_num_bytes = minInt64(
      theory_chunk_num_bytes,
      maxInt64(tensor_bytes[tensor_idx] - chunk_idx * theory_chunk_num_bytes, 0)
    );
    if (actual_num_bytes == 0) {
      return;
    }
    const int64_t chunk_block_count = cum_sum_blocks_per_chunk[tensor_idx + 1] - cum_sum_blocks_per_chunk[tensor_idx];
    const int64_t group_size = chunk_block_count * blockDim.x;
    const int64_t local_tid = (chunk_offset - cum_sum_blocks_per_chunk[tensor_idx]) * blockDim.x + threadIdx.x;
    const int64_t dst_off = chunk_idx * num_bytes_per_chunk + chunk_begin;
    const int64_t src_off = chunk_idx * theory_chunk_num_bytes;
    char* dst_pointer = reinterpret_cast<char*>(out) + dst_off;
    const char* src_pointer = reinterpret_cast<char*>(tensors[tensor_idx]) + src_off;
    const int64_t align_offset =
      divUp(dst_off, BYTES_PER_THREAD) * BYTES_PER_THREAD - dst_off;
    const int64_t begin = align_offset + local_tid * BYTES_PER_THREAD;
    const int64_t end = align_offset + (actual_num_bytes - align_offset) / BYTES_PER_THREAD * BYTES_PER_THREAD;
    const int64_t stride = group_size * BYTES_PER_THREAD;
    const uint4 zero = initialize();
    for (size_t i = begin; i < end; i += stride) {
      uint4 val = zero;
      if(isAligned(src_pointer + i, BYTES_PER_THREAD)) {
        streamLoad128(val, src_pointer + i);
      } else {
        for (size_t j = 0; j < BYTES_PER_THREAD; ++j) {
          reinterpret_cast<char*>(&val)[j] = src_pointer[i + j];
        }
      }
      streamStore128(&dst_pointer[i], val);
    }
    if(local_tid < align_offset && local_tid < theory_chunk_num_bytes) {
      char val = (char) 0;
      if (local_tid < actual_num_bytes) {
        val = src_pointer[local_tid];
      }
      dst_pointer[local_tid] = val;
    }
    if(end + local_tid < theory_chunk_num_bytes) {
      char val = (char) 0;
      if (end + local_tid < actual_num_bytes) {
        val = src_pointer[end + local_tid];
      }
      dst_pointer[end + local_tid] = val;
    }
  }
}

// TODO: Rename as pad_reshape_cat
void unflatten_cat_with_pad_cuda(
  std::vector<at::Tensor> tensors,
  int64_t dim,
  int64_t factor,
  at::Tensor out
) {
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
  const auto device = out.device();
  auto num_tensors = tensors.size();
  TORCH_CHECK(out.is_cuda());
  TORCH_CHECK(out.is_non_overlapping_and_dense());
  TORCH_CHECK(num_tensors > 1, "unflatten_cat_with_pad() has invalid args: should have at least 2 input tensors");
  std::vector<c10::SymInt> leading_dim_sizes;
  for (const auto i : c10::irange(dim)) {
    leading_dim_sizes.push_back(tensors[0].size(i));
  }
  auto leading_numel = c10::multiply_integers(tensors[0].sizes().slice(0, dim));
  std::vector<int64_t> tensor_pointers(num_tensors * leading_numel, 0);
  std::vector<int64_t> tensor_bytes;
  std::vector<int64_t> tensor_idx_to_num_bytes_per_chunk;
  std::vector<int64_t> cum_sum_num_bytes_per_chunk{0};
  for (const auto i : c10::irange(num_tensors)) {
    for(const auto j : c10::irange(dim)) {
      TORCH_CHECK(
        tensors[0].size(j) == leading_dim_sizes[j],
        "unflatten_cat_with_pad() has invalid args: tensors should have same sizes in the first dim dimensions"
      );
    }
    at::Tensor tensor;
    if (dim > 0) {
      tensor = tensors[i].flatten(0, dim-1);
    } else {
      tensor = tensors[i];
    }
    TORCH_CHECK(tensor.is_non_overlapping_and_dense(), "unflatten_cat_with_pad() error: tensor should be non overlapping and dense");
    TORCH_CHECK(tensor.device() == device, "unflatten_cat_with_pad() error: tensor and out should be on the same device");
    TORCH_CHECK(tensor.numel() > 0, "unflatten_cat_with_pad() error: tensor should have at least 1 element");
    const auto sizes = tensor.sizes();
    const int64_t tailing_numel = tensor.numel() / (sizes[0] * sizes[1]);
    const int64_t pad_size_along_dim = divUp(sizes[1], factor) * factor;
    const int64_t num_bytes_per_chunk = pad_size_along_dim * tailing_numel * tensor.element_size() / factor;
    const int64_t base_pointer = reinterpret_cast<int64_t>(tensor.data_ptr());
    tensor_bytes.push_back(tailing_numel * (int64_t) sizes[1] * (int64_t) tensor.element_size()); // TODO: This line leads to floating point exception.
    for (const auto j : c10::irange(leading_numel)) {
      tensor_pointers[j*num_tensors + i] = base_pointer + j * tensor_bytes.back(); // TODO: Double check
    }
    tensor_idx_to_num_bytes_per_chunk.push_back(num_bytes_per_chunk);
    cum_sum_num_bytes_per_chunk.push_back(cum_sum_num_bytes_per_chunk[i] + num_bytes_per_chunk);
  }
  constexpr int64_t max_active_blocks = 32 * 132;
  constexpr int64_t sm_oversub = 2;
  std::vector<int64_t> chunk_offset_to_tensor_idx;
  std::vector<int64_t> cum_sum_blocks_per_chunk{0};
  for (const auto i : c10::irange(num_tensors)) {
    int64_t num_blocks_per_chunk = divUp(tensor_idx_to_num_bytes_per_chunk[i], BLOCK_SIZE * BYTES_PER_THREAD);
    chunk_offset_to_tensor_idx.insert(chunk_offset_to_tensor_idx.end(), num_blocks_per_chunk, i);
    cum_sum_blocks_per_chunk.push_back(cum_sum_blocks_per_chunk.back() + num_blocks_per_chunk);
  }
  const auto num_blocks_per_chunk = cum_sum_blocks_per_chunk.back();
  auto packed = pack(
    {tensor_pointers, chunk_offset_to_tensor_idx, cum_sum_num_bytes_per_chunk, cum_sum_num_bytes_per_chunk, tensor_bytes}, device
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
      /*tensorBytes=*/packed.second[4],
      factor / chunks_per_block,
      leading_numel
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
