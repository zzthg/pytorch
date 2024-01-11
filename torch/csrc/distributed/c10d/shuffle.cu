#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

constexpr int64_t BYTES_PER_THREAD = 16;
constexpr int64_t MAX_NUM_THREADS = 1024;
constexpr int64_t MIN_NUM_THREADS = 128;
constexpr int64_t WARP_SIZE = 32;
constexpr int64_t BLOCK_SIZE = 32;

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

static __global__ void padCatDim0Kernel(
  void **tensors,
  int64_t factor,
  void* out,
  int64_t* blockOffsetToTensorIdx,
  int64_t* cumSumBlocksPerShard,
  int64_t* cumSumNumBytesPerShard,
  int64_t numBytesPerRank,
  int64_t numBlocksPerRank,
  int64_t* tensorBytes,
  int64_t rankStride
) {
  const int64_t blockOffset = blockIdx.x % numBlocksPerRank;
  const int64_t tensorIdx = blockOffsetToTensorIdx[blockOffset];
  for (int64_t rank = blockIdx.x / numBlocksPerRank; rank < factor; rank += rankStride) {
    const int64_t shardBlockCount = cumSumBlocksPerShard[tensorIdx + 1] - cumSumBlocksPerShard[tensorIdx];
    const int64_t groupSize = shardBlockCount * blockDim.x;
    const int64_t localTid = (blockOffset - cumSumBlocksPerShard[tensorIdx]) * blockDim.x + threadIdx.x;
    const int64_t shardBegin = cumSumNumBytesPerShard[tensorIdx];
    const int64_t shardEnd = cumSumNumBytesPerShard[tensorIdx+1];
    const int64_t theoryShardNumBytes = shardEnd - shardBegin;
    const int64_t actualNumBytes = minInt64(theoryShardNumBytes, maxInt64(tensorBytes[tensorIdx] - rank * theoryShardNumBytes, 0));
    const int64_t dstOff = rank * numBytesPerRank + shardBegin;
    const int64_t srcOff = rank * theoryShardNumBytes;
    char* dstPtr = reinterpret_cast<char*>(out) + dstOff;
    const char* srcPtr = reinterpret_cast<char*>(tensors[tensorIdx]) + srcOff;
    const int64_t alignOff =
      divUp(dstOff, BYTES_PER_THREAD) * BYTES_PER_THREAD - dstOff;
    const int64_t begin = alignOff + localTid * BYTES_PER_THREAD;
    const int64_t end = alignOff + (actualNumBytes - alignOff) / BYTES_PER_THREAD * BYTES_PER_THREAD;
    const int64_t stride = groupSize * BYTES_PER_THREAD;
    const uint4 zero = initialize();
    for (size_t i = begin; i < end; i += stride) {
      uint4 val = zero;
      if(isAligned(srcPtr + i, BYTES_PER_THREAD)) {
        streamLoad128(val, srcPtr + i);
      } else {
        for (size_t j = 0; j < BYTES_PER_THREAD; ++j) {
          reinterpret_cast<char*>(&val)[j] = srcPtr[i + j];
        }
      }
      streamStore128(&dstPtr[i], val);
    }
    if(localTid < alignOff && localTid < theoryShardNumBytes) {
      char val = (char) 0;
      if (localTid < actualNumBytes) {
        val = srcPtr[localTid];
      }
      dstPtr[localTid] = val;
    }
    if(end + localTid < theoryShardNumBytes) {
      char val = (char) 0;
      if (end + localTid < actualNumBytes) {
        val = srcPtr[end + localTid];
      }
      dstPtr[end + localTid] = val;
    }
  }
}

// Pad and cat along 0-th dimension. We do not assume that param.numel() % factor == 0.
// TODO: Add more doc.
void padCatDim0(
  std::vector<at::Tensor> tensors,
  int64_t factor,
  at::Tensor out
) {
  const auto device = out.device();
  const auto totalSize = out.numel() * out.element_size();
  TORCH_CHECK(out.is_cuda());
  TORCH_CHECK(out.is_non_overlapping_and_dense());
  std::vector<int64_t> tensorPtrs;
  std::vector<int64_t> tensorBytes;
  std::vector<int64_t> tensorIdxToNumBytesPerShard;
  std::vector<int64_t> cumSumNumBytesPerShard{0};
  // TODO: We may only do boundary check once.
  // There are three types of blocks: within boundary, outside boundary, or on the boundary. We do not need to check boundary many times.
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    TORCH_CHECK(tensor.is_non_overlapping_and_dense());
    TORCH_CHECK(tensor.device() == device);
    TORCH_CHECK(tensor.numel() > 0);
    const auto sizes = tensor.sizes();
    const int64_t sizeRemainingDims = tensor.numel() / sizes[0];
    const int64_t padSizeAlongDim = divUp(sizes[0], factor) * factor;
    const int64_t numBytesPerShard = padSizeAlongDim * sizeRemainingDims * tensor.element_size() / factor;
    tensorPtrs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
    tensorBytes.push_back(tensor.numel() * tensor.element_size());
    tensorIdxToNumBytesPerShard.push_back(numBytesPerShard);
    cumSumNumBytesPerShard.push_back(cumSumNumBytesPerShard[i] + numBytesPerShard);
  }
  constexpr int64_t maxActiveBlocks = 32 * 132;
  constexpr int64_t smOverSubFactor = 1.75;
  std::vector<int64_t> blockOffsetToTensorIdx;
  std::vector<int64_t> cumSumBlocksPerShard{0};
  for (int64_t tensorIdx = 0; tensorIdx < static_cast<int64_t>(tensors.size()); ++tensorIdx) {
    int64_t numBlocksPerShard = divUp(tensorIdxToNumBytesPerShard[tensorIdx], BLOCK_SIZE * BYTES_PER_THREAD);
    blockOffsetToTensorIdx.insert(blockOffsetToTensorIdx.end(), numBlocksPerShard, tensorIdx);
    cumSumBlocksPerShard.push_back(cumSumBlocksPerShard.back() + numBlocksPerShard);
  }
  const auto numBlocksPerRank = cumSumBlocksPerShard.back();
  auto packed = pack(
    {tensorPtrs, blockOffsetToTensorIdx, cumSumBlocksPerShard, cumSumNumBytesPerShard, tensorBytes}, device
  );
  int64_t ranksPerBlock = 1;
  while (numBlocksPerRank * (factor / ranksPerBlock) >
          maxActiveBlocks * smOverSubFactor &&
        ranksPerBlock < factor) {
    ++ranksPerBlock;
  }
  dim3 blocks(numBlocksPerRank * (factor / ranksPerBlock), 1, 1);
  dim3 threads(BLOCK_SIZE, 1, 1);
  padCatDim0Kernel<<<
    blocks,
    threads,
    0,
    at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<void**>(packed.second[0]),
      factor,
      out.data_ptr(),
      /*blockOffsetToTensorIdx=*/packed.second[1],
      /*cumSumBlocksPerShard=*/packed.second[2],
      /*cumSumNumBytesPerShard=*/packed.second[3],
      cumSumNumBytesPerShard.back(),
      numBlocksPerRank,
      /*tensorBytes=*/packed.second[4],
      factor / ranksPerBlock
  );
}

// TODO
void padCat(
  std::vector<at::Tensor> tensors,
  int64_t dim,
  int64_t factor,
  at::Tensor out
) {
  TORCH_CHECK(dim == 0);
  padCatDim0(tensors, factor, out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
