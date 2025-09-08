#include "cast.hpp"
#include "../common.hpp"

#ifdef TRELLIS_HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace trellis::cast {

__global__ static void k_f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

__global__ static void k_f16_to_f32(const __half* __restrict__ src, float* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __half2float(src[i]);
}

void fp32_to_fp16(const float* d_src, void* d_dst_half, size_t elems) {
  int block = 256;
  int grid = (int)((elems + block - 1) / block);
  k_f32_to_f16<<<grid, block>>>(d_src, reinterpret_cast<__half*>(d_dst_half), elems);
  auto st = cudaGetLastError();
  if (st != cudaSuccess) TRELLIS_THROW("fp32_to_fp16 kernel launch failed");
}

void fp16_to_fp32(const void* d_src_half, float* d_dst, size_t elems) {
  int block = 256;
  int grid = (int)((elems + block - 1) / block);
  k_f16_to_f32<<<grid, block>>>(reinterpret_cast<const __half*>(d_src_half), d_dst, elems);
  auto st = cudaGetLastError();
  if (st != cudaSuccess) TRELLIS_THROW("fp16_to_fp32 kernel launch failed");
}

} // namespace trellis::cast

#else

namespace trellis::cast {
void fp32_to_fp16(const float*, void*, size_t) { TRELLIS_THROW("CUDA not available: fp32_to_fp16"); }
void fp16_to_fp32(const void*, float*, size_t) { TRELLIS_THROW("CUDA not available: fp16_to_fp32"); }
} // namespace trellis::cast

#endif

