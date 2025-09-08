#include "ops.hpp"
#include "../common.hpp"

#ifdef TRELLIS_HAVE_CUDA
#include <cuda_runtime.h>

namespace trellis::ops {

__global__ static void k_add_bias(float* y, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += b[i];
}

__global__ static void k_silu(float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = y[i];
    y[i] = x / (1.0f + __expf(-x));
  }
}

void add_bias_inplace(float* d_y, const float* d_b, int len) {
  int block = 256;
  int grid = (len + block - 1) / block;
  k_add_bias<<<grid, block>>>(d_y, d_b, len);
  auto st = cudaGetLastError();
  if (st != cudaSuccess) TRELLIS_THROW("cuda kernel launch failed: add_bias");
}

void silu_inplace(float* d_y, int len) {
  int block = 256;
  int grid = (len + block - 1) / block;
  k_silu<<<grid, block>>>(d_y, len);
  auto st = cudaGetLastError();
  if (st != cudaSuccess) TRELLIS_THROW("cuda kernel launch failed: silu");
}

__global__ static void k_add_bias_rows(float* y, const float* b, int T, int C) {
  int t = blockIdx.x;
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (t < T && i < C) {
    y[t * C + i] += b[i];
  }
}

void add_bias_rows_inplace(float* d_y, const float* d_b, int T, int C) {
  dim3 block(256);
  dim3 grid(T, (C + block.x - 1) / block.x);
  k_add_bias_rows<<<grid, block>>>(d_y, d_b, T, C);
  auto st = cudaGetLastError();
  if (st != cudaSuccess) TRELLIS_THROW("cuda kernel launch failed: add_bias_rows");
}

} // namespace trellis::ops

#else

namespace trellis::ops {
void add_bias_inplace(float*, const float*, int) { TRELLIS_THROW("CUDA not available: add_bias_inplace"); }
void silu_inplace(float*, int) { TRELLIS_THROW("CUDA not available: silu_inplace"); }
} // namespace trellis::ops

#endif
