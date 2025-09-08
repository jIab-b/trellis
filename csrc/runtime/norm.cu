#include "norm.hpp"
#include "../common.hpp"

#ifdef TRELLIS_HAVE_CUDA
#include <cuda_runtime.h>

namespace trellis::norm {

__global__ static void k_layernorm_f32(const float* __restrict__ x, float* __restrict__ y,
                                       int T, int C, const float* __restrict__ gamma,
                                       const float* __restrict__ beta, float eps) {
  int t = blockIdx.x; // one block per row
  if (t >= T) return;
  extern __shared__ float smem[];
  float* ssum = smem;
  float* ssum2 = smem + 1;
  if (threadIdx.x == 0) { *ssum = 0.0f; *ssum2 = 0.0f; }
  __syncthreads();
  // parallel reduce mean and variance
  float local_sum = 0.0f, local_sum2 = 0.0f;
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    float v = x[t * C + i];
    local_sum += v;
    local_sum2 += v * v;
  }
  atomicAdd(ssum, local_sum);
  atomicAdd(ssum2, local_sum2);
  __syncthreads();
  float mean = *ssum / (float)C;
  float var = *ssum2 / (float)C - mean * mean;
  float inv_std = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    float v = x[t * C + i];
    float n = (v - mean) * inv_std;
    float g = gamma ? gamma[i] : 1.0f;
    float b = beta ? beta[i] : 0.0f;
    y[t * C + i] = n * g + b;
  }
}

void layernorm_f32(const float* d_x, float* d_y, int T, int C, const float* d_gamma, const float* d_beta, float eps) {
  int blocks = T;
  int threads = 256;
  size_t shmem = 2 * sizeof(float);
  k_layernorm_f32<<<blocks, threads, shmem>>>(d_x, d_y, T, C, d_gamma, d_beta, eps);
  auto st = cudaGetLastError();
  if (st != cudaSuccess) TRELLIS_THROW("layernorm kernel launch failed");
}

} // namespace trellis::norm

#else

namespace trellis::norm { void layernorm_f32(const float*, float*, int, int, const float*, const float*, float) { TRELLIS_THROW("CUDA not available: layernorm"); } }

#endif

