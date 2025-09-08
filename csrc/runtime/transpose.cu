#include "transpose.hpp"
#include "../common.hpp"

#ifdef TRELLIS_HAVE_CUDA
#include <cuda_runtime.h>

namespace trellis::transpose {

__global__ static void k_t32(const float* __restrict__ src, int M, int N, float* __restrict__ dst) {
  // Tile transpose for coalescing (simple version)
  int x = blockIdx.x * blockDim.x + threadIdx.x; // column idx in src (0..N-1)
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row idx in src (0..M-1)
  if (x < N && y < M) {
    dst[x * M + y] = src[y * N + x];
  }
}

void f32(const float* d_src, int M, int N, float* d_dst) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
  k_t32<<<grid, block>>>(d_src, M, N, d_dst);
  auto st = cudaGetLastError();
  if (st != cudaSuccess) TRELLIS_THROW("transpose kernel launch failed");
}

} // namespace trellis::transpose

#else

namespace trellis::transpose { void f32(const float*, int, int, float*) { TRELLIS_THROW("CUDA not available: transpose"); } }

#endif

