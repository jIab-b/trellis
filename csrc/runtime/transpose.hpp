#pragma once

namespace trellis::transpose {

// Transpose an MxN matrix (row-major) on device: dst[N,M] = src[M,N]^
void f32(const float* d_src, int M, int N, float* d_dst);

}

