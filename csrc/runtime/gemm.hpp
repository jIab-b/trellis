#pragma once

#include <cstddef>

namespace trellis::gemm {

// Row-major GEMM: C[m,n] = alpha * A[m,k] @ B[k,n] + beta * C[m,n]
// All buffers are device pointers to float32.
// Requires CUDA + cuBLAS. Throws trellis::Error on failure.
void f32_rowmajor(const float* d_A, const float* d_B, float* d_C,
                  int m, int n, int k,
                  float alpha = 1.0f, float beta = 0.0f);

// Row-major GEMM for FP16 inputs with FP32 accumulation.
// A[m,k], B[k,n], C[m,n]. d_A/d_B point to device __half buffers.
// Accumulates in FP32 and writes to C as FP32.
void f16_rowmajor_accum_f32(const void* d_A_half, const void* d_B_half, float* d_C,
                            int m, int n, int k,
                            float alpha = 1.0f, float beta = 0.0f);

// Same as above but output C is FP16. Accumulator remains FP32.
void f16_rowmajor_out_f16_accum_f32(const void* d_A_half, const void* d_B_half, void* d_C_half,
                                    int m, int n, int k,
                                    float alpha = 1.0f, float beta = 0.0f);

} // namespace trellis::gemm
