#include "gemm.hpp"
#include "../common.hpp"

#include <cublas_v2.h>
#include <mutex>
#include <cuda_fp16.h>

namespace trellis::gemm {

static cublasHandle_t get_handle() {
  static cublasHandle_t handle = nullptr;
  static std::once_flag once;
  std::call_once(once, [](){
    auto st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
      TRELLIS_THROW("cuBLAS create failed");
    }
  });
  return handle;
}

static void check(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    TRELLIS_THROW(std::string("cuBLAS error in ") + what);
  }
}

void f32_rowmajor(const float* d_A, const float* d_B, float* d_C,
                  int m, int n, int k,
                  float alpha, float beta) {
  // Row-major to column-major mapping: see analysis.
  // We compute C^T = B^T @ A^T with column-major cuBLAS, dims n x m, k.
  // lda, ldb, ldc are leading dimensions for column-major inputs.
  cublasHandle_t h = get_handle();
  cublasOperation_t opA = CUBLAS_OP_T; // A is row-major (m,k) -> treat as (k,m) column-major with transpose
  cublasOperation_t opB = CUBLAS_OP_T; // B is row-major (k,n) -> treat as (n,k) column-major with transpose

  int rows_Ct = n; // rows of C^T
  int cols_Ct = m; // cols of C^T
  int inner    = k; // shared dim

  const float* A_col = d_A;
  const float* B_col = d_B;
  float*       C_col = d_C;

  // In column-major, the leading dimension is the number of rows.
  int lda = m; // A^T has shape (k, m) but data from m,k row-major -> lda = m
  int ldb = k; // B^T has shape (n, k) but data from k,n row-major -> ldb = k
  int ldc = n; // C^T has shape (n, m)

  // Note: cublasSgemm computes: C = alpha*op(A)*op(B) + beta*C with sizes (rows_Ct x cols_Ct)
  // cublasSgemm arguments: (handle, opB, opA, rows_Ct, cols_Ct, inner, ...)
  // However, the conventional is cublasSgemm(handle, opN, opN, m, n, k, ... A, lda, B, ldb, C, ldc)
  // We'll follow the standard order: C(n x m) = B(n x k)*A(k x m)
  check(
    cublasSgemm(h,
                opB, opA,
                rows_Ct, cols_Ct, inner,
                &alpha,
                B_col, ldb,
                A_col, lda,
                &beta,
                C_col, ldc),
    "cublasSgemm");
}

void f16_rowmajor_accum_f32(const void* d_A_half, const void* d_B_half, float* d_C,
                            int m, int n, int k,
                            float alpha, float beta) {
  cublasHandle_t h = get_handle();
  // Map row-major to column-major via transpose, as above.
  cublasOperation_t opA = CUBLAS_OP_T;
  cublasOperation_t opB = CUBLAS_OP_T;
  int rows_Ct = n;
  int cols_Ct = m;
  int inner    = k;
  int lda = m; // A^T (k x m)
  int ldb = k; // B^T (n x k)
  int ldc = n; // C^T (n x m)
  check(
    cublasGemmEx(h,
                 opB, opA,
                 rows_Ct, cols_Ct, inner,
                 &alpha,
                 d_B_half, CUDA_R_16F, ldb,
                 d_A_half, CUDA_R_16F, lda,
                 &beta,
                 d_C, CUDA_R_32F, ldc,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
    "cublasGemmEx f16->f32");
}

void f16_rowmajor_out_f16_accum_f32(const void* d_A_half, const void* d_B_half, void* d_C_half,
                                    int m, int n, int k,
                                    float alpha, float beta) {
  cublasHandle_t h = get_handle();
  cublasOperation_t opA = CUBLAS_OP_T;
  cublasOperation_t opB = CUBLAS_OP_T;
  int rows_Ct = n;
  int cols_Ct = m;
  int inner    = k;
  int lda = m;
  int ldb = k;
  int ldc = n;
  check(
    cublasGemmEx(h,
                 opB, opA,
                 rows_Ct, cols_Ct, inner,
                 &alpha,
                 d_B_half, CUDA_R_16F, ldb,
                 d_A_half, CUDA_R_16F, lda,
                 &beta,
                 d_C_half, CUDA_R_16F, ldc,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
    "cublasGemmEx f16->f16");
}

} // namespace trellis::gemm
