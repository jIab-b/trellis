#include "device.hpp"
#include "../common.hpp"

#ifdef TRELLIS_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace trellis::device {

bool available() {
#ifdef TRELLIS_HAVE_CUDA
  int count = 0; auto st = cudaGetDeviceCount(&count); return (st == cudaSuccess) && count > 0;
#else
  return false;
#endif
}

void* malloc(size_t bytes) {
#ifdef TRELLIS_HAVE_CUDA
  void* p = nullptr; auto st = cudaMalloc(&p, bytes); if (st != cudaSuccess) TRELLIS_THROW("cudaMalloc failed"); return p;
#else
  (void)bytes; TRELLIS_THROW("CUDA not available: rebuild with TRELLIS_HAVE_CUDA");
#endif
}

void free(void* p) noexcept {
#ifdef TRELLIS_HAVE_CUDA
  if (p) cudaFree(p);
#else
  (void)p;
#endif
}

void memcpy_htod(void* dst_device, const void* src_host, size_t bytes) {
#ifdef TRELLIS_HAVE_CUDA
  auto st = cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice);
  if (st != cudaSuccess) TRELLIS_THROW("cudaMemcpy HtoD failed");
#else
  (void)dst_device; (void)src_host; (void)bytes; TRELLIS_THROW("CUDA not available");
#endif
}

void memcpy_dtoh(void* dst_host, const void* src_device, size_t bytes) {
#ifdef TRELLIS_HAVE_CUDA
  auto st = cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost);
  if (st != cudaSuccess) TRELLIS_THROW("cudaMemcpy DtoH failed");
#else
  (void)dst_host; (void)src_device; (void)bytes; TRELLIS_THROW("CUDA not available");
#endif
}

MemInfo meminfo() {
  MemInfo m{0,0};
#ifdef TRELLIS_HAVE_CUDA
  size_t free_b=0, total_b=0; auto st = cudaMemGetInfo(&free_b, &total_b);
  if (st != cudaSuccess) TRELLIS_THROW("cudaMemGetInfo failed");
  m.free_bytes = free_b; m.total_bytes = total_b;
#endif
  return m;
}

} // namespace trellis::device

