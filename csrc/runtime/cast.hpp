#pragma once

#include <cstddef>

namespace trellis::cast {

// Device casts between FP32 and FP16 for contiguous arrays.
// Pointers are device pointers. Throws if CUDA not available.
void fp32_to_fp16(const float* d_src, void* d_dst_half, size_t elems);
void fp16_to_fp32(const void* d_src_half, float* d_dst, size_t elems);

}

