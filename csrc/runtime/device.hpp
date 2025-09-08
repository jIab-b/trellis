#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace trellis::device {

struct MemInfo { size_t free_bytes; size_t total_bytes; };

bool available();
void* malloc(size_t bytes);
void free(void* p) noexcept;
void memcpy_htod(void* dst_device, const void* src_host, size_t bytes);
void memcpy_dtoh(void* dst_host, const void* src_device, size_t bytes);
MemInfo meminfo();

} // namespace trellis::device

