#pragma once

#include <cstddef>

namespace trellis {

struct DeviceAllocator {
  // placeholder; future: pools, CUDA allocations
  void* malloc(std::size_t /*bytes*/) { return nullptr; }
  void free(void* /*p*/) {}
};

} // namespace trellis
