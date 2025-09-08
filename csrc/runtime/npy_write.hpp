#pragma once

#include <string>
#include <vector>

namespace trellis {

// Write a NumPy .npy file with dtype '<f4' (float32), C-order.
// shape: e.g., {1,1,16,16,16}
// Returns true on success; throws on I/O errors.
bool write_npy_f32(const std::string& path,
                   const float* data,
                   size_t elem_count,
                   const std::vector<long long>& shape);

} // namespace trellis
