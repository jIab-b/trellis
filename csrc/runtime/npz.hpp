#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace trellis {

// Minimal scaffold for reading a single array from an NPZ file.
// Implementation is a stub; to be replaced with a small zip + NPY parser.

struct NpyArray {
  std::vector<int64_t> shape;
  std::vector<uint8_t> data; // raw bytes
  std::string dtype;         // e.g., "<f4"
};

NpyArray load_from_npz(const std::string& npz_path, const std::string& member_name);

} // namespace trellis

