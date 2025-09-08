#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <fstream>
#include <memory>

namespace trellis {

struct SafeTensorInfo {
  std::string name;
  std::string dtype; // e.g., "F16", "F32"
  std::vector<int64_t> shape;
  uint64_t offset_begin{0};
  uint64_t offset_end{0};
};

struct SafeTensorHeader {
  // Raw JSON header as a string, for now (scaffold)
  std::string raw_json;
  // Optional parsed view (to be implemented later)
  std::vector<SafeTensorInfo> tensors;
};

class SafeTensorsFile {
 public:
  explicit SafeTensorsFile(const std::string& path);
  ~SafeTensorsFile();
  const SafeTensorHeader& header() const { return header_; }
  const std::string& path() const { return path_; }
  // Return pointer range in file for a tensor slice. For now, only exposes offsets.
  const SafeTensorInfo* find(const std::string& name) const;
  // Read raw bytes for a named tensor into out buffer. Returns false if not found.
  bool read_raw(const std::string& name, std::vector<uint8_t>& out) const;
  // Read as float32 (converts from F16 if needed) into out_f32.
  bool read_as_f32(const std::string& name, std::vector<float>& out_f32) const;
  // Zero-copy view into mmapped data section for a named tensor. Returns false if not found.
  bool view_raw(const std::string& name, const uint8_t*& ptr, size_t& size) const;
  // Pin/unpin the mmapped region for faster HtoD (CUDA only). No-op if CUDA is unavailable or mmap failed.
  bool pin_mmapped() const;
  void unpin_mmapped() const;

 private:
  std::string path_;
  SafeTensorHeader header_;
  uint64_t header_len_{0};
  // mmap state (optional, may be null if mapping failed)
  void* map_{nullptr};
  size_t map_size_{0};
  int fd_{-1};
  mutable bool pinned_{false};
};

} // namespace trellis
