#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <cstring>

#include "device.hpp"

namespace trellis {

enum class DType { F32, F16, U8 };

struct Shape {
  std::vector<int64_t> dims;
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
  }
  std::string str() const;
};

inline std::string Shape::str() const {
  std::string s = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    s += std::to_string(dims[i]);
    if (i + 1 < dims.size()) s += ", ";
  }
  s += "]";
  return s;
}

struct Tensor {
  DType dtype{DType::F32};
  Shape shape;
  std::vector<uint8_t> host; // simple host-owned buffer for scaffolding

  size_t bytes() const {
    size_t bpe = (dtype == DType::F32 ? 4 : dtype == DType::F16 ? 2 : 1);
    return static_cast<size_t>(shape.numel()) * bpe;
  }

  void resize_bytes(size_t n) { host.resize(n); }
  void resize(const Shape& s) { shape = s; resize_bytes(bytes()); }
  void* data() { return host.data(); }
  const void* data() const { return host.data(); }
};

} // namespace trellis

namespace trellis {

struct DeviceTensor {
  DType dtype{DType::F32};
  Shape shape;
  void* ptr{nullptr};

  size_t bytes() const {
    size_t bpe = (dtype == DType::F32 ? 4 : dtype == DType::F16 ? 2 : 1);
    return static_cast<size_t>(shape.numel()) * bpe;
  }
  void allocate(const Shape& s, DType dt) {
    shape = s; dtype = dt; free(); ptr = device::malloc(bytes());
  }
  void free() {
    if (ptr) { device::free(ptr); ptr = nullptr; }
  }
  ~DeviceTensor() { free(); }
};

} // namespace trellis
