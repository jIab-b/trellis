#include "safetensors.hpp"
#include "json.hpp"
#include "../common.hpp"

#include <vector>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef TRELLIS_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace trellis {

static uint64_t read_u64_le(std::ifstream& in) {
  uint64_t v = 0;
  char buf[8];
  in.read(buf, 8);
  if (!in) TRELLIS_THROW("Failed to read safetensors header length");
  std::memcpy(&v, buf, 8);
  return v; // host assumed little-endian
}

SafeTensorsFile::SafeTensorsFile(const std::string& path) : path_(path) {
  std::ifstream in(path_, std::ios::binary);
  if (!in) TRELLIS_THROW(("Cannot open safetensors file: ") + path_);
  const uint64_t header_len = read_u64_le(in);
  header_len_ = header_len;
  std::string j(header_len, '\0');
  in.read(j.data(), static_cast<std::streamsize>(header_len));
  if (!in) TRELLIS_THROW("Short read of safetensors header");
  header_.raw_json = std::move(j);

  // Parse header JSON to collect tensor entries.
  using namespace trellis::json;
  Value root = parse(header_.raw_json);
  if (!root.is_object()) TRELLIS_THROW("safetensors header root is not an object");
  for (const auto& kv : root.o) {
    const std::string& name = kv.first;
    if (name == "__metadata__") continue;
    const Value& spec = kv.second;
    if (!spec.is_object()) continue;
    SafeTensorInfo info;
    info.name = name;
    // dtype
    if (const Value* v = get(spec.o, "dtype"); v && v->is_string()) {
      info.dtype = v->s;
    }
    // shape
    if (const Value* v = get(spec.o, "shape"); v && v->is_array()) {
      for (const auto& x : v->a) {
        if (!x.is_number()) TRELLIS_THROW("safetensors: shape element not number for " + name);
        info.shape.push_back(static_cast<int64_t>(x.num));
      }
    }
    // data_offsets
    if (const Value* v = get(spec.o, "data_offsets"); v && v->is_array() && v->a.size()==2) {
      if (!v->a[0].is_number() || !v->a[1].is_number()) TRELLIS_THROW("safetensors: data_offsets not numbers for " + name);
      info.offset_begin = static_cast<uint64_t>(v->a[0].num);
      info.offset_end   = static_cast<uint64_t>(v->a[1].num);
    }
    header_.tensors.push_back(std::move(info));
  }
  // mmap the entire file for zero-copy views (best-effort)
  struct stat st;
  fd_ = ::open(path_.c_str(), O_RDONLY);
  if (fd_ >= 0 && fstat(fd_, &st) == 0) {
    map_size_ = static_cast<size_t>(st.st_size);
    if (map_size_ >= 8 + header_len_) {
      void* p = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (p != MAP_FAILED) {
        map_ = p;
      }
    }
  }
}

const SafeTensorInfo* SafeTensorsFile::find(const std::string& name) const {
  for (const auto& t : header_.tensors) if (t.name == name) return &t;
  return nullptr;
}

bool SafeTensorsFile::read_raw(const std::string& name, std::vector<uint8_t>& out) const {
  const SafeTensorInfo* t = find(name);
  if (!t) return false;
  const uint64_t sz = t->offset_end - t->offset_begin;
  out.resize(static_cast<size_t>(sz));
  if (map_) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(map_);
    const uint8_t* data = base + 8 + header_len_ + t->offset_begin;
    std::memcpy(out.data(), data, static_cast<size_t>(sz));
  } else {
    std::ifstream in(path_, std::ios::binary);
    if (!in) TRELLIS_THROW("safetensors: cannot reopen file for read_raw");
    // skip header (8B len + header JSON)
    uint64_t header_len = read_u64_le(in);
    in.seekg(static_cast<std::streamoff>(header_len), std::ios::cur);
    in.seekg(static_cast<std::streamoff>(t->offset_begin), std::ios::cur);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sz));
    if (!in) TRELLIS_THROW("safetensors: short read for tensor");
  }
  return true;
}

static inline float f16_to_f32(uint16_t h) {
  // IEEE 754 half to float conversion
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp  = (h & 0x7C00u) >> 10;
  uint32_t mant = (h & 0x03FFu);
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      // subnormal
      exp = 127 - 15 + 1; // renormalize
      while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
      mant &= 0x03FFu;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    // inf/NaN
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    exp = exp + (127 - 15);
    f = sign | (exp << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

bool SafeTensorsFile::read_as_f32(const std::string& name, std::vector<float>& out_f32) const {
  const SafeTensorInfo* t = find(name);
  if (!t) return false;
  std::vector<uint8_t> raw;
  if (!read_raw(name, raw)) return false;
  size_t elems = 1; for (auto d : t->shape) elems *= static_cast<size_t>(d);
  out_f32.resize(elems);
  if (t->dtype == "F32") {
    if (raw.size() != elems * 4) TRELLIS_THROW("safetensors: F32 size mismatch");
    std::memcpy(out_f32.data(), raw.data(), raw.size());
  } else if (t->dtype == "F16") {
    if (raw.size() != elems * 2) TRELLIS_THROW("safetensors: F16 size mismatch");
    const uint16_t* p = reinterpret_cast<const uint16_t*>(raw.data());
    for (size_t i = 0; i < elems; ++i) out_f32[i] = f16_to_f32(p[i]);
  } else {
    TRELLIS_THROW("safetensors: unsupported dtype for read_as_f32: " + t->dtype);
  }
  return true;
}

bool SafeTensorsFile::view_raw(const std::string& name, const uint8_t*& ptr, size_t& size) const {
  const SafeTensorInfo* t = find(name);
  if (!t) return false;
  if (!map_) return false;
  const uint8_t* base = reinterpret_cast<const uint8_t*>(map_);
  ptr = base + 8 + header_len_ + t->offset_begin;
  size = static_cast<size_t>(t->offset_end - t->offset_begin);
  return true;
}

// Cleanup mmap
SafeTensorsFile::~SafeTensorsFile() {
#ifdef TRELLIS_HAVE_CUDA
  if (pinned_) {
    cudaHostUnregister(map_);
    pinned_ = false;
  }
#endif
  if (map_) {
    munmap(map_, map_size_);
    map_ = nullptr; map_size_ = 0;
  }
  if (fd_ >= 0) {
    ::close(fd_); fd_ = -1;
  }
}

bool SafeTensorsFile::pin_mmapped() const {
#ifdef TRELLIS_HAVE_CUDA
  if (!map_ || pinned_) return false;
  cudaError_t st = cudaHostRegister(map_, map_size_, cudaHostRegisterDefault);
  if (st == cudaSuccess) { pinned_ = true; return true; }
#endif
  return false;
}

void SafeTensorsFile::unpin_mmapped() const {
#ifdef TRELLIS_HAVE_CUDA
  if (map_ && pinned_) { cudaHostUnregister(map_); pinned_ = false; }
#endif
}

} // namespace trellis
