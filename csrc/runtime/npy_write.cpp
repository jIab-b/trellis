#include "npy_write.hpp"
#include "../common.hpp"

#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdint>

namespace trellis {

static void write_all(std::ofstream& out, const void* data, size_t n) {
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
  if (!out) TRELLIS_THROW("npy_write: short write");
}

static std::string make_npy_header_v1(const std::vector<long long>& shape) {
  // Build the dict: {'descr': '<f4', 'fortran_order': False, 'shape': (d0, d1, ...), }
  std::ostringstream dict;
  dict << "{";
  dict << "'descr': '<f4', ";
  dict << "'fortran_order': False, ";
  dict << "'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    dict << shape[i];
    if (shape.size() == 1) dict << ","; // python tuple requires trailing comma for 1-elt
    if (i + 1 < shape.size()) dict << ", ";
  }
  dict << ")";
  dict << "}";

  std::string header = dict.str();
  // v1.0: magic + version(1,0) + header_len(2 bytes little-endian) + header padded to 16-byte alignment and ending with \n
  const char magic[] = "\x93NUMPY"; // 6 bytes
  std::string out;
  out.append(magic, magic + 6);
  out.push_back(1); // major
  out.push_back(0); // minor

  // compute padded header
  // header_len counts only the dict + padding + newline (not including the initial 10 bytes)
  size_t preamble = 10; // 6 magic + 2 ver + 2 hlen
  size_t hlen = header.size();
  size_t total = preamble + 2 + hlen; // we'll replace +2 with actual hlen field below
  // We'll pad with spaces so that the total file position after header ends on 16-byte alignment
  size_t pad = 16 - ((10 + 2 + hlen) % 16);
  if (pad == 16) pad = 0;
  std::string header_padded = header;
  header_padded.append(pad, ' ');
  header_padded.push_back('\n');

  uint16_t hlen_le = static_cast<uint16_t>(header_padded.size());
  out.push_back(static_cast<char>(hlen_le & 0xFF));
  out.push_back(static_cast<char>((hlen_le >> 8) & 0xFF));
  out += header_padded;
  return out;
}

bool write_npy_f32(const std::string& path,
                   const float* data,
                   size_t elem_count,
                   const std::vector<long long>& shape) {
  std::ofstream out(path, std::ios::binary);
  if (!out) TRELLIS_THROW(std::string("npy_write: cannot open ") + path);
  std::string hdr = make_npy_header_v1(shape);
  write_all(out, hdr.data(), hdr.size());
  if (elem_count) write_all(out, data, elem_count * sizeof(float));
  return true;
}

} // namespace trellis
