#include "npz.hpp"
#include "../common.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#if TRELLIS_HAVE_ZLIB
#include <zlib.h>
#endif

namespace trellis {

static uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1]<<8)); }
static uint32_t rd32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24); }
static uint64_t rd64(const uint8_t* p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p+4) << 32); }

struct ZipEntry {
  std::string name;
  uint16_t method{0};
  uint32_t comp_size{0};
  uint32_t uncomp_size{0};
  uint32_t local_header_ofs{0};
};

static std::vector<uint8_t> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) TRELLIS_THROW("cannot open file: " + path);
  f.seekg(0, std::ios::end);
  size_t n = (size_t)f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(n);
  if (n) f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)n);
  return buf;
}

static bool parse_eocd(const std::vector<uint8_t>& z, uint32_t& cd_ofs, uint32_t& cd_size, uint16_t& entries) {
  // EOCD signature 0x06054b50, located within last 64K+22 bytes
  size_t max_back = std::min<size_t>(z.size(), 0x10000 + 22);
  for (size_t back = 22; back < max_back; ++back) {
    size_t i = z.size() - back;
    if (z[i+0]==0x50 && z[i+1]==0x4b && z[i+2]==0x05 && z[i+3]==0x06) {
      entries = rd16(&z[i+10]);
      cd_size = rd32(&z[i+12]);
      cd_ofs  = rd32(&z[i+16]);
      return true;
    }
  }
  return false;
}

static std::vector<ZipEntry> parse_central_dir(const std::vector<uint8_t>& z, uint32_t cd_ofs, uint16_t entries) {
  std::vector<ZipEntry> out;
  size_t i = cd_ofs;
  for (uint16_t e = 0; e < entries; ++e) {
    if (!(z[i+0]==0x50 && z[i+1]==0x4b && z[i+2]==0x01 && z[i+3]==0x02)) TRELLIS_THROW("zip: bad central dir sig");
    uint16_t method = rd16(&z[i+10]);
    uint32_t comp   = rd32(&z[i+20]);
    uint32_t uncomp = rd32(&z[i+24]);
    uint16_t nlen   = rd16(&z[i+28]);
    uint16_t xlen   = rd16(&z[i+30]);
    uint16_t clen   = rd16(&z[i+32]);
    uint32_t lho    = rd32(&z[i+42]);
    i += 46;
    std::string name(reinterpret_cast<const char*>(&z[i]), nlen);
    i += nlen + xlen + clen;
    out.push_back({name, method, comp, uncomp, lho});
  }
  return out;
}

static void read_local_header(const std::vector<uint8_t>& z, const ZipEntry& e, size_t& data_ofs) {
  size_t i = e.local_header_ofs;
  if (!(z[i+0]==0x50 && z[i+1]==0x4b && z[i+2]==0x03 && z[i+3]==0x04)) TRELLIS_THROW("zip: bad local header sig");
  uint16_t nlen = rd16(&z[i+26]);
  uint16_t xlen = rd16(&z[i+28]);
  data_ofs = i + 30 + nlen + xlen;
}

static std::vector<uint8_t> unzip_member(const std::vector<uint8_t>& z, const ZipEntry& e) {
  size_t data_ofs = 0;
  read_local_header(z, e, data_ofs);
  const uint8_t* src = &z[data_ofs];
  if (e.method == 0) {
    // stored
    return std::vector<uint8_t>(src, src + e.comp_size);
  } else if (e.method == 8) {
#if TRELLIS_HAVE_ZLIB
    std::vector<uint8_t> out(e.uncomp_size);
    // zlib inflate raw deflate stream
    z_stream strm{};
    strm.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(src));
    strm.avail_in = e.comp_size;
    strm.next_out = reinterpret_cast<Bytef*>(out.data());
    strm.avail_out = e.uncomp_size;
    if (inflateInit2(&strm, -MAX_WBITS) != Z_OK) TRELLIS_THROW("zlib inflateInit2 failed");
    int ret = inflate(&strm, Z_FINISH);
    inflateEnd(&strm);
    if (ret != Z_STREAM_END) TRELLIS_THROW("zlib inflate failed");
    return out;
#else
    TRELLIS_THROW("zip: DEFLATE member requires zlib at build time");
#endif
  } else {
    TRELLIS_THROW("zip: unsupported compression method: " + std::to_string(e.method));
  }
}

static NpyArray parse_npy(const std::vector<uint8_t>& data) {
  // NPY magic: \x93NUMPY
  if (data.size() < 10) TRELLIS_THROW("npy: file too small");
  if (!(data[0]==0x93 && data[1]=='N' && data[2]=='U' && data[3]=='M' && data[4]=='P' && data[5]=='Y'))
    TRELLIS_THROW("npy: bad magic");
  uint8_t major = data[6];
  uint8_t minor = data[7];
  size_t pos = 8;
  uint32_t hlen = 0;
  if (major == 1) {
    hlen = rd16(&data[pos]); pos += 2;
  } else if (major >= 2) {
    hlen = rd32(&data[pos]); pos += 4;
  } else {
    TRELLIS_THROW("npy: unsupported version");
  }
  if (pos + hlen > data.size()) TRELLIS_THROW("npy: short header");
  std::string header(reinterpret_cast<const char*>(&data[pos]), hlen);
  pos += hlen;

  // Parse header dict by simple searches (format is restricted)
  auto find_str = [&](const char* key)->std::string{
    size_t k = header.find(std::string("'" ) + key + "':");
    if (k==std::string::npos) k = header.find(std::string("\"") + key + "\":");
    if (k==std::string::npos) TRELLIS_THROW(std::string("npy: missing key ")+key);
    size_t q1 = header.find_first_of("'\"", k);
    if (q1==std::string::npos) TRELLIS_THROW("npy: bad header (q1)");
    size_t q2 = header.find(header[q1], q1+1);
    if (q2==std::string::npos) TRELLIS_THROW("npy: bad header (q2)");
    return header.substr(q1+1, q2-(q1+1));
  };
  auto find_shape = [&](){
    size_t k = header.find("shape");
    if (k==std::string::npos) TRELLIS_THROW("npy: missing shape");
    size_t l = header.find('(', k);
    size_t r = header.find(')', l);
    if (l==std::string::npos || r==std::string::npos || r<l) TRELLIS_THROW("npy: bad shape");
    std::vector<int64_t> shp;
    std::string inside = header.substr(l+1, r-(l+1));
    size_t start=0;
    while (start < inside.size()) {
      while (start<inside.size() && std::isspace(static_cast<unsigned char>(inside[start]))) ++start;
      size_t end=start;
      while (end<inside.size() && (std::isdigit(static_cast<unsigned char>(inside[end])))) ++end;
      if (end>start) {
        shp.push_back(std::stoll(inside.substr(start, end-start)));
      }
      size_t comma = inside.find(',', end);
      if (comma==std::string::npos) break;
      start = comma+1;
    }
    return shp;
  };

  NpyArray arr;
  arr.dtype = find_str("descr");
  arr.shape = find_shape();
  if (pos > data.size()) TRELLIS_THROW("npy: no payload");
  arr.data.assign(data.begin()+pos, data.end());
  return arr;
}

NpyArray load_from_npz(const std::string& npz_path, const std::string& member_name) {
  auto buf = slurp(npz_path);
  uint32_t cd_ofs=0, cd_size=0; uint16_t entries=0;
  if (!parse_eocd(buf, cd_ofs, cd_size, entries)) TRELLIS_THROW("zip: EOCD not found");
  auto ents = parse_central_dir(buf, cd_ofs, entries);
  const ZipEntry* target = nullptr;
  for (const auto& e : ents) {
    if (e.name == member_name) { target = &e; break; }
  }
  if (!target) TRELLIS_THROW("zip: member not found: " + member_name);
  auto npy_bytes = unzip_member(buf, *target);
  return parse_npy(npy_bytes);
}

} // namespace trellis
