#pragma once

#include <stdexcept>
#include <string>

namespace trellis {

struct Error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define TRELLIS_THROW(msg) throw ::trellis::Error(std::string("[trellis] ") + (msg))

#define TRELLIS_NOT_IMPLEMENTED() TRELLIS_THROW("Not implemented: " __FILE__ ":" + std::to_string(__LINE__))

} // namespace trellis

