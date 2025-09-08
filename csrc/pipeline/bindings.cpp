#include "bindings.hpp"
#include "../common.hpp"

#include <algorithm>

namespace trellis {

void BindingTable::add(const std::string& name, const std::string& path) {
  items_.push_back({name, path});
}

const std::string& BindingTable::get(const std::string& name) const {
  auto it = std::find_if(items_.begin(), items_.end(), [&](const WeightBinding& w){ return w.name == name; });
  if (it == items_.end()) TRELLIS_THROW("binding not found: " + name);
  return it->path;
}

} // namespace trellis

