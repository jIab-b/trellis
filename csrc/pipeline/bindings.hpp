#pragma once

#include <string>
#include <vector>

namespace trellis {

struct WeightBinding {
  std::string name;
  std::string path;
};

class BindingTable {
 public:
  void add(const std::string& name, const std::string& path);
  const std::string& get(const std::string& name) const;
 private:
  std::vector<WeightBinding> items_;
};

} // namespace trellis
