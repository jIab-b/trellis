#pragma once

#include <string>
#include <vector>
#include "config.hpp"

namespace trellis {

struct EmbeddingPaths {
  std::string tokens_npz;
  std::string uncond_npz;
};

struct OutputsConfig {
  std::string out_dir;
  std::string target; // gs|rf|mesh|all
};

class TrellisImageTo3D {
 public:
  TrellisImageTo3D(const PipelineConfig& pc);
  void run(const EmbeddingPaths& emb, const OutputsConfig& out);
 private:
  PipelineConfig pc_;
};

} // namespace trellis

