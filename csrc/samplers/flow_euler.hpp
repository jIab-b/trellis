#pragma once

#include <cstdint>

namespace trellis {

struct FlowEulerConfig {
  int steps{25};
  float sigma_min{1e-5f};
  float sigma_max{1.0f};
  float cfg_strength{5.0f};
  float cfg_lo{0.5f};
  float cfg_hi{1.0f};
  float rescale_t{3.0f};
};

class FlowEulerSampler {
 public:
  explicit FlowEulerSampler(const FlowEulerConfig& cfg);
  void sample(/* latent, models, cond */);
 private:
  FlowEulerConfig cfg_;
};

} // namespace trellis

