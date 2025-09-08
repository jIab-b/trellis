#include "flow_euler.hpp"
#include "../common.hpp"
#include <vector>
#include <cmath>

#ifdef TRELLIS_HAVE_CUDA
#include "../runtime/graphs.hpp"
#endif

namespace trellis {

FlowEulerSampler::FlowEulerSampler(const FlowEulerConfig& cfg) : cfg_(cfg) {}

void FlowEulerSampler::sample() {
  // Minimal skeleton: build a log-linear sigma schedule and no-op iterate.
  const int S = cfg_.steps > 0 ? cfg_.steps : 1;
  std::vector<float> sigmas(S);
  const float s_min = cfg_.sigma_min > 0 ? cfg_.sigma_min : 1e-5f;
  const float s_max = cfg_.sigma_max > s_min ? cfg_.sigma_max : s_min * 10;
  const float log_min = std::log(s_min);
  const float log_max = std::log(s_max);
  for (int i = 0; i < S; ++i) {
    float t = (float)i / (float)(S - 1);
    sigmas[i] = std::exp(log_max + t * (log_min - log_max));
  }
  (void)sigmas;

#ifdef TRELLIS_HAVE_CUDA
  trellis::graphs::demo_capture_and_launch();
#endif
}

} // namespace trellis
