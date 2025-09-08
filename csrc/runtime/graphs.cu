#include "graphs.hpp"
#include "../common.hpp"

#ifdef TRELLIS_HAVE_CUDA
#include <cuda_runtime.h>
static __global__ void graph_dummy_kernel() {}

namespace trellis::graphs {

void demo_capture_and_launch() {
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  graph_dummy_kernel<<<1,1,0,stream>>>();
  cudaStreamEndCapture(stream, &graph);
  if (graph) {
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
  cudaStreamDestroy(stream);
}

} // namespace trellis::graphs

#else

namespace trellis::graphs { void demo_capture_and_launch() {} }

#endif
