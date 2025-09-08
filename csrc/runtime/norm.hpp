#pragma once

namespace trellis::norm {

// LayerNorm over last dimension C for T rows: y[t,:] = (x[t,:]-mean)/sqrt(var+eps)*gamma + beta
// All pointers are device FP32, shapes: x/y [T,C], gamma/beta [C].
void layernorm_f32(const float* d_x, float* d_y, int T, int C, const float* d_gamma, const float* d_beta, float eps);

}

