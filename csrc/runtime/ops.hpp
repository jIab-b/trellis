#pragma once

namespace trellis::ops {

// In-place add bias to vector of length len: y[i] += b[i]
void add_bias_inplace(float* d_y, const float* d_b, int len);

// In-place SiLU activation: y = x * sigmoid(x)
void silu_inplace(float* d_y, int len);

// Add channel bias across rows: for t in [0,T): y[t,i] += b[i]
void add_bias_rows_inplace(float* d_y, const float* d_b, int T, int C);

}
