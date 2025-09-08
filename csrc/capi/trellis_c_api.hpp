#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Minimal, versioned C API for a single-call inference.
// This pre-alpha ABI uses file paths (embeddings, weights root) rather than
// full pointer manifests. It can be extended in future versions.

#define TRELLIS_C_API_VERSION 1

typedef struct TrellisRunSpecV1 {
  // Paths (UTF-8, null-terminated)
  const char* weights_root;  // directory containing .safetensors
  const char* tokens_npz;    // path to .npz with member "tokens.npy"
  const char* uncond_npz;    // path to .npz with member "tokens.npy"
  const char* out_dir;       // output directory to write files
  const char* target;        // "gs"|"rf"|"mesh"|"all"

  // Sampler params (subset)
  int32_t steps_ss;
  int32_t steps_slat;
  float sigma_min;
  float sigma_max;
  float cfg_strength;
  float cfg_interval_lo;
  float cfg_interval_hi;
} TrellisRunSpecV1;

typedef struct TrellisErrorBuf {
  char* message;     // buffer to write a null-terminated error message
  uint64_t capacity; // capacity of message buffer in bytes (including terminator)
} TrellisErrorBuf;

// Returns 0 on success, non-zero on failure. On failure, writes an error
// message into err->message if provided.
int trellis_run_v1(const TrellisRunSpecV1* spec, TrellisErrorBuf* err);

#ifdef __cplusplus
} // extern "C"
#endif

