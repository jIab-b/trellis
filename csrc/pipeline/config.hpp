#pragma once

#include <string>

namespace trellis {

struct SparseStructureFlowCfg { int resolution{16}; int in_channels{8}; int out_channels{8}; int model_channels{1024}; int num_blocks{24}; int num_heads{16}; int mlp_ratio{4}; int patch_size{1}; bool qk_rms_norm{true}; };
struct SparseStructureDecoderCfg { int latent_channels{8}; int out_channels{1}; };
struct SLatFlowCfg { int resolution{64}; int in_channels{8}; int out_channels{8}; int model_channels{1024}; int num_blocks{24}; int num_heads{16}; int mlp_ratio{4}; int patch_size{2}; bool qk_rms_norm{true}; };
struct SLatNormalizationCfg { float mean[8]; float std[8]; };

struct PipelineConfig {
  std::string weights_root;
  std::string sparse_structure_flow_name;
  std::string sparse_structure_decoder_name;
  std::string slat_flow_name;
  std::string slat_decoder_gs_name;
  std::string slat_decoder_rf_name;
  std::string slat_decoder_mesh_name;
  // Sampler params and normalization would be parsed here later.
};

// Scaffold: build from provided paths without parsing JSON.
PipelineConfig make_minimal_config(const std::string& weights_root);

} // namespace trellis

