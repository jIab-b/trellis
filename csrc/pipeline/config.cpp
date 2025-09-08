#include "config.hpp"

namespace trellis {

PipelineConfig make_minimal_config(const std::string& weights_root) {
  PipelineConfig pc;
  pc.weights_root = weights_root;
  pc.sparse_structure_flow_name = "ss_flow_img_dit_L_16l8_fp16";
  pc.sparse_structure_decoder_name = "ss_dec_conv3d_16l8_fp16";
  pc.slat_flow_name = "slat_flow_img_dit_L_64l8p2_fp16";
  pc.slat_decoder_gs_name = "slat_dec_gs_swin8_B_64l8gs32_fp16";
  pc.slat_decoder_rf_name = "slat_dec_rf_swin8_B_64l8r16_fp16";
  pc.slat_decoder_mesh_name = "slat_dec_mesh_swin8_B_64l8m256c_fp16";
  return pc;
}

} // namespace trellis

