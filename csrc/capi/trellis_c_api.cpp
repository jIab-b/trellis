#include "trellis_c_api.hpp"

#include <cstring>
#include <string>
#include <exception>

#include "../common.hpp"
#include "../pipeline/config.hpp"
#include "../pipeline/trellis_image_to_3d.hpp"

using namespace trellis;

static void write_err(TrellisErrorBuf* err, const std::string& msg) {
  if (!err || !err->message || err->capacity == 0) return;
  size_t n = msg.size();
  if (n + 1 > err->capacity) n = (size_t)err->capacity - 1;
  std::memcpy(err->message, msg.data(), n);
  err->message[n] = '\0';
}

extern "C" int trellis_run_v1(const TrellisRunSpecV1* spec, TrellisErrorBuf* err) {
  try {
    if (!spec) TRELLIS_THROW("trellis_run_v1: spec is null");
    if (!spec->weights_root || !spec->tokens_npz || !spec->uncond_npz || !spec->out_dir || !spec->target) {
      TRELLIS_THROW("trellis_run_v1: one or more required string fields are null");
    }

    PipelineConfig pc = make_minimal_config(spec->weights_root);
    TrellisImageTo3D app(pc);
    EmbeddingPaths emb{spec->tokens_npz, spec->uncond_npz};
    OutputsConfig out{spec->out_dir, spec->target};
    app.run(emb, out);
    return 0;
  } catch (const trellis::Error& e) {
    write_err(err, e.what());
    return 1;
  } catch (const std::exception& e) {
    write_err(err, std::string("[error] ") + e.what());
    return 2;
  } catch (...) {
    write_err(err, "[error] unknown exception");
    return 3;
  }
}

