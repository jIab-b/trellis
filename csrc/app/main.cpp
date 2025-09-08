#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <optional>

#include "../pipeline/config.hpp"
#include "../pipeline/trellis_image_to_3d.hpp"
#include "../common.hpp"

namespace fs = std::filesystem;
using namespace trellis;

struct Args {
  std::string pipeline_json = "models/TRELLIS-image-large/pipeline.json";
  std::string weights_root = "models/TRELLIS-image-large/ckpts";
  std::optional<std::string> tokens_npz;
  std::optional<std::string> uncond_npz;
  std::string emb_index = "embeddings/index.json"; // not used yet in scaffold
  int item = 0;                                      // not used yet
  std::string target = "gs"; // gs|rf|mesh|all
  int steps_ss = 25;
  int steps_slat = 25;
  float sigma_min = 1e-5f;
  float sigma_max = 1.0f;
  float cfg = 5.0f;
  float cfg_lo = 0.5f, cfg_hi = 1.0f;
  std::string out_dir = "out";
};

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "  --pipeline <path>\n"
            << "  --weights-root <dir>\n"
            << "  --tokens <path> --uncond <path>   # preferred in scaffold\n"
            << "  --target <gs|rf|mesh|all>\n"
            << "  --steps-ss N --steps-slat N\n"
            << "  --sigma-min F --sigma-max F --cfg F --cfg-interval a,b\n"
            << "  --out <dir>\n";
}

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](int n){ if (i + n >= argc) TRELLIS_THROW("missing value for " + k); };
    if (k == "--pipeline") { need(1); a.pipeline_json = argv[++i]; }
    else if (k == "--weights-root") { need(1); a.weights_root = argv[++i]; }
    else if (k == "--tokens") { need(1); a.tokens_npz = argv[++i]; }
    else if (k == "--uncond") { need(1); a.uncond_npz = argv[++i]; }
    else if (k == "--target") { need(1); a.target = argv[++i]; }
    else if (k == "--steps-ss") { need(1); a.steps_ss = std::stoi(argv[++i]); }
    else if (k == "--steps-slat") { need(1); a.steps_slat = std::stoi(argv[++i]); }
    else if (k == "--sigma-min") { need(1); a.sigma_min = std::stof(argv[++i]); }
    else if (k == "--sigma-max") { need(1); a.sigma_max = std::stof(argv[++i]); }
    else if (k == "--cfg") { need(1); a.cfg = std::stof(argv[++i]); }
    else if (k == "--cfg-interval") { need(1); std::string v = argv[++i]; auto p=v.find(','); if(p==std::string::npos) TRELLIS_THROW("--cfg-interval expects a,b"); a.cfg_lo=std::stof(v.substr(0,p)); a.cfg_hi=std::stof(v.substr(p+1)); }
    else if (k == "--out") { need(1); a.out_dir = argv[++i]; }
    else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
    else { std::cerr << "Unknown arg: " << k << "\n"; usage(argv[0]); std::exit(2); }
  }
  return a;
}

int main(int argc, char** argv) {
  try {
    Args args = parse(argc, argv);

    if (!fs::exists(args.pipeline_json)) {
      std::cerr << "[warn] pipeline json not found: " << args.pipeline_json << " (scaffold does not parse it yet)\n";
    }
    if (!fs::exists(args.weights_root)) {
      std::cerr << "[warn] weights root not found: " << args.weights_root << "\n";
    }

    if (!args.tokens_npz || !args.uncond_npz) {
      std::cerr << "[note] Provide --tokens and --uncond paths. In scaffold, embeddings index parsing is not implemented.\n";
    }

    fs::create_directories(args.out_dir);

    PipelineConfig pc = make_minimal_config(args.weights_root);
    TrellisImageTo3D app(pc);
    EmbeddingPaths emb{ args.tokens_npz.value_or(""), args.uncond_npz.value_or("") };
    OutputsConfig out{ args.out_dir, args.target };

    std::cout << "[trellis] starting with target='" << args.target << "'\n";
    std::cout << "[trellis] tokens='" << emb.tokens_npz << "' uncond='" << emb.uncond_npz << "'\n";
    std::cout << "[trellis] weights_root='" << args.weights_root << "'\n";

    // Orchestrator call (will throw Not Implemented in scaffold)
    app.run(emb, out);

    std::cout << "[trellis] done\n";
    return 0;
  } catch (const trellis::Error& e) {
    std::cerr << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
}

