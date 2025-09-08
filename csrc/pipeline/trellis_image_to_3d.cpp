#include "trellis_image_to_3d.hpp"
#include "../common.hpp"
#include "../runtime/npz.hpp"
#include "../runtime/tensor.hpp"
#include "../runtime/device.hpp"
#include "../runtime/safetensors.hpp"
#include "../runtime/npy_write.hpp"
#include "../samplers/flow_euler.hpp"
#include "../runtime/gemm.hpp"
#include "../models/mlp.hpp"
#include "../runtime/cast.hpp"
#include "../models/attention.hpp"
#include "../models/dit_block.hpp"
#include "../models/ss_decoder_conv.hpp"
#include <iostream>
#include <filesystem>
#include <cmath>
#include <numeric>

namespace trellis {

TrellisImageTo3D::TrellisImageTo3D(const PipelineConfig& pc) : pc_(pc) {}

void TrellisImageTo3D::run(const EmbeddingPaths& emb, const OutputsConfig& out) {
  // Load embeddings (npz -> npy)
  if (emb.tokens_npz.empty() || emb.uncond_npz.empty()) {
    TRELLIS_THROW("Embedding paths are empty; pass --tokens and --uncond");
  }
  auto tok = trellis::load_from_npz(emb.tokens_npz, "tokens.npy");
  auto unc = trellis::load_from_npz(emb.uncond_npz, "tokens.npy");
  std::cout << "[trellis] tokens: dtype=" << tok.dtype << " shape=";
  for (size_t i=0;i<tok.shape.size();++i) { std::cout << (i?"x":"") << tok.shape[i]; }
  std::cout << "  bytes=" << tok.data.size() << "\n";
  std::cout << "[trellis] uncond: dtype=" << unc.dtype << " shape=";
  for (size_t i=0;i<unc.shape.size();++i) { std::cout << (i?"x":"") << unc.shape[i]; }
  std::cout << "  bytes=" << unc.data.size() << "\n";

  auto numel_from_shape = [](const std::vector<int64_t>& shp) -> size_t {
    return std::accumulate(shp.begin(), shp.end(), (size_t)1, [](size_t a, int64_t b){ return a * (size_t)b; });
  };

  auto f16_to_f32 = [](uint16_t h)->float{
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);
    uint32_t f;
    if (exp == 0) {
      if (mant == 0) {
        f = sign;
      } else {
        exp = 127 - 15 + 1;
        while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
        mant &= 0x03FFu;
        f = sign | (exp << 23) | (mant << 13);
      }
    } else if (exp == 0x1F) {
      f = sign | 0x7F800000u | (mant << 13);
    } else {
      exp = exp + (127 - 15);
      f = sign | (exp << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
  };

  auto npy_to_f32 = [&](const trellis::NpyArray& arr) -> std::vector<float> {
    const std::string& dt = arr.dtype;
    size_t elems = numel_from_shape(arr.shape);
    std::vector<float> outf(elems);
    if (dt.find("f4") != std::string::npos) {
      if (arr.data.size() != elems * sizeof(float)) TRELLIS_THROW("npy: f4 size mismatch");
      std::memcpy(outf.data(), arr.data.data(), arr.data.size());
      return outf;
    } else if (dt.find("f2") != std::string::npos) {
      if (arr.data.size() != elems * sizeof(uint16_t)) TRELLIS_THROW("npy: f2 size mismatch");
      const uint16_t* p = reinterpret_cast<const uint16_t*>(arr.data.data());
      for (size_t i=0;i<elems;++i) outf[i] = f16_to_f32(p[i]);
      return outf;
    } else {
      TRELLIS_THROW("npy: unsupported dtype (expected f4 or f2): " + dt);
    }
  };

  std::vector<float> tok_f32 = npy_to_f32(tok);
  std::vector<float> unc_f32 = npy_to_f32(unc);

  const bool have_cuda = trellis::device::available();
  if (!have_cuda) {
    std::cout << "[trellis] CUDA device not available; proceeding with CPU-only scaffold path (no GPU uploads).\n";
  } else {
    auto mem0 = trellis::device::meminfo();
    std::cout << "[trellis] VRAM before: free=" << (mem0.free_bytes/(1024*1024)) << " MiB / total=" << (mem0.total_bytes/(1024*1024)) << " MiB\n";
  }

  // Upload tokens/uncond to device as float32 (if CUDA is available)
  trellis::DeviceTensor d_tokens;
  trellis::DeviceTensor d_uncond;
  if (have_cuda) {
    d_tokens.allocate({tok.shape}, trellis::DType::F32);
    d_uncond.allocate({unc.shape}, trellis::DType::F32);
    trellis::device::memcpy_htod(d_tokens.ptr, tok_f32.data(), tok_f32.size()*sizeof(float));
    trellis::device::memcpy_htod(d_uncond.ptr, unc_f32.data(), unc_f32.size()*sizeof(float));

    auto mem1 = trellis::device::meminfo();
    std::cout << "[trellis] VRAM after tokens: free=" << (mem1.free_bytes/(1024*1024)) << " MiB\n";
  }

  // Allocate SS latent (16^3 tokens x 1024 dims) as FP16 for low VRAM
  const int64_t ss_tokens = 16*16*16; const int64_t model_dim = 1024; 
  trellis::DeviceTensor d_ss_latent;
  if (have_cuda) {
    d_ss_latent.allocate({{1, ss_tokens, model_dim}}, trellis::DType::F16);
    auto mem2 = trellis::device::meminfo();
    std::cout << "[trellis] VRAM after SS latent: free=" << (mem2.free_bytes/(1024*1024)) << " MiB\n";
  }

  // Demo: open SS flow weights and inspect one tensor (e.g., pos_emb)
  namespace fs = std::filesystem;
  fs::path wt_root = pc_.weights_root;
  fs::path ss_flow = wt_root / (pc_.sparse_structure_flow_name + ".safetensors");
  if (fs::exists(ss_flow)) {
    trellis::SafeTensorsFile st(ss_flow.string());
    if (have_cuda) {
      if (st.pin_mmapped()) {
        std::cout << "[trellis] pinned safetensors mmap for faster HtoD: " << ss_flow << "\n";
      }
    }
    if (auto t = st.find("pos_emb")) {
      std::cout << "[trellis] ss_flow pos_emb dtype=" << t->dtype << " shape=";
      for (size_t i=0;i<t->shape.size();++i) std::cout << (i?"x":"") << t->shape[i];
      std::cout << " offsets=[" << t->offset_begin << "," << t->offset_end << "]\n";
      // Stream pos_emb to host then device to demonstrate staging
      std::vector<float> pos_emb_f32;
      if (st.read_as_f32("pos_emb", pos_emb_f32)) {
        // cheap checksum
        double acc = 0.0; for (size_t i=0;i<pos_emb_f32.size(); i+=4096) acc += pos_emb_f32[i];
        std::cout << "[trellis] pos_emb loaded (f32) elems=" << pos_emb_f32.size() << " checksum=" << acc << "\n";
        if (have_cuda) {
          trellis::DeviceTensor d_pos; d_pos.allocate({{t->shape[0], t->shape[1]}}, trellis::DType::F32);
          trellis::device::memcpy_htod(d_pos.ptr, pos_emb_f32.data(), pos_emb_f32.size()*sizeof(float));
          auto mem3 = trellis::device::meminfo();
          std::cout << "[trellis] VRAM after pos_emb upload: free=" << (mem3.free_bytes/(1024*1024)) << " MiB\n";
          // Free immediately (staged upload demo)
          d_pos.free();
        }
      }
    } else {
      std::cout << "[trellis] ss_flow pos_emb not found in header\n";
    }
  } else {
    std::cout << "[trellis] warn: weights file not found: " << ss_flow << "\n";
  }

  // Release tokens early if desired to cap VRAM; keep in this demo
  (void)out;
  // Demo: SS flow first-block wiring (host math demo)
  try {
    namespace fs = std::filesystem;
    fs::path ss_flow = fs::path(pc_.weights_root) / (pc_.sparse_structure_flow_name + ".safetensors");
    trellis::SafeTensorsFile st(ss_flow.string());
    auto rd = [&](const std::string& n){ std::vector<float> v; if(!st.read_as_f32(n, v)) TRELLIS_THROW("missing tensor: "+n); return v; };
    auto dot_mv = [](const std::vector<float>& W, const std::vector<float>& x, int out_dim, int in_dim){
      std::vector<float> y(out_dim, 0.0f);
      for (int o=0;o<out_dim;++o){ const float* wrow = &W[o*in_dim]; float acc=0; for (int i=0;i<in_dim;++i) acc += wrow[i]*x[i]; y[o]=acc; }
      return y;
    };
    auto add_bias = [](std::vector<float>& y, const std::vector<float>& b){ for (size_t i=0;i<y.size();++i) y[i]+=b[i]; };
    auto silu = [](std::vector<float>& y){ for (auto& v: y){ v = v / (1.0f + std::exp(-v)); } };

    // Time-embed MLP: input dim 256, hidden/out 1024
    const float sigma_demo = 0.7f; // placeholder
    std::vector<float> t_in(256, 0.0f); t_in[0] = sigma_demo;
    auto w0 = rd("t_embedder.mlp.0.weight"); // [1024,256]
    auto b0 = rd("t_embedder.mlp.0.bias");   // [1024]
    std::vector<float> t0 = dot_mv(w0, t_in, 1024, 256);
    add_bias(t0, b0); silu(t0);
    auto w2 = rd("t_embedder.mlp.2.weight"); // [1024,1024]
    auto b2 = rd("t_embedder.mlp.2.bias");   // [1024]
    std::vector<float> t1 = dot_mv(w2, t0, 1024, 1024);
    add_bias(t1, b2);
    double t_checksum=0; for (int i=0;i<1024;i+=64) t_checksum+=t1[i];
    std::cout << "[trellis] t-embed computed (demo) checksum=" << t_checksum << "\n";

    // Self-attn block 0 linear projections (demo on a single token vector x0=0)
    std::vector<float> x0(1024, 0.0f);
    auto wqkv = rd("blocks.0.self_attn.to_qkv.weight"); // [3072,1024]
    auto bqkv = rd("blocks.0.self_attn.to_qkv.bias");   // [3072]
    std::vector<float> qkv = dot_mv(wqkv, x0, 3072, 1024); add_bias(qkv, bqkv);
    // Project out
    auto wout = rd("blocks.0.self_attn.to_out.weight"); // [1024,1024]
    auto bout = rd("blocks.0.self_attn.to_out.bias");   // [1024]
    std::vector<float> y0 = dot_mv(wout, x0, 1024, 1024); add_bias(y0, bout);
    double y_checksum=0; for (int i=0;i<1024;i+=64) y_checksum+=y0[i];
    std::cout << "[trellis] attn block0 linear projections (demo) checksum=" << y_checksum << " (no attention softmax applied)\n";

    // GPU MLP demo: compute t1_gpu = MLP(t_in) using cuBLAS + device ops, compare with CPU t1
    if (trellis::device::available()) {
      const int in_dim = 256;
      const int hid = 1024;
      const int out_dim = 1024;
      size_t bytes_x = (size_t)in_dim * sizeof(float);
      size_t bytes_w0 = (size_t)hid * in_dim * sizeof(float);
      size_t bytes_b0 = (size_t)hid * sizeof(float);
      size_t bytes_w1 = (size_t)out_dim * hid * sizeof(float);
      size_t bytes_b1 = (size_t)out_dim * sizeof(float);
      size_t bytes_y = (size_t)out_dim * sizeof(float);

      void* dx = trellis::device::malloc(bytes_x);
      void* dW0 = trellis::device::malloc(bytes_w0);
      void* db0 = trellis::device::malloc(bytes_b0);
      void* dW1 = trellis::device::malloc(bytes_w1);
      void* db1 = trellis::device::malloc(bytes_b1);
      void* dy = trellis::device::malloc(bytes_y);

      trellis::device::memcpy_htod(dx, t_in.data(), bytes_x);
      trellis::device::memcpy_htod(dW0, w0.data(), bytes_w0);
      trellis::device::memcpy_htod(db0, b0.data(), bytes_b0);
      trellis::device::memcpy_htod(dW1, w2.data(), bytes_w1);
      trellis::device::memcpy_htod(db1, b2.data(), bytes_b1);

      trellis::models::mlp2_silu_f32((const float*)dx, in_dim,
                                     (const float*)dW0, (const float*)db0, hid,
                                     (const float*)dW1, (const float*)db1, out_dim,
                                     (float*)dy);

      std::vector<float> y_gpu(out_dim);
      trellis::device::memcpy_dtoh(y_gpu.data(), dy, bytes_y);

      trellis::device::free(dx); trellis::device::free(dW0); trellis::device::free(db0);
      trellis::device::free(dW1); trellis::device::free(db1); trellis::device::free(dy);

      // Compare a few entries
      double diff=0.0, ref=0.0;
      for (int i=0;i<out_dim;i+=16) { double d = (double)y_gpu[i] - (double)t1[i]; diff += d*d; ref += (double)t1[i]*t1[i]; }
      double rel = std::sqrt(diff) / (std::sqrt(ref) + 1e-12);
      std::cout << "[trellis] MLP demo rel L2 (sampled) = " << rel << "\n";

      // Optional FP16 GEMM check for first layer: W0@x with FP16 inputs + FP32 accum
      try {
        void* dW0h = trellis::device::malloc(bytes_w0 / 2);
        void* dxh  = trellis::device::malloc(bytes_x / 2);
        void* dyf  = trellis::device::malloc(bytes_w0 / in_dim); // m*1*f32
        trellis::cast::fp32_to_fp16((const float*)dW0, dW0h, (size_t)hid * in_dim);
        trellis::cast::fp32_to_fp16((const float*)dx,  dxh,  (size_t)in_dim);
        trellis::gemm::f16_rowmajor_accum_f32(dW0h, dxh, (float*)dyf, hid, 1, in_dim, 1.0f, 0.0f);
        std::vector<float> t0_f16(hid);
        trellis::device::memcpy_dtoh(t0_f16.data(), dyf, (size_t)hid * sizeof(float));
        trellis::device::free(dW0h); trellis::device::free(dxh); trellis::device::free(dyf);
        double diff2=0.0, ref2=0.0; for (int i=0;i<hid;i+=16){ double d=(double)t0_f16[i] - (double)t0[i]; diff2+=d*d; ref2+=(double)t0[i]*t0[i]; }
        double rel2 = std::sqrt(diff2) / (std::sqrt(ref2) + 1e-12);
        std::cout << "[trellis] FP16 GEMM W0@x rel L2 (sampled) = " << rel2 << "\n";
      } catch (const std::exception& e) {
        std::cout << "[trellis] FP16 GEMM check failed: " << e.what() << "\n";
      }

      // Hook minimal DiT block (attn-only) to real block0 QKV/out weights for a small T
      try {
        const int Tsmall = 16; const int Cdim = 1024;
        // Input tokens X: simple deterministic pattern
        std::vector<float> X(Tsmall*Cdim);
        for (int t=0;t<Tsmall;++t){ for(int i=0;i<Cdim;++i){ X[t*Cdim+i] = 0.01f * ((t*7 + i*3) % 23 - 11); }}

        // Load weights from safetensors
        auto wqkv = rd("blocks.0.self_attn.to_qkv.weight"); // [3072,1024]
        auto bqkv = rd("blocks.0.self_attn.to_qkv.bias");   // [3072]
        auto wout = rd("blocks.0.self_attn.to_out.weight"); // [1024,1024]
        auto bout = rd("blocks.0.self_attn.to_out.bias");   // [1024]
        if ((int)wqkv.size() != 3*Cdim*Cdim || (int)bqkv.size() != 3*Cdim || (int)wout.size() != Cdim*Cdim || (int)bout.size() != Cdim) {
          throw std::runtime_error("unexpected block0 weight sizes");
        }

        // CPU baseline implementing LN(=identity gamma=1,beta=0), QKV, SDPA, OUT, residual
        auto cpu_layernorm = [&](const std::vector<float>& x)->std::vector<float>{
          std::vector<float> y(Tsmall*Cdim);
          for (int t=0;t<Tsmall;++t){
            const float* xp = &x[t*Cdim]; float* yp = &y[t*Cdim];
            double m=0.0, v=0.0; for (int i=0;i<Cdim;++i){ m += xp[i]; v += xp[i]*xp[i]; }
            m/=Cdim; v/=Cdim; v -= m*m; float inv = 1.0f/std::sqrt((float)v + 1e-5f);
            for (int i=0;i<Cdim;++i){ yp[i] = (xp[i]-m)*inv; }
          }
          return y;
        };
        auto cpu_mm = [&](const std::vector<float>& A, const std::vector<float>& B_T, int M, int N, int K){
          // A: [M,K], B_T: [N,K] (i.e., B^T), returns C: [M,N]
          std::vector<float> C(M*N, 0.0f);
          for (int m=0;m<M;++m){
            for (int n=0;n<N;++n){
              double acc=0.0; const float* a=&A[m*K]; const float* bt=&B_T[n*K];
              for (int k=0;k<K;++k) acc += (double)a[k]*(double)bt[k];
              C[m*N+n]=(float)acc;
            }
          }
          return C;
        };
        auto lnX = cpu_layernorm(X);
        // qkv = lnX @ Wqkv^T + b
        // Build Wqkv^T: [3072,1024]^T -> [1024,3072]; but mm uses B_T [N,K] with N=3072, K=1024
        std::vector<float> qkv = cpu_mm(lnX, wqkv, Tsmall, 3*Cdim, Cdim);
        for (int t=0;t<Tsmall;++t) for (int i=0;i<3*Cdim;++i) qkv[t*3*Cdim+i] += bqkv[i];
        auto ptrQ = [&](int t){ return &qkv[t*3*Cdim + 0*Cdim]; };
        auto ptrK = [&](int t){ return &qkv[t*3*Cdim + 1*Cdim]; };
        auto ptrV = [&](int t){ return &qkv[t*3*Cdim + 2*Cdim]; };
        // S = Q K^T / sqrt(C)
        std::vector<float> S(Tsmall*Tsmall);
        float invs = 1.0f/std::sqrt((float)Cdim);
        for (int t=0;t<Tsmall;++t){
          for (int u=0;u<Tsmall;++u){ double acc=0.0; const float* q=ptrQ(t); const float* k=ptrK(u);
            for (int i=0;i<Cdim;++i) acc += (double)q[i]*(double)k[i]; S[t*Tsmall+u]=(float)(acc*invs); }
        }
        // softmax rows
        for (int t=0;t<Tsmall;++t){
          float m=-1e30f; for (int u=0;u<Tsmall;++u) m = std::max(m, S[t*Tsmall+u]);
          double sum=0.0; for (int u=0;u<Tsmall;++u){ float e=std::exp(S[t*Tsmall+u]-m); S[t*Tsmall+u]=e; sum+=e; }
          float inv=(float)(1.0/(sum+1e-8)); for (int u=0;u<Tsmall;++u) S[t*Tsmall+u]*=inv;
        }
        // O = S @ V  -> [T,C]
        std::vector<float> O(Tsmall*Cdim, 0.0f);
        for (int t=0;t<Tsmall;++t){ for (int c=0;c<Cdim;++c){ double acc=0.0; for (int u=0;u<Tsmall;++u) acc += (double)S[t*Tsmall+u]*(double)ptrV(u)[c]; O[t*Cdim+c]=(float)acc; }}
        // proj = O @ Wout^T + bout
        std::vector<float> proj = cpu_mm(O, wout, Tsmall, Cdim, Cdim);
        for (int t=0;t<Tsmall;++t) for (int i=0;i<Cdim;++i) proj[t*Cdim+i] += bout[i];
        // y = X + proj
        std::vector<float> Y_cpu(Tsmall*Cdim);
        for (int i=0;i<Tsmall*Cdim;++i) Y_cpu[i] = X[i] + proj[i];

        // GPU run
        void *dx2=trellis::device::malloc(Tsmall*Cdim*sizeof(float));
        void *dy2=trellis::device::malloc(Tsmall*Cdim*sizeof(float));
        void *dg2=trellis::device::malloc(Cdim*sizeof(float));
        void *db2=trellis::device::malloc(Cdim*sizeof(float));
        std::vector<float> gamma(Cdim,1.0f), beta(Cdim,0.0f);
        trellis::device::memcpy_htod(dx2, X.data(), Tsmall*Cdim*sizeof(float));
        trellis::device::memcpy_htod(dg2, gamma.data(), Cdim*sizeof(float));
        trellis::device::memcpy_htod(db2, beta.data(), Cdim*sizeof(float));
        void *dwqkv2=trellis::device::malloc(wqkv.size()*sizeof(float));
        void *dbqkv2=trellis::device::malloc(bqkv.size()*sizeof(float));
        void *dwout2=trellis::device::malloc(wout.size()*sizeof(float));
        void *dbout2=trellis::device::malloc(bout.size()*sizeof(float));
        trellis::device::memcpy_htod(dwqkv2, wqkv.data(), wqkv.size()*sizeof(float));
        trellis::device::memcpy_htod(dbqkv2, bqkv.data(), bqkv.size()*sizeof(float));
        trellis::device::memcpy_htod(dwout2, wout.data(), wout.size()*sizeof(float));
        trellis::device::memcpy_htod(dbout2, bout.data(), bout.size()*sizeof(float));
        trellis::models::dit_block_attn_only_f32((const float*)dx2, Tsmall, Cdim, 16,
                                                 (const float*)dg2, (const float*)db2,
                                                 (const float*)dwqkv2, (const float*)dbqkv2,
                                                 (const float*)dwout2, (const float*)dbout2,
                                                 (float*)dy2);
        std::vector<float> Y_gpu(Tsmall*Cdim);
        trellis::device::memcpy_dtoh(Y_gpu.data(), dy2, Tsmall*Cdim*sizeof(float));
        trellis::device::free(dx2); trellis::device::free(dy2);
        trellis::device::free(dg2); trellis::device::free(db2);
        trellis::device::free(dwqkv2); trellis::device::free(dbqkv2);
        trellis::device::free(dwout2); trellis::device::free(dbout2);

        // Compare
        double diff=0.0, ref=0.0; for (size_t i=0;i<Y_gpu.size(); i+=97){ double d=(double)Y_gpu[i] - (double)Y_cpu[i]; diff+=d*d; ref += (double)Y_cpu[i]*(double)Y_cpu[i]; }
        double rel = std::sqrt(diff) / (std::sqrt(ref)+1e-12);
        std::cout << "[trellis] DiT block0 (attn-only) rel L2 (sampled) = " << rel << "\n";
      } catch (const std::exception& e) {
        std::cout << "[trellis] DiT block attach failed: " << e.what() << "\n";
      }

      // Full DiT block (attn + MLP) using real weights if available (no CPU baseline)
      try {
        const int Tsmall = 16; const int Cdim = 1024; const int H = 16; const int hidden = 4096;
        // Build input
        std::vector<float> X(Tsmall*Cdim); for (int t=0;t<Tsmall;++t) for (int i=0;i<Cdim;++i) X[t*Cdim+i]=0.01f*((t*5+i*11)%31-15);
        // Try several name candidates for LN and MLP
        auto try_vec = [&](const std::vector<std::string>& names, int expect)->std::vector<float>{
          for (auto& n : names) { std::vector<float> v; if (st.read_as_f32(n, v) && (int)v.size()==expect) return v; }
          return {}; };
        auto g1 = try_vec({"blocks.0.ln1.weight","blocks.0.pre_norm.weight","blocks.0.pre_attn_norm.weight"}, Cdim);
        auto b1 = try_vec({"blocks.0.ln1.bias","blocks.0.pre_norm.bias","blocks.0.pre_attn_norm.bias"}, Cdim);
        auto g2 = try_vec({"blocks.0.ln2.weight","blocks.0.post_norm.weight","blocks.0.pre_mlp_norm.weight"}, Cdim);
        auto b2 = try_vec({"blocks.0.ln2.bias","blocks.0.post_norm.bias","blocks.0.pre_mlp_norm.bias"}, Cdim);
        auto w0 = try_vec({"blocks.0.mlp.0.weight"}, hidden*Cdim);
        auto bb0 = try_vec({"blocks.0.mlp.0.bias"}, hidden);
        auto w1 = try_vec({"blocks.0.mlp.2.weight"}, Cdim*hidden);
        auto bb1 = try_vec({"blocks.0.mlp.2.bias"}, Cdim);
        if (g1.empty()||b1.empty()||g2.empty()||b2.empty()||w0.empty()||bb0.empty()||w1.empty()||bb1.empty()) {
          throw std::runtime_error("missing LN/MLP weights for full DiT block");
        }
        // Upload
        void *dx=trellis::device::malloc(Tsmall*Cdim*sizeof(float));
        void *dy=trellis::device::malloc(Tsmall*Cdim*sizeof(float));
        void *dg1=trellis::device::malloc(Cdim*sizeof(float)), *db1d=trellis::device::malloc(Cdim*sizeof(float));
        void *dg2d=trellis::device::malloc(Cdim*sizeof(float)), *db2d=trellis::device::malloc(Cdim*sizeof(float));
        void *dwq=trellis::device::malloc(wqkv.size()*sizeof(float)), *dbq=trellis::device::malloc(bqkv.size()*sizeof(float));
        void *dwo=trellis::device::malloc(wout.size()*sizeof(float)), *dbo=trellis::device::malloc(bout.size()*sizeof(float));
        void *dw0=trellis::device::malloc(w0.size()*sizeof(float)), *db0=trellis::device::malloc(bb0.size()*sizeof(float));
        void *dw1=trellis::device::malloc(w1.size()*sizeof(float)), *db1b=trellis::device::malloc(bb1.size()*sizeof(float));
        trellis::device::memcpy_htod(dx, X.data(), Tsmall*Cdim*sizeof(float));
        trellis::device::memcpy_htod(dg1, g1.data(), Cdim*sizeof(float));
        trellis::device::memcpy_htod(db1d, b1.data(), Cdim*sizeof(float));
        trellis::device::memcpy_htod(dg2d, g2.data(), Cdim*sizeof(float));
        trellis::device::memcpy_htod(db2d, b2.data(), Cdim*sizeof(float));
        trellis::device::memcpy_htod(dwq, wqkv.data(), wqkv.size()*sizeof(float));
        trellis::device::memcpy_htod(dbq, bqkv.data(), bqkv.size()*sizeof(float));
        trellis::device::memcpy_htod(dwo, wout.data(), wout.size()*sizeof(float));
        trellis::device::memcpy_htod(dbo, bout.data(), bout.size()*sizeof(float));
        trellis::device::memcpy_htod(dw0, w0.data(), w0.size()*sizeof(float));
        trellis::device::memcpy_htod(db0, bb0.data(), bb0.size()*sizeof(float));
        trellis::device::memcpy_htod(dw1, w1.data(), w1.size()*sizeof(float));
        trellis::device::memcpy_htod(db1b, bb1.data(), bb1.size()*sizeof(float));
        trellis::models::dit_block_full_f32((const float*)dx, Tsmall, Cdim, H,
                                            (const float*)dg1,(const float*)db1d,
                                            (const float*)dwq,(const float*)dbq,
                                            (const float*)dwo,(const float*)dbo,
                                            (const float*)dg2d,(const float*)db2d,
                                            (const float*)dw0,(const float*)db0, hidden,
                                            (const float*)dw1,(const float*)db1b,
                                            (float*)dy);
        std::vector<float> Y(Tsmall*Cdim); trellis::device::memcpy_dtoh(Y.data(), dy, Tsmall*Cdim*sizeof(float));
        double ck=0.0; for (size_t i=0;i<Y.size(); i+=137) ck += Y[i];
        std::cout << "[trellis] DiT block0 full (attn+MLP) checksum=" << ck << "\n";
        trellis::device::free(dx); trellis::device::free(dy);
        trellis::device::free(dg1); trellis::device::free(db1d);
        trellis::device::free(dg2d); trellis::device::free(db2d);
        trellis::device::free(dwq); trellis::device::free(dbq);
        trellis::device::free(dwo); trellis::device::free(dbo);
        trellis::device::free(dw0); trellis::device::free(db0);
        trellis::device::free(dw1); trellis::device::free(db1b);
      } catch (const std::exception& e) {
        std::cout << "[trellis] DiT full block demo skipped: " << e.what() << "\n";
      }
    }
  } catch (const std::exception& e) {
    std::cout << "[trellis] demo block wiring error: " << e.what() << "\n";
  }

  // Invoke a minimal sampler skeleton (no-op math) to validate control flow.
  try {
    trellis::FlowEulerConfig cfg; // defaults
    cfg.steps = 5; // shorten for scaffold
    cfg.sigma_min = 1e-5f; cfg.sigma_max = 1.0f;
    trellis::FlowEulerSampler sampler(cfg);
    sampler.sample();
    std::cout << "[trellis] sampler skeleton executed (no-op).\n";
  } catch (const std::exception& e) {
    std::cout << "[trellis] sampler error: " << e.what() << "\n";
  }
  std::cout << "[trellis] scaffold run complete (no outputs generated yet).\n";

  // Toy DiT block (attn-only) with synthetic small dims to exercise block forward
  if (trellis::device::available()) {
    try {
      const int Ttoy = 8, Ctoy = 64;
      size_t bytes_x = (size_t)Ttoy*Ctoy*sizeof(float);
      // Host init
      std::vector<float> h_x(Ttoy*Ctoy), h_gamma(Ctoy, 1.0f), h_beta(Ctoy, 0.0f);
      // fill h_x
      for (int t=0; t<Ttoy; ++t) {
        for (int i=0; i<Ctoy; ++i) {
          h_x[t*Ctoy+i] = ((t+1)*(i+3) % 17) * 0.03f;
        }
      }
      // Synthetic weights
      std::vector<float> h_wqkv(3*Ctoy*Ctoy), h_bqkv(3*Ctoy), h_wout(Ctoy*Ctoy), h_bout(Ctoy);
      for (int o=0;o<3*Ctoy;++o) for (int i=0;i<Ctoy;++i) h_wqkv[o*Ctoy+i] = 0.01f * ((o+i)%7 - 3);
      for (int o=0;o<3*Ctoy;++o) h_bqkv[o] = 0.001f * (o%5 - 2);
      for (int o=0;o<Ctoy;++o) for (int i=0;i<Ctoy;++i) h_wout[o*Ctoy+i] = (o==i) ? 0.9f : 0.0f;
      for (int i=0;i<Ctoy;++i) h_bout[i] = 0.0f;

      // Device alloc
      void *dx = trellis::device::malloc(bytes_x), *dy = trellis::device::malloc(bytes_x);
      void *dg = trellis::device::malloc(Ctoy*sizeof(float)), *db = trellis::device::malloc(Ctoy*sizeof(float));
      void *dwqkv = trellis::device::malloc(h_wqkv.size()*sizeof(float));
      void *dbqkv = trellis::device::malloc(h_bqkv.size()*sizeof(float));
      void *dwout = trellis::device::malloc(h_wout.size()*sizeof(float));
      void *dbout = trellis::device::malloc(h_bout.size()*sizeof(float));
      trellis::device::memcpy_htod(dx, h_x.data(), bytes_x);
      trellis::device::memcpy_htod(dg, h_gamma.data(), Ctoy*sizeof(float));
      trellis::device::memcpy_htod(db, h_beta.data(), Ctoy*sizeof(float));
      trellis::device::memcpy_htod(dwqkv, h_wqkv.data(), h_wqkv.size()*sizeof(float));
      trellis::device::memcpy_htod(dbqkv, h_bqkv.data(), h_bqkv.size()*sizeof(float));
      trellis::device::memcpy_htod(dwout, h_wout.data(), h_wout.size()*sizeof(float));
      trellis::device::memcpy_htod(dbout, h_bout.data(), h_bout.size()*sizeof(float));

      trellis::models::dit_block_attn_only_f32((const float*)dx, Ttoy, Ctoy, 8,
                                               (const float*)dg, (const float*)db,
                                               (const float*)dwqkv, (const float*)dbqkv,
                                               (const float*)dwout, (const float*)dbout,
                                               (float*)dy);
      std::vector<float> h_y(Ttoy*Ctoy);
      trellis::device::memcpy_dtoh(h_y.data(), dy, bytes_x);
      double cks=0.0; for (int i=0;i<Ttoy*Ctoy;i+=19) cks += h_y[i];
      std::cout << "[trellis] toy DiT block checksum=" << cks << "\n";
      trellis::device::free(dx); trellis::device::free(dy);
      trellis::device::free(dg); trellis::device::free(db);
      trellis::device::free(dwqkv); trellis::device::free(dbqkv);
      trellis::device::free(dwout); trellis::device::free(dbout);
    } catch (const std::exception& e) {
      std::cout << "[trellis] toy DiT error: " << e.what() << "\n";
    }
  }
  try {
    // SS decoding: small Conv3d stack
    fs::create_directories(out.out_dir);
    const int D = 16; // SS resolution
    const int T = D*D*D;
    const int Ctok = 1024; // token feature dim
    const int Cin = 8;     // input channels to decoder
    const int Cmid = 16;   // hidden channels
    // Build a synthetic tokens buffer
    std::vector<float> h_tokens(T*Ctok);
    for (int t=0;t<T;++t) for (int i=0;i<Ctok;++i) h_tokens[t*Ctok+i] = 0.01f * ((t + i*7) % 23 - 11);
    void* d_tokens = trellis::device::malloc((size_t)T*Ctok*sizeof(float));
    trellis::device::memcpy_htod(d_tokens, h_tokens.data(), (size_t)T*Ctok*sizeof(float));
    void* d_grid = trellis::device::malloc(T*sizeof(float));
    trellis::models::ss_decode_conv3d_f32((const float*)d_tokens, T, Ctok, D, Cin, Cmid, (float*)d_grid);
    std::vector<float> grid(T);
    trellis::device::memcpy_dtoh(grid.data(), d_grid, T*sizeof(float));
    trellis::device::free(d_tokens); trellis::device::free(d_grid);
    std::string out_path = (fs::path(out.out_dir) / "ss_grid_conv3d.npy").string();
    trellis::write_npy_f32(out_path, grid.data(), grid.size(), {1,1,D,D,D});
    std::cout << "[trellis] wrote SS grid (conv3d decoder): " << out_path << "\n";

    // If target requires GS, generate a minimal Gaussian Splat set
    auto write_gs = [&](int K){
      std::vector<float> xyz(K*3);
      std::vector<float> feat(K*3);
      std::vector<float> op(K*1);
      std::vector<float> sca(K*3);
      std::vector<float> rot(K*4);
      // Fill deterministic pattern
      int s = std::cbrt((double)K);
      if (s < 1) s = 1; int sx=s, sy=s, sz= s; int idx=0;
      for (int z=0; z<sz && idx<K; ++z) for (int y=0; y<sy && idx<K; ++y) for (int x=0; x<sx && idx<K; ++x) {
        float fx = -1.0f + 2.0f * (x + 0.5f) / (float)sx;
        float fy = -1.0f + 2.0f * (y + 0.5f) / (float)sy;
        float fz = -1.0f + 2.0f * (z + 0.5f) / (float)sz;
        xyz[idx*3+0]=fx; xyz[idx*3+1]=fy; xyz[idx*3+2]=fz;
        feat[idx*3+0]=0.7f; feat[idx*3+1]=0.7f; feat[idx*3+2]=0.7f;
        op[idx]=1.0f;
        sca[idx*3+0]=0.02f; sca[idx*3+1]=0.02f; sca[idx*3+2]=0.02f;
        rot[idx*4+0]=0.0f; rot[idx*4+1]=0.0f; rot[idx*4+2]=0.0f; rot[idx*4+3]=1.0f;
        ++idx;
      }
      trellis::write_npy_f32((fs::path(out.out_dir)/"gs_xyz.npy").string(), xyz.data(), xyz.size(), {(long long)K,3});
      trellis::write_npy_f32((fs::path(out.out_dir)/"gs_features_dc.npy").string(), feat.data(), feat.size(), {(long long)K,3});
      trellis::write_npy_f32((fs::path(out.out_dir)/"gs_opacity.npy").string(), op.data(), op.size(), {(long long)K,1});
      trellis::write_npy_f32((fs::path(out.out_dir)/"gs_scaling.npy").string(), sca.data(), sca.size(), {(long long)K,3});
      trellis::write_npy_f32((fs::path(out.out_dir)/"gs_rotation.npy").string(), rot.data(), rot.size(), {(long long)K,4});
      std::cout << "[trellis] wrote dummy GS arrays (K="<<K<<") to "<< out.out_dir <<"\n";
    };

    auto write_mesh = [&](){
      // Write minimal cube mesh as vertices.npy [8,3] and faces.npy [12,3] (triangles)
      const float V[8][3] = {{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
      const int F[12][3] = {{0,1,2},{0,2,3},{4,5,6},{4,6,7},{0,1,5},{0,5,4},{2,3,7},{2,7,6},{1,2,6},{1,6,5},{3,0,4},{3,4,7}};
      std::vector<float> verts(8*3);
      std::vector<float> faces(12*3);
      for (int i=0;i<8;++i){ verts[i*3+0]=V[i][0]; verts[i*3+1]=V[i][1]; verts[i*3+2]=V[i][2]; }
      for (int i=0;i<12;++i){ faces[i*3+0]=(float)F[i][0]; faces[i*3+1]=(float)F[i][1]; faces[i*3+2]=(float)F[i][2]; }
      trellis::write_npy_f32((fs::path(out.out_dir)/"mesh_vertices.npy").string(), verts.data(), verts.size(), {8,3});
      trellis::write_npy_f32((fs::path(out.out_dir)/"mesh_faces.npy").string(), faces.data(), faces.size(), {12,3});
      std::cout << "[trellis] wrote dummy mesh arrays to "<< out.out_dir <<"\n";
    };

    auto write_rf = [&](){
      // Simple factorization placeholder: U [64,16], V [16,64], bias [64]
      const int M=64, N=64, R=16;
      std::vector<float> U(M*R, 0.0f), V(R*N, 0.0f), b(M, 0.0f);
      // Fill with small deterministic values
      for (int i=0;i<M;++i) for (int r=0;r<R;++r) U[i*R+r] = 0.01f * (i + r);
      for (int r=0;r<R;++r) for (int j=0;j<N;++j) V[r*N+j] = 0.01f * (r + j);
      for (int i=0;i<M;++i) b[i] = 0.1f * i;
      trellis::write_npy_f32((fs::path(out.out_dir)/"rf_U.npy").string(), U.data(), U.size(), {M,R});
      trellis::write_npy_f32((fs::path(out.out_dir)/"rf_V.npy").string(), V.data(), V.size(), {R,N});
      trellis::write_npy_f32((fs::path(out.out_dir)/"rf_bias.npy").string(), b.data(), b.size(), {M});
      std::cout << "[trellis] wrote dummy RF arrays to "<< out.out_dir <<"\n";
    };

    if (out.target == "gs" || out.target == "all") {
      write_gs(512);
    }
    if (out.target == "mesh" || out.target == "all") {
      write_mesh();
    }
    if (out.target == "rf" || out.target == "all") {
      write_rf();
    }
  } catch (const std::exception& e) {
    std::cout << "[trellis] failed to write dummy output: " << e.what() << "\n";
  }
  return;
}

} // namespace trellis
