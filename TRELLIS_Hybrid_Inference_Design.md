# TRELLIS Hybrid Inference Design (Python launcher + C/CUDA runtime)

This plan adapts the repository to a hybrid execution model that:

- Uses Python for file I/O and a single launch call.
- Runs all compute/memory-intensive work in C/CUDA (one entrypoint). 
- Minimizes CPU/GPU memory usage while maximizing throughput.
- Targets TRELLIS inference only (encoder excluded).

---

## 1) Objectives

- One Python→C call per inference (no per-layer/per-step crossings).
- Python: parse configs, mmap/load weights and embeddings, save outputs.
- C/CUDA: validate, upload, run full sampler + models, return results.
- Precision policy tuned for minimal memory + high throughput (FP16 storage with FP32 accumulation).

---

## 2) Division of Responsibilities

- Python
  - Parse `models/TRELLIS-image-large/pipeline.json` and any `ckpts/*.json`.
  - Load embeddings `embeddings/*.npz` (NumPy arrays) and weights `ckpts/*.safetensors` (mmap, zero-copy host).
  - Build flat manifests for weights and embeddings (pointers, shapes, dtypes) and a `RunSpec` with sampler/model options.
  - Call a single C entrypoint and then save outputs (e.g., `.npy/.npz`, `.ply`).

- C/CUDA
  - Validate manifests, create device pools/streams.
  - Upload FP16/FP32 weights as-is; convert only when required inside kernels.
  - Implement full pipeline:
    - SS flow + Euler sampler + SS decoder.
    - SLat flow + normalization + chosen decoder(s) (GS/RF/Mesh).
  - Optional CUDA Graph capture to reduce launch overhead.
  - Stage outputs back to host buffers provided by Python.

---

## 3) Single-call C API (C ABI)

A single function exported from a shared library (Linux: `libtrellis_infer.so`). Python calls via `ctypes`/`cffi`.

- Function (illustrative signature):

```
int trellis_run(
  const TrellisRunSpec* spec,
  const TrellisWeightTable* weights,
  const TrellisEmbeddings* emb,
  TrellisOutputs* outs,
  TrellisError* err
);
```

- Inputs
  - TrellisRunSpec: steps per stage, `sigma_min/max`, `cfg_strength`, `cfg_interval`, `rescale_t`, targets (`gs|rf|mesh|all`), flags (`low_memory`, `use_kv_cache`, `use_graphs`, `deterministic`).
  - TrellisWeightTable: N entries with name/id, dtype (F16/F32), shape, host pointer, byte size. Python keeps buffers alive for call duration (mmap).
  - TrellisEmbeddings: pointers/shapes for `tokens` and `uncond` (commonly F32 on disk). Optional flag to permit device-side F32→F16 cast on upload.

- Outputs
  - TrellisOutputs: preallocated host buffers for SS grid and selected decoders, or null to let C allocate and return pointers/sizes. Python saves to disk after return.

- Errors
  - Return non-zero on failure and fill `err->message`. No exceptions across the FFI boundary.

---

## 4) Precision and Memory Policy

- Weights
  - Keep native dtype (mostly FP16; some FP32). Do not upcast to FP32 on host or device.
  - On device, compute with FP16 inputs using FP32 accumulation (cuBLAS/CUTLASS). Keep norms/small tensors FP32 if needed for stability.

- Activations/Latents
  - Store in FP16; use FP32 accumulators in GEMMs and reductions.
  - Euler updates: compute `x + Δt·v` in FP32 temporary, cast result to FP16 for storage.

- Norms & Softmax
  - Compute in FP32 internally; parameters may be stored in FP16. Output FP16.

- Embeddings
  - Host is F32 (from `.npy`). Optionally convert to FP16 on device upload and keep FP16 thereafter.

- KV cache
  - Default: recompute (no KV) to cap VRAM. Expose flag to enable KV only for SLat when memory permits.

- Determinism/Perf
  - Disable TF32 unless requested. Use CUDA Graphs to cut launch overhead. Stream-ordered memory pools to reduce alloc/free churn.

---

## 5) Runtime Execution Flow (inside C)

1. Validate manifests (names, shapes, dtypes). Cross-check against sidecars if provided.
2. Setup device (streams, cuBLAS handles, memory pools).
3. Upload embeddings and selected component weights for current stage.
4. Build/capture CUDA Graph for a full sampler step (optionally cond+uncond in one capture).
5. Run SS flow sampling for `steps_ss`. Decode SS to grid and (optionally) export.
6. Release SS stage weights; load SLat stage weights.
7. Run SLat flow sampling for `steps_slat`. Apply `slat_normalization`. Run requested decoder(s).
8. Copy outputs to host buffers and return. Free temporary device allocations.

---

## 6) File Structure (proposed additions on top of `csrc/`)

Current (`csrc/`):

- `app/` CLI (dev tool)
- `kernels/` CUDA stubs
- `pipeline/` orchestrator, config, bindings
- `runtime/` device, tensor, safetensors, npz, json
- `samplers/` flow euler sampler (stub)

Additions and adjustments:

- `csrc/capi/`
  - `trellis_c_api.hpp` (C ABI structs/enums, exported function declarations)
  - `trellis_c_api.cpp` (thin wrapper: parse `TrellisRunSpec/Weights/Embeddings/Outputs`, call core pipeline)

- `csrc/models/` (to implement)
  - `attention.{hpp,cpp}` (self/cross with QK RMSNorm)
  - `mlp.{hpp,cpp}`
  - `dit_block.{hpp,cpp}`
  - `swin3d.{hpp,cpp}` (window partition + shift)
  - `conv3d.{hpp,cpp}` (SS decoder blocks)
  - `ss_decoder.{hpp,cpp}`
  - `slat_decoders.{hpp,cpp}` (GS/RF/Mesh)

- `csrc/kernels/`
  - Implementations for norms, softmax, attention QK, window ops, conv3d/upsampling, marching cubes

- `csrc/samplers/`
  - `flow_euler.{hpp,cpp}` full implementation (CFG interval + schedules)

- Optional dev utilities (kept out of hot path):
  - Minimal `.ply` writer for GS/Mesh export (if writing from C is desired for testing; production prefers Python save).

---

## 7) Python-side Caller (conceptual)

- Parse `pipeline.json` to choose component names and sampler params.
- Load embeddings (`np.load`), weights (`safetensors` mmap), sidecars (optional validate only).
- Build C ABI structs using `ctypes`/`cffi`:
  - Fill weight entries with raw pointers and metadata; retain references to base arrays to pin memory during call.
  - Preallocate output buffers or pass nulls to let C allocate.
- `ctypes.CDLL(…).trellis_run(...)`
- On success, save outputs (NumPy save, write PLY/OBJ, etc.).

---

## 8) Build and Packaging

- Build shared library target in `csrc/CMakeLists.txt`:
  - `add_library(trellis_infer SHARED …)` and export `trellis_run` with `extern "C"`.
  - Link `CUDA::cudart` and cuBLAS/CUTLASS as added.
  - Optionally keep `trellis_infer` CLI for local testing.

- Python
  - Load `.so` with `ctypes` in `run_trellis.py` (no Python extensions needed).

---

## 9) Development Milestones

1) API & Loader Integration
- Define C ABI structs and `trellis_run` in `csrc/capi/`.
- Python caller script builds manifests and invokes once per inference.

2) Core Math Baseline
- cuBLAS GEMM wrappers; FP16 w/ FP32 accum. Norm and softmax kernels (FP32 internal).
- Implement MLP and attention paths.

3) SS Path
- Implement DiT blocks and Euler sampler (CFG interval). Verify shapes and run a few steps. 
- Implement SS decoder (conv3d). Export SS grid.

4) SLat Path
- Implement windowed Swin-3D attention (partition/shift). Euler sampler. `slat_normalization` in FP32.
- Implement GS decoder first (validate 32×14 unpack). Optionally RF/Mesh next.

5) Performance & Memory
- CUDA Graph capture, memory pools, recompute vs KV cache flag, optional BF16.

6) Packaging & Validation
- Determinism baseline, small numeric parity checks on block outputs.
- End-to-end smoke test with provided embeddings.

---

## 10) Risks and Mitigations

- Swin-3D windowing specifics: derive shift pattern and partition ordering from weight names and configs; add tests.
- Mixed precision corner cases: keep norms and softmax in FP32; fallback to FP32 for sensitive small layers if needed.
- Memory pressure in SLat: default recompute; provide KV cache only when memory allows.
- API drift: lock C ABI early; extend via versioned structs if needed.

---

## 11) Output Policy

- C fills host buffers; Python saves artifacts.
- Grids: `.npy/.npz`.
- GS: arrays for `_xyz [K,3]`, `_features_dc [K,3]`, `_opacity [K,1]`, `_scaling [K,3]`, `_rotation [K,4]`.
- RF: factor matrices/parameters.
- Mesh: vertices/faces/colors; Python converts to `.ply`/`.obj`.

---

## 12) Memory Accounting (VRAM and Host RAM)

- **Weights (device-resident):**
  - Keep only the current stage resident. SS ~1.0–1.2 GB; SLat ~1.0–1.2 GB.
  - Load SS, run, free; then load SLat. Avoid holding both stages concurrently.
- **Embeddings and Latents (device):**
  - Tokens/uncond uploaded as FP16 to save VRAM (a few MB total for B=1).
  - SS latent [1, 4096, 1024] FP16 ≈ 8 MB; SLat latent [1, 32768, 1024] FP16 ≈ 64 MB.
- **Activations and Intermediates (device):**
  - Attention: Q/K/V projections; tile to avoid full 32k×1024 buffers. Use FlashAttention-style kernels to avoid storing logits.
  - MLP hidden (ratio 4 → 4096): tile tokens (e.g., 2k–4k) so the 4096 hidden slab is never fully resident.
- **Softmax/Logits scratch (device):**
  - SS (global) requires tiling/FA-style kernels to avoid 4096×4096 FP32 matrices.
  - SLat uses Swin-3D windows (512 tokens/window); process small window batches to cap scratch to tens of MB.
- **KV cache (device, optional):**
  - SLat per layer ≈ 128 MB (FP16 K+V); ×24 layers ≈ ~3 GB (×2 with CFG concurrency). Disabled by default in low-memory mode.
- **CFG duplication (device):**
  - Sequential cond/uncond halves activation residency versus concurrent execution.
- **Scratch/workspace/overhead (device):**
  - cuBLAS/CUTLASS workspaces, partition buffers, marching cubes, CUDA context/fragmentation: budget ~0.3–0.7 GB.
- **Host RAM:**
  - mmap `.safetensors` (zero-copy, relies on OS page cache), NumPy arrays for embeddings/outputs, optional pinned staging buffers.

---

## 13) RTX 2060 Super (6 GB) Preset

- **Goals:** Keep peak VRAM < ~6 GB while maintaining good throughput.
- **Preset configuration:**
  - dtype_policy: FP16 storage with FP32 accumulations (no host/device upcasting of whole tensors).
  - embeddings_upload_dtype: FP16 (device-side cast on upload).
  - cfg_mode: sequential (run cond and uncond separately per step).
  - kv_cache: false (recompute attention to save ~3–6 GB).
  - attn_algo: flash_windowed (online softmax, no full logits; Swin-3D windows).
  - attn_window_batch: small (1–2 windows concurrently) to cap scratch.
  - mlp_token_chunk: 2k–4k tokens per chunk (project back immediately; discard hidden tile).
  - per_stage_weight_residency: true (SS then SLat; free between).
  - decoder_order: single-target first; others sequential to limit concurrent residency.
  - graphs: true (optional; reduces launch overhead, neutral to peak VRAM).
  - deterministic: true (disable TF32 unless explicitly enabled).
- **Execution strategy:**
  - SS stage: tiled/FA-style attention to avoid global logits; run and free SS weights.
  - SLat stage: Swin-3D windows with small batches; tiled QKV and MLP; recompute (no KV cache); sequential CFG.
- **Expected peaks:**
  - SS: ~1.3–1.6 GB (weights + latents + modest scratch/overhead).
  - SLat: ~3.0–4.5 GB (weights + latent + tiled QKV/MLP + scratch/overhead).
  - Headroom preserved under 6 GB for decoders and runtime variance.
- **Trade-offs:**
  - Disabling KV cache and sequential CFG slightly reduce throughput but dramatically lower peak VRAM.
  - FlashAttention-style kernels and window tiling preserve high utilization with low scratch.

---

## 14) Summary

This hybrid design keeps Python in charge of lightweight I/O and orchestration while concentrating all compute and memory-intensive logic in a single C/CUDA entrypoint. It preserves minimal memory usage (FP16 storage everywhere feasible) and high throughput (FP32 accumulation where needed), and it aligns with the TRELLIS pipeline as captured in `pipeline.json` and ckpt sidecars.
