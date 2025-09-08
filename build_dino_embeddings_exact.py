#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

import timm
HOME = Path.cwd()

OUT_ROOT = HOME / "embeddings"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def write_json(path: Path, data: Dict[str, Any]):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def resolve_preprocess_local():
    # Use local model config instead of downloading from timm
    size = (518, 518)  # From local config.json
    mean = (0.485, 0.456, 0.406)  # From local config.json
    std = (0.229, 0.224, 0.225)   # From local config.json
    interpolation = "bicubic"     # From local config.json
    transform = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC if interpolation=="bicubic" else T.InterpolationMode.BILINEAR),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    meta = {"size": size, "mean": mean, "std": std, "interpolation": interpolation}
    return transform, meta

def expected_patch_count(size_hw: Tuple[int,int], patch: int = 14) -> int:
    h, w = size_hw
    if h % patch != 0 or w % patch != 0:
        # many ViT pipelines do exact multiple of patch; warn if not.
        raise SystemExit(f"Input size {h}x{w} not divisible by patch {patch}")
    return (h // patch) * (w // patch)

@torch.no_grad()
def encode_images_dino_exact(images: List[Path], dino_model_path: str, device: str = "cuda"):
    ensure_dir(OUT_ROOT)
    model_name = "vit_large_patch14_reg4_dinov2.lvd142m"

    # Load timm DINOv2 reg model from local path
    model = timm.create_model(model_name, pretrained=False).to(device).eval()
    
    # Load weights from local model
    weights_path = Path(dino_model_path) / "model.safetensors"
    if weights_path.exists():
        import safetensors.torch
        state_dict = safetensors.torch.load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise SystemExit(f"Model weights not found at {weights_path}")

    # Discover register count - fixed approach
    num_registers = 4  # We know this model has 4 registers based on the config
    print(f"[info] Using model with {num_registers} register tokens")

    transform, pre_meta = resolve_preprocess_local()
    H, W = pre_meta["size"]
    exp_patches = expected_patch_count((H, W), patch=14)

    index_items = []
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        # Obtain token sequence prior to pooling/head
        feats = model.forward_features(x)
        if isinstance(feats, dict):
            tokens = feats.get("x", None)
            if tokens is None:
                # fallbacks commonly used by timm
                tokens = feats.get("x_norm", feats.get("x_prenorm", None))
                if tokens is None:
                    # Find any (B,N,D) tensor in dict as last resort
                    candidates = [v for v in feats.values() if isinstance(v, torch.Tensor) and v.dim()==3]
                    if not candidates:
                        raise RuntimeError("Could not find token sequence in forward_features output.")
                    tokens = candidates[-1]
        else:
            tokens = feats

        if tokens.dim() != 3:
            raise RuntimeError(f"Token tensor not (B,N,D): got {tuple(tokens.shape)}")

        B, N, D = tokens.shape
        # Assume CLS is first token. Remove it to get PATCH+REGISTER tokens.
        tokens_no_cls = tokens[:, 1:, :]
        N_no_cls = tokens_no_cls.shape[1]

        # Validate counts: N_no_cls should equal exp_patches + num_registers
        if N_no_cls != exp_patches + num_registers:
            raise RuntimeError(
                f"Token count mismatch: got N_no_cls={N_no_cls}, expected patches({exp_patches}) + registers({num_registers}) = {exp_patches + num_registers}. "
                "Check model id and preprocessing size."
            )

        # Save ONLY the exact tensor Trellis cross-attn expects (PATCH+REGISTER, no CLS)
        stem = img_path.stem
        out_npz = OUT_ROOT / f"{stem}.tokens.npz"
        out_json = OUT_ROOT / f"{stem}.json"
        out_uncond = OUT_ROOT / f"{stem}.uncond.npz"

        np_tokens = to_np(tokens_no_cls)  # (1, P+R, D)
        np_uncond = np.zeros_like(np_tokens)

        # Save
        np.savez_compressed(out_npz, tokens=np_tokens)
        np.savez_compressed(out_uncond, tokens=np_uncond)

        meta = {
            "source": str(img_path),
            "encoder": model_name,
            "device": device,
            "dtype": str(tokens_no_cls.dtype).replace("torch.", ""),
            "shape": list(tokens_no_cls.shape),
            "has_cls_token": False,
            "num_registers": int(num_registers),
            "patch_tokens": int(exp_patches),
            "total_tokens": int(N_no_cls),
            "preprocess": pre_meta,
            "contract": "tokens = PATCH+REGISTER (CLS removed). Compatible with Trellis cross-attn.",
        }
        write_json(out_json, meta)
        index_items.append({"image": str(img_path), "tokens": str(out_npz), "uncond": str(out_uncond), "meta": meta})
        print(f"[ok] {img_path} -> tokens {list(tokens_no_cls.shape)} (patch={exp_patches}, registers={num_registers})")

    index_path = OUT_ROOT / "index.json"
    index = {"items": index_items, "encoder": model_name, "input_size": [H, W], "registers": num_registers}
    write_json(index_path, index)
    print(f"[written] {index_path}")

def main():
    ap = argparse.ArgumentParser(description="Build *exact* DINOv2-reg embeddings for Trellis (PATCH+REGISTER, CLS removed).")
    ap.add_argument("--images", nargs="+", type=Path, required=True, help="Image file paths")
    ap.add_argument("--dino-model-path", type=str, default="models/vit_large_patch14_reg4_dinov2.lvd142m", help="Path to local DINO model directory")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    imgs = [p for p in args.images if p.exists()]
    if not imgs:
        raise SystemExit("No valid image paths")
    encode_images_dino_exact(imgs, args.dino_model_path, device=args.device)

if __name__ == "__main__":
    main()
