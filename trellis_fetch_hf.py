#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import os
from typing import Dict, Any, Optional
from huggingface_hub import snapshot_download, HfApi

def ensure_hf_token_note():
    if os.environ.get("HF_TOKEN"):
        return
    # Not an error—just a hint for private repos / faster downloads
    print("[note] You can set HF_TOKEN=<your token> for private repos or higher rate limits.")

def download_repo(repo_id: str, local_dir: Path) -> Path:
    """Download an HF repo using snapshot_download into local_dir/repo_name."""
    target = local_dir / repo_id.split("/")[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=target, local_dir_use_symlinks=False, tqdm_class=None)
    return target

def read_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def size_of(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0

def build_manifest_for_trellis(pipeline_dir: Path) -> Dict[str, Any]:
    pipe_file = pipeline_dir / "pipeline.json"
    if not pipe_file.exists():
        raise FileNotFoundError(f"{pipe_file} not found")
    pipeline = read_json(pipe_file)
    ckpts_dir = pipeline_dir / "ckpts"
    if not ckpts_dir.exists():
        raise FileNotFoundError(f"{ckpts_dir} missing")

    # collect component stems referenced anywhere in pipeline.json that exist in ckpts/*.json
    available = {p.stem for p in ckpts_dir.glob("*.json")}
    used = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            if o in available:
                used.add(o)

    walk(pipeline)

    components: Dict[str, Any] = {}
    for stem in sorted(used):
        cfg = ckpts_dir / f"{stem}.json"
        wt  = ckpts_dir / f"{stem}.safetensors"
        comp = {
            "name": stem,
            "config_path": str(cfg),
            "weights_path": str(wt),
            "weights_bytes": size_of(wt),
            "config": read_json(cfg) if cfg.exists() else {},
        }
        components[stem] = comp

    man = {
        "about": {
            "generated_by": "trellis_fetch_hf.py",
            "trellis_path": str(pipeline_dir),
        },
        "pipeline": pipeline,
        "components": components,
    }
    return man

def main():
    ap = argparse.ArgumentParser(description="Download Trellis models via Hugging Face Hub (no git-lfs) and build a merged manifest.json")
    ap.add_argument("--root", type=Path, default=Path("./models"), help="Root folder to place all repos")
    ap.add_argument("--pipeline", choices=["image-large", "text-base"], default="image-large")
    ap.add_argument("--dino-repo", type=str, default="timm/vit_large_patch14_reg4_dinov2.lvd142m",
                   help="DINOv2 ViT-L/14 with registers repo id (image pipeline)")
    ap.add_argument("--with-clip", action="store_true", help="Also fetch CLIP ViT-L/14 (useful for text pipeline)")
    args = ap.parse_args()

    ensure_hf_token_note()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    trellis_repo = "microsoft/TRELLIS-image-large" if args.pipeline == "image-large" else "microsoft/TRELLIS-text-base"
    trellis_dir = download_repo(trellis_repo, root)

    external = {"external_models": {}, "trellis_repo": trellis_repo, "trellis_path": str(trellis_dir)}

    if args.pipeline == "image-large":
        dino_dir = download_repo(args.dino_repo, root)
        external["external_models"]["image_cond_model"] = {"repo": args.dino_repo, "path": str(dino_dir)}

    if args.pipeline == "text-base" or args.with_clip:
        clip_repo = "openai/clip-vit-large-patch14"
        clip_dir = download_repo(clip_repo, root)
        external["external_models"]["text_cond_model"] = {"repo": clip_repo, "path": str(clip_dir)}

    # Build Trellis manifest
    manifest = build_manifest_for_trellis(trellis_dir)
    (trellis_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[written] {trellis_dir / 'manifest.json'}")

    # Write external models info
    (root / "external_models.json").write_text(json.dumps(external, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[written] {root / 'external_models.json'}")

    print("\nDone. Parse manifest.json from your custom runtime, mmap *.safetensors, and launch your CUDA graph.")

if __name__ == "__main__":
    main()
