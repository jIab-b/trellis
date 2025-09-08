#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import struct
import hashlib
from typing import Dict, Any, List

def read_safetensors_header(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"{path}: not a valid safetensors file (short header length)")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header = f.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path}: not a valid safetensors file (short header)")
        meta = json.loads(header.decode("utf-8"))
    return meta

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def inventory(root: Path, pattern: str, do_hash: bool) -> Dict[str, Any]:
    files: List[Path] = sorted(root.rglob(pattern))
    index = {"root": str(root), "files": []}
    for p in files:
        try:
            meta = read_safetensors_header(p)
            tensors = []
            for name, spec in meta.items():
                if name == "__metadata__":
                    continue
                tensors.append({
                    "name": name,
                    "dtype": spec.get("dtype"),
                    "shape": spec.get("shape"),
                    "data_offsets": spec.get("data_offsets"),
                })
            out = {
                "path": str(p),
                "num_tensors": len(tensors),
                "tensors": tensors,
                "metadata": meta.get("__metadata__", {}),
            }
            if do_hash:
                out["sha256"] = sha256_file(p)
            # write sidecar
            sidecar = p.with_suffix(p.suffix + ".shapes.json")
            sidecar.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            index["files"].append({"path": str(p), "sidecar": str(sidecar), "num_tensors": len(tensors)})
            print(f"[ok] {p.name}: {len(tensors)} tensors -> {sidecar.name}")
        except Exception as e:
            print(f"[warn] {p}: {e}")
    return index

def main():
    ap = argparse.ArgumentParser(description="Inventory safetensors headers and emit shapes JSON files (no framework required).")
    ap.add_argument("--root", type=Path, required=True, help="Root folder to scan")
    ap.add_argument("--pattern", type=str, default="*.safetensors", help="Glob pattern under root (default: *.safetensors)")
    ap.add_argument("--hash", action="store_true", help="Also compute sha256 for each file")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    idx = inventory(root, args.pattern, args.hash)

    index_path = root / "safetensors_index.json"
    index_path.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[written] {index_path}")

if __name__ == "__main__":
    main()
