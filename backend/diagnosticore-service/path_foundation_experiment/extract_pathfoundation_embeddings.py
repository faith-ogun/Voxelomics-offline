#!/usr/bin/env python3
"""
Extract Path Foundation embeddings for TP53 tile manifests.

Uses the official Hugging Face loading path for Path Foundation:
`huggingface_hub.from_pretrained_keras("google/path-foundation")`

Outputs:
- <output-dir>/embeddings.npy
- <output-dir>/embedding_rows.csv
- <output-dir>/embedding_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Path Foundation tile embeddings")
    parser.add_argument(
        "--tile-manifest",
        default="output/tcga_brca_tp53_tiles_manifest_full_200.csv",
        help="Tile manifest with output_png, split, label, case/file identifiers",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-id",
        default="google/path-foundation",
        help="HF model id or local model path",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tiles", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--split-filter",
        default="",
        help="Optional comma-separated split filter, e.g. train,val,test",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not attempt network download; use local HF cache/files only",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional HF token for gated model access",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Patch resize for Path Foundation input",
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def build_embedding_array(outputs) -> np.ndarray:
    # The documented Path Foundation signature returns a dict with output_0.
    if isinstance(outputs, dict) and outputs:
        tensor = next(iter(outputs.values()))
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        tensor = outputs[0]
    else:
        tensor = outputs
    arr = np.asarray(tensor)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")
    return arr.astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.tile_manifest).resolve()
    rows = read_rows(manifest_path)

    split_filter = [s.strip() for s in args.split_filter.split(",") if s.strip()]
    if split_filter:
        allowed = set(split_filter)
        rows = [r for r in rows if r.get("split", "") in allowed]

    filtered = []
    missing = 0
    for r in rows:
        png = Path(r.get("output_png", ""))
        if png.exists():
            filtered.append(r)
        else:
            missing += 1
    rows = filtered
    if args.max_tiles > 0:
        rows = rows[: args.max_tiles]

    if not rows:
        raise RuntimeError("No valid tile rows after filtering.")

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "TensorFlow is required for google/path-foundation in this script. "
            "Use a Python 3.11/3.12 environment and install TensorFlow + huggingface_hub."
        ) from exc
    try:
        from huggingface_hub import from_pretrained_keras
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "huggingface_hub.from_pretrained_keras is unavailable in your installed version. "
            "Install compatible packages:\n"
            "  python -m pip install 'huggingface_hub>=0.26,<1.0' tf-keras"
        ) from exc

    print("[embed] loading Path Foundation Keras model")
    token = args.hf_token.strip() or None
    model = from_pretrained_keras(
        args.model_id,
        token=token,
        local_files_only=args.local_files_only,
    )
    infer = model.signatures["serving_default"] if hasattr(model, "signatures") else model

    emb_batches: List[np.ndarray] = []
    out_rows: List[Dict[str, str]] = []

    n_failed = 0
    idx_counter = 0
    pbar = tqdm(range(0, len(rows), args.batch_size), desc="Path Foundation embed", leave=False)
    for i in pbar:
        batch_rows = rows[i : i + args.batch_size]
        images = []
        kept_rows = []
        for r in batch_rows:
            try:
                img = Image.open(r["output_png"]).convert("RGB")
                images.append(img)
                kept_rows.append(r)
            except Exception:
                n_failed += 1

        if not images:
            continue

        batch = []
        for img in images:
            if img.size != (args.input_size, args.input_size):
                img = img.resize((args.input_size, args.input_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            batch.append(arr)
        batch_np = np.stack(batch, axis=0).astype(np.float32)
        outputs = infer(tf.constant(batch_np))
        emb = build_embedding_array(outputs)

        emb_batches.append(emb)
        for j, r in enumerate(kept_rows):
            out = dict(r)
            out["embedding_index"] = str(idx_counter + j)
            out_rows.append(out)
        idx_counter += emb.shape[0]

    if not emb_batches:
        raise RuntimeError("Embedding extraction produced zero batches.")

    embeddings = np.vstack(emb_batches).astype(np.float32)
    if embeddings.shape[0] != len(out_rows):
        raise RuntimeError(
            f"Embedding count mismatch: embeddings={embeddings.shape[0]}, rows={len(out_rows)}"
        )

    emb_path = out_dir / "embeddings.npy"
    rows_path = out_dir / "embedding_rows.csv"
    manifest_out = out_dir / "embedding_manifest.json"

    np.save(emb_path, embeddings)
    write_csv(rows_path, out_rows)

    by_split: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    for r in out_rows:
        by_split[r.get("split", "unknown")] = by_split.get(r.get("split", "unknown"), 0) + 1
        by_label[r.get("label", "unknown")] = by_label.get(r.get("label", "unknown"), 0) + 1

    summary = {
        "config": {
            "tile_manifest": str(manifest_path),
            "model_id": args.model_id,
            "batch_size": int(args.batch_size),
            "max_tiles": int(args.max_tiles),
            "split_filter": split_filter,
            "local_files_only": bool(args.local_files_only),
            "backend": "tensorflow_keras",
            "input_size": int(args.input_size),
        },
        "n_input_rows": len(read_rows(manifest_path)),
        "n_missing_png_rows": int(missing),
        "n_embedded_rows": int(len(out_rows)),
        "n_failed_image_loads": int(n_failed),
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "by_split": by_split,
        "by_label": by_label,
        "outputs": {
            "embeddings_npy": str(emb_path),
            "embedding_rows_csv": str(rows_path),
        },
    }
    manifest_out.write_text(json.dumps(sanitize_json(summary), indent=2), encoding="utf-8")

    print(f"[embed] wrote {emb_path}")
    print(f"[embed] wrote {rows_path}")
    print(f"[embed] wrote {manifest_out}")
    print(
        f"[embed] rows={len(out_rows)} dim={embeddings.shape[1]} missing_png={missing} failed_image_load={n_failed}"
    )


if __name__ == "__main__":
    main()
