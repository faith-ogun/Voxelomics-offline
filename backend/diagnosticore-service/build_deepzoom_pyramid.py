#!/usr/bin/env python3
"""
Builds an OpenSeadragon-compatible DeepZoom pyramid from a WSI slide.

Example:
python build_deepzoom_pyramid.py \
  --wsi-path data/gdc_wsi/<file_id>/<slide>.svs \
  --out-dir output/deepzoom \
  --slug TCGA-A1-A0SP
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DeepZoom pyramid for one WSI slide.")
    parser.add_argument("--wsi-path", required=True, help="Path to source .svs/.tif slide.")
    parser.add_argument("--out-dir", default="output/deepzoom", help="Directory for .dzi and tile pyramid.")
    parser.add_argument(
        "--slug",
        default="",
        help="Output slug. Writes <slug>.dzi and <slug>_files/. Defaults to case prefix from filename.",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="DeepZoom tile size.")
    parser.add_argument("--overlap", type=int, default=1, help="DeepZoom overlap in pixels.")
    parser.add_argument(
        "--format",
        default="jpeg",
        choices=["jpeg", "png"],
        help="Tile image format.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=78, help="JPEG quality (1-95).")
    parser.add_argument(
        "--limit-bounds",
        action="store_true",
        help="If set, trims non-content boundaries in compatible slide formats.",
    )
    return parser.parse_args()


def infer_slug(wsi_path: Path) -> str:
    # TCGA-A1-A0SP-01A-01-BS1... -> TCGA-A1-A0SP
    tokens = wsi_path.stem.split("-")
    if len(tokens) >= 3 and tokens[0].upper() == "TCGA":
        return "-".join(tokens[:3]).upper()
    return wsi_path.stem


def main() -> None:
    args = parse_args()
    wsi_path = Path(args.wsi_path).resolve()
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI slide not found: {wsi_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = (args.slug or infer_slug(wsi_path)).strip()
    if not slug:
        raise ValueError("Unable to resolve output slug.")

    try:
        import openslide
        from openslide import deepzoom
    except Exception as exc:
        raise RuntimeError(
            "Missing OpenSlide runtime. Install openslide-python and the OpenSlide system library."
        ) from exc

    try:
        from PIL import Image  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Missing Pillow runtime.") from exc

    slide = openslide.OpenSlide(str(wsi_path))
    dz = deepzoom.DeepZoomGenerator(
        slide,
        tile_size=max(64, int(args.tile_size)),
        overlap=max(0, int(args.overlap)),
        limit_bounds=bool(args.limit_bounds),
    )

    dzi_path = out_dir / f"{slug}.dzi"
    tile_root = out_dir / f"{slug}_files"
    tile_root.mkdir(parents=True, exist_ok=True)

    # Write descriptor.
    dzi_path.write_text(dz.get_dzi(args.format), encoding="utf-8")

    # Estimate tile count for progress.
    total_tiles = 0
    for level in range(dz.level_count):
        cols, rows = dz.level_tiles[level]
        total_tiles += cols * rows

    print(f"Slide: {wsi_path}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"DeepZoom levels: {dz.level_count}")
    print(f"Estimated tiles: {total_tiles}")
    print(f"Output DZI: {dzi_path}")
    print(f"Output tiles: {tile_root}")

    written = 0
    fmt = args.format.lower()
    save_fmt = "JPEG" if fmt == "jpeg" else "PNG"
    ext = fmt
    jpeg_quality = min(95, max(1, int(args.jpeg_quality)))

    for level in range(dz.level_count):
        cols, rows = dz.level_tiles[level]
        level_dir = tile_root / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        for row in range(rows):
            for col in range(cols):
                tile = dz.get_tile(level, (col, row)).convert("RGB")
                out_path = level_dir / f"{col}_{row}.{ext}"
                if save_fmt == "JPEG":
                    tile.save(out_path, save_fmt, quality=jpeg_quality, optimize=True)
                else:
                    tile.save(out_path, save_fmt, optimize=True)
                written += 1

        pct = (written / total_tiles) * 100 if total_tiles else 100.0
        print(
            f"Level {level + 1}/{dz.level_count} complete "
            f"({written}/{total_tiles} tiles, {pct:.1f}%)."
        )

    print("DeepZoom pyramid build complete.")
    print(f"DZI URL hint: /mdt/<case_id>/diagnosticore/deepzoom.dzi")


if __name__ == "__main__":
    main()
