#!/usr/bin/env python3
"""
crop_white_masks_from_merged.py

For each object in merged alias JSON:
  - Use mask_polygon to create a white object on black background
  - Crop tightly around the polygon
  - Save each crop into a directory
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw


def _clip_point(x: int, y: int, W: int, H: int) -> Tuple[int, int]:
    """Clip a point (x, y) to image bounds [0, W-1] x [0, H-1]."""
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    return x, y


def _polygon_bbox(pts: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """Given a list of (x, y) points, return (min_x, min_y, max_x, max_y)."""
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def crop_white_masks_from_merged(
    image_path: str,
    merged_json_path: str,
    out_dir: str = "object_white_masks",
    pad: int = 4,
) -> list[str]:
    """
    For each object in merged_json:
      - If mask_polygon exists:
          * Build a white-on-black mask image just for that object.
          * Crop tightly around the polygon (with padding).
          * Save as PNG.

      - If no mask_polygon:
          * Fall back to bounding_box rectangle filled white on black.

    Returns list of output file paths.
    """
    # Open image just to get size
    base_im = Image.open(image_path).convert("RGB")
    W, H = base_im.size

    with open(merged_json_path, "r") as f:
        merged = json.load(f)

    alias = (merged.get("alias") or "object").strip() or "object"
    stem = Path(image_path).stem

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    objects = merged.get("objects", [])
    print(f"[white-mask] Found {len(objects)} object(s) in JSON for alias '{alias}'")

    for idx, obj in enumerate(objects):
        poly = obj.get("mask_polygon")
        bbox = obj.get("bounding_box") or [0, 0, 0, 0]

        # Make a blank black grayscale mask
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)

        pts: List[Tuple[int, int]] = []

        if poly:
            # Use polygon
            for p in poly:
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    continue
                x, y = p
                xi, yi = _clip_point(int(x), int(y), W, H)
                pts.append((xi, yi))

            if len(pts) >= 3:
                draw.polygon(pts, fill=255, outline=255)
            else:
                pts = []
        # If no polygon (or bad polygon), use bbox rectangle
        if not pts:
            x, y, w, h = bbox
            x1, y1 = _clip_point(int(x), int(y), W, H)
            x2, y2 = _clip_point(int(x + w), int(y + h), W, H)
            if x2 > x1 and y2 > y1:
                draw.rectangle([x1, y1, x2, y2], fill=255, outline=255)
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # Compute tight bbox from pts
        bbox_poly = _polygon_bbox(pts)
        if bbox_poly is None:
            continue
        x1, y1, x2, y2 = bbox_poly

        # Add padding and clip to image bounds
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad)
        y2 = min(H, y2 + pad)

        if x2 <= x1 or y2 <= y1:
            continue

        # At this point, mask has object in white (255) and background in black (0)
        # If you want 3-channel white/black instead of grayscale, convert here:
        mask_crop = mask.crop((x1, y1, x2, y2))
        mask_rgb = Image.merge("RGB", [mask_crop] * 3)  # white = [255,255,255], black = [0,0,0]

        obj_name = (obj.get("object_name") or alias).strip() or alias
        safe_name = obj_name.replace(" ", "_")
        out_path = out_dir_path / f"{stem}_{safe_name}_{idx:03d}_white_mask.png"
        mask_rgb.save(out_path)
        written.append(str(out_path))

    print(f"[white-mask] Saved {len(written)} mask crop(s) → {out_dir_path}")
    return written


if __name__ == "__main__":
    # === EDIT THESE ===
    IMAGE_PATH = "previewar_test#1.jpg"
    MERGED_JSON_PATH = "previewar_test#1_yolo11_o365_merged_alias.json"
    OUT_DIR = "object_white_masks"
    PADDING = 4
    # ==================

    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    if not Path(MERGED_JSON_PATH).exists():
        raise FileNotFoundError(f"Merged JSON not found: {MERGED_JSON_PATH}")

    crop_white_masks_from_merged(
        image_path=IMAGE_PATH,
        merged_json_path=MERGED_JSON_PATH,
        out_dir=OUT_DIR,
        pad=PADDING,
    )
