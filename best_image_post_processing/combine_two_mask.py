#!/usr/bin/env python3
"""
xor_two_masks.py

White (255) where mask A and mask B do NOT overlap (exactly one is white),
Black (0) everywhere else (background + overlap).
"""

from pathlib import Path
from PIL import Image
import numpy as np


def load_mask_bool(path: str):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return arr > 0, img.size  # (bool mask, (W, H))


def xor_two_masks(
    mask_a_path: str,
    mask_b_path: str,
    out_path: str = "xor_mask.png",
) -> str:
    # Load A
    mask_a_bool, size_a = load_mask_bool(mask_a_path)
    W, H = size_a

    # Load B and resize to match A if needed
    img_b = Image.open(mask_b_path).convert("L")
    if img_b.size != size_a:
        img_b = img_b.resize(size_a, resample=Image.NEAREST)
    arr_b = np.array(img_b)
    mask_b_bool = arr_b > 0

    # XOR: True where exactly one mask is True
    xor_mask = np.logical_xor(mask_a_bool, mask_b_bool)

    # Build output: white where xor_mask = True, black otherwise
    combined = np.zeros((H, W), dtype=np.uint8)
    combined[xor_mask] = 255

    out_img = Image.fromarray(combined, mode="L")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    print(f"[xor] Saved XOR mask → {out_path}")
    return out_path


if __name__ == "__main__":
    # ---- EDIT THESE ----
    MASK_A_PATH = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/object_white_masks/previewar_test#1_Couch_000_white_mask.png"
    MASK_B_PATH = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/grounded_sam_002_00.png"
    OUT_PATH    = "mask_non_overlapping.png"
    # --------------------

    if not Path(MASK_A_PATH).exists():
        raise FileNotFoundError(MASK_A_PATH)
    if not Path(MASK_B_PATH).exists():
        raise FileNotFoundError(MASK_B_PATH)

    xor_two_masks(MASK_A_PATH, MASK_B_PATH, OUT_PATH)
