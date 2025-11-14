#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

WHITE_THRESH = 250  # 0..255, treat pixels >= this on all channels as white
USE_ALPHA_IF_PRESENT = True  # if mask has an alpha channel, use it directly


def load_image(path: str, flags=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def ensure_3ch(img: np.ndarray) -> np.ndarray:
    """Convert 1ch/4ch masks to 3ch BGR for uniform processing."""
    if img.ndim == 2:  # gray
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:  # BGRA -> BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img  # already BGR


def build_keep_mask(mask_img: np.ndarray) -> np.ndarray:
    """
    Returns a boolean 2D mask: True = keep pixel from source.
    Prefers alpha channel if present (USE_ALPHA_IF_PRESENT), otherwise white threshold.
    """
    # alpha path (if requested & present)
    if USE_ALPHA_IF_PRESENT and mask_img.ndim == 3 and mask_img.shape[2] == 4:
        alpha = mask_img[:, :, 3]
        return alpha > 0  # keep where any alpha > 0 (tweakable)

    # otherwise, use "white means keep" heuristic on 3-channel
    m3 = ensure_3ch(mask_img)
    keep = (
        (m3[:, :, 0] >= WHITE_THRESH)
        & (m3[:, :, 1] >= WHITE_THRESH)
        & (m3[:, :, 2] >= WHITE_THRESH)
    )
    return keep


def apply_white_background(src_bgr: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Keep src where keep_mask is True; else set to white."""
    if keep_mask.dtype != bool:
        keep_mask = keep_mask.astype(bool)
    keep3 = np.repeat(keep_mask[:, :, None], 3, axis=2)
    out = np.where(keep3, src_bgr, 255)
    return out


def main(original_image: str, mask_image: str, out_image: str):
    # Load images
    src = load_image(original_image, cv2.IMREAD_COLOR)      # BGR
    mask = load_image(mask_image, cv2.IMREAD_UNCHANGED)     # could be 1/3/4ch

    # Resize mask to source size if needed
    h, w = src.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        if mask.ndim == 2:
            interp = cv2.INTER_NEAREST
        else:
            ch = mask.shape[2]
            interp = cv2.INTER_NEAREST if ch in (1, 4) else cv2.INTER_LINEAR
        mask = cv2.resize(mask, (w, h), interpolation=interp)

    # Build keep mask (alpha preferred; otherwise white-threshold)
    keep = build_keep_mask(mask)

    # Compose output
    out = apply_white_background(src, keep)

    # Ensure output folder exists
    Path(out_image).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_image, out)
    print(f"Saved → {out_image}")


if __name__ == "__main__":
    # --- Set your paths here, then call main() ---
    original_image = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/crops/previewar_test#1_couch_001.jpg"
    mask_image     = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/grounded_sam_002_00.png"
    out_image      = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/kept_with_white_bg.png"

    main(original_image, mask_image, out_image)

