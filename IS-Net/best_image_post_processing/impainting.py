#!/usr/bin/env python3
"""
sdxl_inpaint_fill.py

Takes:
  - an input image
  - a binary mask image (white = region to modify, black = keep)
  - a text prompt

and uses SDXL inpainting to fill the masked region according to the prompt.
Requires: diffusers, torch, pillow, accelerate, safetensors, transformers.
"""

from pathlib import Path

from diffusers import AutoPipelineForInpainting
from PIL import Image
import torch


def load_rgb(path: str) -> Image.Image:
    """Load an image as RGB."""
    img = Image.open(path).convert("RGB")
    return img


def load_mask(path: str, size: tuple[int, int]) -> Image.Image:
    """
    Load mask as single-channel (L), resize to `size` if needed.

    Convention: white (255) = area to modify, black (0) = keep.
    """
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return m


def build_pipeline(model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"):
    """
    Build an SDXL inpainting pipeline, using GPU if available.
    """
    if torch.cuda.is_available():
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
    else:
        # CPU fallback (slower, but works)
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
    pipe.enable_attention_slicing()
    return pipe


def inpaint_image(
    image_path: str,
    mask_path: str,
    prompt: str,
    out_path: str,
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    num_steps: int = 40,
    guidance_scale: float = 7.0,
    strength: float = 0.99,
) -> str:
    """
    Run SDXL inpainting on (image, mask, prompt) and save the result.

    Args:
        image_path: path to original image.
        mask_path: path to binary mask (white = modify).
        prompt: text prompt describing what should fill the masked area.
        out_path: where to save the output image.
        model_id: HF model id for the inpainting pipeline.
        num_steps: diffusion steps.
        guidance_scale: classifier-free guidance (higher = more prompt-driven).
        strength: how strongly to overwrite the masked area (0–1).

    Returns:
        The output path.
    """
    # Load data
    img = load_rgb(image_path)
    mask = load_mask(mask_path, img.size)

    # Load pipeline
    pipe = build_pipeline(model_id)

    # Run inpainting
    result = pipe(
        prompt=prompt,
        image=img,
        mask_image=mask,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        strength=strength,
    ).images[0]

    # Save result
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"[inpaint] Saved inpainted image → {out_path}")
    return out_path


if __name__ == "__main__":
    # --------- EDIT THESE VARIABLES ---------
    IMAGE_PATH = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/kept_with_white_bg.png"          # your original image
    MASK_PATH  = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/mask_non_overlapping.png"     # white where you want to change (e.g., blanket holes)
    PROMPT     = (
        "fill the masked region with the same texture as the rest of the couch"
    )
    OUT_PATH   = "output_couch_inpainted.png"
    # optionally tweak these:
    NUM_STEPS = 40
    GUIDANCE  = 7.0
    STRENGTH  = 0.99
    MODEL_ID  = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    # ---------------------------------------

    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    if not Path(MASK_PATH).exists():
        raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

    inpaint_image(
        image_path=IMAGE_PATH,
        mask_path=MASK_PATH,
        prompt=PROMPT,
        out_path=OUT_PATH,
        model_id=MODEL_ID,
        num_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        strength=STRENGTH,
    )

