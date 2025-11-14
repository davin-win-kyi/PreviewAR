# pip install diffusers transformers accelerate safetensors pillow torch --upgrade
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageOps
import numpy as np
import torch

img_path = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/kept_with_white_bg.png"

# 1) Load your image
image = Image.open(img_path).convert("RGB")
W, H = image.size

# 2) Build a full-white mask (whole image editable).
#    If you want extra safety, make a *soft* mask: light gray (e.g., 35–60) everywhere,
#    and paint brighter values (e.g., 220–255) roughly over the gap zone.
use_soft_mask = True
if use_soft_mask:
    base = np.full((H, W), 40, dtype=np.uint8)    # very light gray -> "edit a little"
    # (Optional) brighten a rough region where gaps are (example box):
    # base[y0:y1, x0:x1] = 255
    mask = Image.fromarray(base, mode="L")
else:
    mask = Image.new("L", (W, H), color=255)      # pure white (edit everywhere)

# 3) SDXL inpainting pipeline
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

prompt = (
    "repair only damaged or blank areas of the couch by continuing the same dark ribbed fabric; "
    "match rib direction, texture and lighting; keep all other regions unchanged"
)
neg = (
    "blurry, texture mismatch, color shift, artifacts, changes to background, changes to pillows, "
    "extra patterns, plastic shine, text, watermark"
)

# Key knobs when using a full mask:
# - strength ↓ keeps more of the original (try 0.15–0.35)
# - guidance_scale moderate (5–6) to avoid drifting from the source
# - fewer steps also reduces drift
out = pipe(
    prompt=prompt,
    negative_prompt=neg,
    image=image,
    mask_image=mask,
    num_inference_steps=20,
    guidance_scale=5.5,
    strength=0.25,   # <-- very important when full mask is used
).images[0]

out.save("couch_inpaint_fullmask.png")
print("Saved couch_inpaint_fullmask.png")
