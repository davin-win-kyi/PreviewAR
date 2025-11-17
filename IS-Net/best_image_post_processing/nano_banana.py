#!/usr/bin/env python3
import os
import uuid
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

# Your earlier pipeline (the big script you posted with MaskGenerationRunner)
from mask_generation import MaskGenerationRunner  # adjust filename if needed

# -------------------------------------------------------------------
# Environment + dirs
# -------------------------------------------------------------------
load_dotenv()

UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


def format_object_list(other_objects):
    """
    Formats a list of strings into a human-readable "or" list.
    
    - [] -> ""
    - ['A'] -> "A"
    - ['A', 'B'] -> "A or B"
    - ['A', 'B', 'C'] -> "A, B, or C"
    """
    n = len(other_objects)
    
    if n == 0:
        # Case 0: Empty list
        return ""
    elif n == 1:
        # Case 1: One object
        # "object#1"
        return other_objects[0]
    elif n == 2:
        # Case 2: Two objects
        # "object#1 or object#2"
        return f"{other_objects[0]} or {other_objects[1]}"
        # Or, as a join:
        # return " or ".join(other_objects)
    else:
        # Case 3: Three or more objects
        # "object#1, object#2, or object#3"
        
        # Join all items except the last one with a comma
        all_but_last = ", ".join(other_objects[:-1])
        
        # Get the last item
        last_item = other_objects[-1]
        
        # Combine them with ", or "
        return f"{all_but_last}, or {last_item}"


# -------------------------------------------------------------------
# Prompt builder
# -------------------------------------------------------------------
def build_inpaint_prompt(target_class: str, classes_in_crop: List[str]) -> str:
    """
    Builds a structured inpainting prompt like:

    "can you fill the mask with the same texture and fabric as the rest of the <target_object>
     with only the <target_object> and no white gaps or holes in the <target_object>.
     You must not include the following objects in the image <other_object#1>, ..., <other_object#n>."
    """
    other_objects = sorted({
        c for c in classes_in_crop
        if c.lower() != target_class.lower()
    })

    if other_objects:
        others_str = format_object_list(other_objects=other_objects)
        prompt = (
            f"can you fill the mask with the same texture and fabric as the rest of the {target_class} "
            f"with only the {target_class} and no white gaps or holes in the {target_class}. "
            f"Also make sure to not include {others_str}."
        )
    else:
        prompt = (
            f"Can you fill the mask with the same texture and fabric as the rest of the {target_class} "
            f"with only the {target_class} and no white gaps or holes in the {target_class}. "
            f"There are no other objects that should be included."
        )

    return prompt


# -------------------------------------------------------------------
# Image utilities: combine original + mask into BW background mask
# -------------------------------------------------------------------
def combine_images_with_mask(
    original_path: str,
    mask_path: str,
    output_path: str,
    resize_mask: bool = True,
    white_threshold: int = 250,
) -> str:
    """
    Creates an image that:
      - Keeps ALL non-white pixels from the original image.
      - On original white background pixels:
          * if mask is white ( > 0 ) -> set to pure white (255,255,255)
          * if mask is black ( == 0 ) -> set to pure black (0,0,0)

    So:
      - The couch / target object (non-white) stays unchanged.
      - The white background becomes a black/white mask, indicating where to inpaint.

    Args:
        original_path: Path to original RGB/RGBA image with white background.
        mask_path: Path to mask image (white where you want the inpaint region).
        output_path: Where to save result.
        resize_mask: Resize mask to match original size if needed.
        white_threshold: Pixels above this on all channels are treated as “white”.

    Returns:
        output_path
    """
    try:
        # Load original as RGB
        original = Image.open(original_path).convert("RGB")
        orig_size = original.size

        # Load mask as grayscale
        mask = Image.open(mask_path).convert("L")
        if resize_mask and mask.size != orig_size:
            mask = mask.resize(orig_size, Image.NEAREST)
            print(f"[combine] Resized mask to {orig_size}")

        # Convert to numpy
        orig_np = np.array(original)           # (H, W, 3)
        mask_np = np.array(mask)               # (H, W)

        # Identify white background in original (within a threshold)
        white_bg = (
            (orig_np[:, :, 0] >= white_threshold) &
            (orig_np[:, :, 1] >= white_threshold) &
            (orig_np[:, :, 2] >= white_threshold)
        )

        # Mask “on” where mask image is non-zero (white)
        mask_on = mask_np > 0

        # Start with original image
        out = orig_np.copy()

        # On white background:
        #  - where mask_on -> white
        #  - where not mask_on -> black
        bg_and_mask = white_bg & mask_on
        bg_and_not_mask = white_bg & ~mask_on

        out[bg_and_mask] = [255, 255, 255]
        out[bg_and_not_mask] = [0, 0, 0]

        # Save
        out_img = Image.fromarray(out, mode="RGB")
        out_img.save(output_path)
        print(f"[combine] Combined original+mask image saved to {output_path}")
        return output_path

    except FileNotFoundError:
        raise FileNotFoundError("Error: One of the files was not found.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred in combine_images_with_mask: {e}")


# -------------------------------------------------------------------
# Gemini inpainting
# -------------------------------------------------------------------
def recontext_masked_area(
    combined_image_path: str,
    prompt: str,
    inpainted_output_path: str | None = None,
) -> Image.Image:
    """
    Sends the combined image + prompt to Gemini to in-paint the masked area.
    """
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set. Please export it in your environment.")

    UPLOADS_DIR = "uploads"
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # Create a single global client
    client = genai.Client(api_key=GOOGLE_API_KEY)

    full_prompt = prompt
    print("[gemini] prompt:", full_prompt)

    image = Image.open(combined_image_path)

    if inpainted_output_path is None:
        inpainted_output_path = combined_image_path

    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[full_prompt, image],
        config=GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            candidate_count=1,
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print("[gemini] text:", part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save(inpainted_output_path)
            print(f"[gemini] Inpainted image saved to {inpainted_output_path}")

    return image


def inpaint_image_with_mask(
    original_image_path: str,
    mask_image_path: str,
    prompt: str,
    combined_output_path: str | None = None,
    inpainted_output_path: str | None = None,
) -> tuple[str, str]:
    """
    1. Combines original + mask as described above (object kept, background -> BW mask).
    2. Sends that image + prompt to Gemini to in-paint the masked (white) area.

    Returns:
        (combined_path, inpainted_path)
    """
    if combined_output_path is None:
        combined_output_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}_combined.png")
    if inpainted_output_path is None:
        inpainted_output_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}_inpainted.png")

    print("[pipeline] Combining original image and mask...")
    combined_path = combine_images_with_mask(
        original_path=original_image_path,
        mask_path=mask_image_path,
        output_path=combined_output_path,
        resize_mask=True,
    )

    print("[pipeline] Sending combined image to Gemini for inpainting...")
    recontext_masked_area(combined_path, prompt, inpainted_output_path=inpainted_output_path)

    return combined_path, inpainted_output_path


# -------------------------------------------------------------------
# High-level: use original + manual mask, plus MaskGenerationRunner for prompt
# -------------------------------------------------------------------
def run_full_inpaint_with_manual_mask(
    original_image_path: str,
    mask_image_path: str,
    product_url: str,
    conf: float = 0.25,
    iou: float = 0.45,
    use_gpt: bool = True,
    adjustment_factor: int = -15,
) -> dict:
    """
    Full end-to-end:
      1) Run MaskGenerationRunner on the original image to get:
         - target_class
         - classes_in_crop
      2) Build the structured prompt from those classes.
      3) Combine the *original image* and a *user-provided mask*.
      4) Call Gemini to inpaint.
      5) Return paths + prompt.
    """
    # 1) Run your existing pipeline just to get target + other objects
    runner = MaskGenerationRunner()
    summary = runner.run(
        image_path=original_image_path,
        product_url=product_url,
        conf=conf,
        iou=iou,
        use_gpt=use_gpt,
        adjustment_factor=adjustment_factor,
    )

    target_class = summary.get("target_class")
    classes_in_crop = summary.get("classes_in_crop", [])

    if not target_class:
        raise RuntimeError("MaskGenerationRunner did not return a target_class.")

    print(f"[summary] target_class: {target_class}")
    print(f"[summary] classes_in_crop: {classes_in_crop}")

    # 2) Build prompt
    prompt = build_inpaint_prompt(target_class, classes_in_crop)
    print("[summary] inpaint prompt:", prompt)

    # 3–4) Combine + inpaint using your manual mask
    combined_output_path = os.path.join(UPLOADS_DIR, "combined_manual_mask.png")
    inpainted_output_path = os.path.join(UPLOADS_DIR, "inpainted_manual_mask.png")

    combined_path, inpainted_path = inpaint_image_with_mask(
        original_image_path=original_image_path,
        mask_image_path=mask_image_path,
        prompt=prompt,
        combined_output_path=combined_output_path,
        inpainted_output_path=inpainted_output_path,
    )

    return {
        "target_class": target_class,
        "classes_in_crop": classes_in_crop,
        "prompt": prompt,
        "combined_image_path": combined_path,
        "inpainted_image_path": inpainted_path,
    }


# -------------------------------------------------------------------
# Main: set variables and run
# -------------------------------------------------------------------
if __name__ == "__main__":
    # === SET THESE VARIABLES ===
    ORIGINAL_IMAGE_PATH = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/kept_with_white_bg.png"
    MASK_IMAGE_PATH = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/mask_non_overlapping.png"  # your own mask
    PRODUCT_URL = (
        "https://www.amazon.com/MODNEST-Modular-Sectional-Boneless-Assembly/dp/"
        "B0FCYC86TT/ref=sr_1_5_sspa?crid=3RIV4C6CTWYQA&dib=eyJ2IjoiMSJ9."
        "mCR5SjVr3IuoofQI97UxVmePO3nKmQbrqGkH6q7BhCWvXZfA2gaJDgyWsF100Jp3IznRSrEL8WKrwF2Xtlr-Q6YdVwugk_h-"
        "vhluo-EvhxJBqShb2gctTmjV71AXRyHOoPE6xC5K1iS8ITO3gdrhSf93HanYw7yk5iuIDU0gQFvfQLiPHo05ZX5PuYYk5As943eAeCxhe_d7i07UPtixVaCT_4yDty6lWukpMHvQmtssdgjG_"
        "zKvPqz8uqzZ5oAjIljfU3T1fj2lJKgKnqjlWxjA504G0RVwfRlQuUKr5bM.bAdATDJIW3OxMs6M16JiHRG9zAvYg7_B-SoRvPV4i5s&dib_tag=se&keywords=couch&qid=1762565138&"
        "sprefix=cou%2Caps%2C264&sr=8-5-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"
    )      # used by MaskGenerationRunner

    result = run_full_inpaint_with_manual_mask(
        original_image_path=ORIGINAL_IMAGE_PATH,
        mask_image_path=MASK_IMAGE_PATH,
        product_url=PRODUCT_URL,
        conf=0.25,
        iou=0.45,
        use_gpt=True,
        adjustment_factor=-15,
    )

    print("\n=== DONE ===")
    print("Target class:", result["target_class"])
    print("Classes in crop:", result["classes_in_crop"])
    print("Prompt used:\n", result["prompt"])
    print("Combined image:", result["combined_image_path"])
    print("Inpainted image:", result["inpainted_image_path"])

