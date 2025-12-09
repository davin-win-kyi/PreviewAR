from __future__ import annotations
from typing import Optional
import os
from PIL import Image

# 1) import your scaler
from scale_and_place import main as scale_single_image


# ============================================================
# Combine two (already scaled) images side-by-side
# ============================================================

def combine_side_by_side(
    left_image_path: str,
    right_image_path: str,
    output_path: str = "output_scaled/combined_side_by_side.png",
    padding_px: int = 20,
    background_rgba=(255, 255, 255, 255),
) -> str:
    """
    Combine two images side-by-side onto a single canvas.

    The images are vertically centered; we DO NOT rescale them here,
    so their relative pixel sizes (and thus physical scales) are preserved.
    """
    left = Image.open(left_image_path).convert("RGBA")
    right = Image.open(right_image_path).convert("RGBA")

    lw, lh = left.size
    rw, rh = right.size

    canvas_w = lw + padding_px + rw
    canvas_h = max(lh, rh)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), background_rgba)

    left_y = (canvas_h - lh) // 2
    right_y = (canvas_h - rh) // 2

    canvas.paste(left, (0, left_y), left)
    canvas.paste(right, (lw + padding_px, right_y), right)

    # Make sure directory exists if one is specified
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    canvas.save(output_path)

    return output_path


# ============================================================
# Nano Banana integration using your nano_banana_gemini()
# ============================================================

def call_nano_banana_object_transfer(
    combined_image_path: str,
    user_prompt: str,
    output_path: str = "output_scaled/nano_banana_result.png",
):
    """
    Wraps your nano_banana_gemini() call so that it:
      - Takes the combined side-by-side image
      - Uses a prompt that tells Gemini to place the object from the left
        image into the right image WITHOUT changing scale.
    """
    from nano_banana_testing import nano_banana_gemini  # your function

    full_prompt = (
        user_prompt.strip()
        + " Make sure to keep the objects at exactly the same scale as in the "
          "original images. Do not resize the objects, only reposition and "
          "blend them naturally in the scene."
    )

    return nano_banana_gemini(
        image_path=combined_image_path,
        prompt=full_prompt,
        output_path=output_path,
    )


# ============================================================
# Orchestrator: full pipeline using scale_and_place + nano banana
# ============================================================

def pipeline_place_object_from_img1_onto_img2_using_nano_banana(
    img1_path: str,
    img2_path: str,
    img1_width_inches: float,
    img1_height_inches: float,
    img2_width_inches: float,
    img2_height_inches: float,
    user_prompt: str,
    final_output_path: str = "output_scaled/final_nano_banana.png",
) -> str:
    """
    Full pipeline:

      1. Use scale_and_place.main to crop + scale image 1 and image 2 to
         user-provided real-world width/height.
      2. Combine them side-by-side (no further scaling).
      3. Call nano_banana_gemini on the combined image with a prompt that
         instructs Gemini to place the object from the left image into the
         right image, without changing the scale of any objects.

    Returns:
        Path to the final nano banana output image.
    """

    os.makedirs("output_scaled", exist_ok=True)

    # -------- Step 1: scale each image via scale_and_place.main --------
    # img1: table
    scaled1 = scale_single_image(
        real_width_inches=img1_width_inches,
        real_height_inches=img1_height_inches,
        image_path=img1_path,
        highlight_output_path="output/table_highlighted.png",
        crop_output_path="output/table_cropped.png",
        scaled_output_path="output_scaled/table.png",
    )

    # img2: plate (or whatever second object)
    scaled2 = scale_single_image(
        real_width_inches=img2_width_inches,
        real_height_inches=img2_height_inches,
        image_path=img2_path,
        highlight_output_path="output/plate_highlighted.png",
        crop_output_path="output/plate_cropped.png",
        scaled_output_path="output_scaled/plate.png",
    )

    if scaled1 is None or scaled2 is None:
        raise RuntimeError("Scaling failed for one of the images (no object found).")

    # -------- Step 2: combine side-by-side (preserves scale) --------
    combined_path = combine_side_by_side(
        left_image_path=scaled1,
        right_image_path=scaled2,
        output_path="output_scaled/nano_input_combined.png",
        padding_px=20,
    )

    # -------- Step 3: nano banana compositing --------
    result_path = call_nano_banana_object_transfer(
        combined_image_path=combined_path,
        user_prompt=user_prompt,
        output_path=final_output_path,
    )

    return result_path


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    prompt = (
        "Place one plate on the right onto each of the black `x` marks "
        "on the table on the left of the image only. Do not put" \
        " plates on locations not marked with an black `x`. Output this " \
        "as a new image with the items combined naturally . " \
        "Remove the x's from the original image and the plate to the right."
    )

    pipeline_place_object_from_img1_onto_img2_using_nano_banana(
        img1_path="table.png",
        img2_path="plate.jpg",
        img1_width_inches=80.0,
        img1_height_inches=35.0,
        img2_width_inches=15.0,
        img2_height_inches=15.0,
        user_prompt=prompt,
        final_output_path="output_scaled/final_nano_banana.png",
    )
