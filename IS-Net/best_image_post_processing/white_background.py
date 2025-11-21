#!/usr/bin/env python3
import os
import uuid
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig

# -------------------------------------------------------------------
# Environment + dirs
# -------------------------------------------------------------------
load_dotenv()  # expects GEMINI_API_KEY in your .env or environment

UPLOADS_DIR = "best_image_post_processing/uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


def nano_banana_black_bg_to_white(
    original_image_path: str,
    output_path: str | None = None,
) -> str:
    """
    Uses Nano Banana (Gemini 2.5 Flash Image) to turn a black background
    into a clean white background while keeping the main object unchanged.
    """

    # Create client (Gemini Developer API).
    client = genai.Client()  # GEMINI_API_KEY picked up from env

    # Load image
    image = Image.open(original_image_path)

    # If no explicit output path is provided, write into uploads/ with a UUID
    if output_path is None:
        base_name = f"post_processing_image.png"
        output_path = os.path.join(UPLOADS_DIR, base_name)

    # Prompt: tell Nano Banana exactly what to do
    prompt = (
        "Replace the entire black background in this photo with a clean, solid white "
        "studio background. Keep the subject, lighting, and non-background details "
        "exactly the same. Only change the black background to pure white."
    )

    print("[nano-banana] Sending image for black→white background edit...")
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",  # Nano Banana
        contents=[prompt, image],
        config=GenerateContentConfig(
            response_modalities=["IMAGE"],
            candidate_count=1,
        ),
    )

    # Extract the edited image from the response
    edited_img = None
    candidate = response.candidates[0]
    for part in candidate.content.parts:
        if getattr(part, "inline_data", None) is not None:
            edited_img = Image.open(BytesIO(part.inline_data.data))
            break

    if edited_img is None:
        raise RuntimeError("Nano Banana did not return an edited image.")

    edited_img.save(output_path)
    print(f"[nano-banana] Edited image saved to: {output_path}")

    return output_path


# -------------------------------------------------------------------
# Main: set variables and run
# -------------------------------------------------------------------
if __name__ == "__main__":
    # === SET THIS TO YOUR INPUT IMAGE ===
    ORIGINAL_IMAGE_PATH = (
       "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/uploads/inpainted_manual_mask.png"
    )

    out_path = nano_banana_black_bg_to_white(
        original_image_path=ORIGINAL_IMAGE_PATH,
        output_path=None,  # or set an explicit path
    )

    print("\n=== DONE ===")
    print("Output image with white background:", out_path)

