#!/usr/bin/env python3
import os
from io import BytesIO
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

# Load GOOGLE_API_KEY from .env or environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Please export it or put it in a .env file.")

# Create a single Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)


def nano_banana_gemini(
    image_path: str,
    prompt: str,
    output_path: str = "nano_banana_gemini_output.png",
) -> str:
    """
    Takes an input image and a text prompt, and uses Gemini to generate
    a new image conditioned on both.

    Args:
        image_path: Path to the input image.
        prompt: Text describing how to modify / reinterpret the image.
        output_path: Where to save the generated image.

    Returns:
        output_path (string)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the input image
    image = Image.open(image_path).convert("RGB")

    # Call Gemini with [prompt, image]
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[prompt, image],
        config=GenerateContentConfig(
            response_modalities=["IMAGE"],  # we only need the image back
            candidate_count=1,
        ),
    )

    # Find the image part in the response
    gen_image = None
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            gen_image = Image.open(BytesIO(part.inline_data.data))
            break

    if gen_image is None:
        raise RuntimeError("Gemini did not return an image in the response.")

    # Ensure the output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the generated image
    gen_image.save(output_path)
    print(f"[nano_banana_gemini] Saved generated image to: {output_path}")
    return output_path


if __name__ == "__main__":
    # ===== CHANGE THESE FOR YOUR TEST =====
    IMAGE_PATH = "/Users/davinwinkyi/nano_banana_testing/test2.png"
    PROMPT = (
        "Make this couch a dark blue velvet sectional in a clean, modern living room. "
        "Remove blankets and extra objects around it."
    )
    OUTPUT_PATH = "output/nano_banana_gemini_result.png"

    nano_banana_gemini(
        image_path=IMAGE_PATH,
        prompt=PROMPT,
        output_path=OUTPUT_PATH,
    )
