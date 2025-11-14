import base64
import numpy as np
import cv2
from openai import OpenAI
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def check_and_resize_mask(input_image_path: str, mask_path: str, resized_mask_path: str):
    """
    Ensures the black/white mask matches the input image size.
    If not, it resizes it to match.
    """

    # Load input image size
    with Image.open(input_image_path) as img:
        img_w, img_h = img.size

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask not found or unreadable: {mask_path}")

    mask_h, mask_w = mask.shape

    print(f"Image size: {img_w} x {img_h}")
    print(f"Mask size:  {mask_w} x {mask_h}")

    if (img_w, img_h) != (mask_w, mask_h):
        print("❗ Mask size does not match image size. Resizing mask...")
        resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(resized_mask_path, resized)
        print(f"Saved resized mask to: {resized_mask_path}")
        return resized_mask_path
    
    print("✔ Mask size matches image size.")
    return mask_path


def create_transparent_mask(mask_input_path: str, mask_output_path: str):
    """
    Converts a black/white mask into a transparent PNG for DALL·E editing.
    - White (255) = keep original image (alpha=255)
    - Black (0)   = editable region (alpha=0)
    """

    mask = cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask file not found or unreadable: {mask_input_path}")

    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0:3] = 255  # irrelevant, alpha is what matters
    rgba[:, :, 3] = mask

    cv2.imwrite(mask_output_path, rgba)
    print(f"Saved transparent (RGBA) mask to: {mask_output_path}")


def run_dalle_edit(input_image_path, transparent_mask_path, prompt, output_path="edited.png"):
    print("Preparing image size...")

    # Load input image to get dimensions
    with Image.open(input_image_path) as img:
        width, height = img.size

    size_str = f"{width}x{height}"
    print("Using size:", size_str)

    print("Sending request to gpt-image-1...")

    result = client.images.edit(
        model="gpt-image-1",
        image=("input.png", open(input_image_path, "rb"), "image/png"),
        mask=("mask.png", open(transparent_mask_path, "rb"), "image/png"),
        prompt=prompt,
        size="auto",
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    with open(output_path, "wb") as f:
        f.write(image_bytes)

    print(f"Saved edited image to: {output_path}")


if __name__ == "__main__":

    input_image = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/kept_with_white_bg.png"
    blackwhite_mask = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/mask_non_overlapping.png"

    resized_mask = "mask_resized.png"
    transparent_mask = "mask_rgba.png"

    prompt = "Fill the missing couch area with realistic brown leather texture."

    # ✔ Step 1: Ensure mask is same size as image
    correct_mask_path = check_and_resize_mask(input_image, blackwhite_mask, resized_mask)

    # ✔ Step 2: Convert corrected mask to transparent PNG
    create_transparent_mask(correct_mask_path, transparent_mask)

    # ✔ Step 3: Run DALL·E inpainting / generative fill
    run_dalle_edit(input_image, transparent_mask, prompt)

