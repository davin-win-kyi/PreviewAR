# from openai import OpenAI
# from PIL import Image
# import os
# from dotenv import load_dotenv

# load_dotenv()
# client = OpenAI()

# # ---- Paths ----
# orig_image_path = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/kept_with_white_bg.png"
# orig_mask_path   = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/mask_non_overlapping.png"

# resized_image_path = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/kept_with_white_bg_1024.png"
# resized_mask_path  = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/mask_non_overlapping_1024.png"

# TARGET_SIZE = (1024, 1024)  # could also be (512, 512) or (256, 256)


# def prepare_image_and_mask():
#     # Image: keep color, ensure PNG, resize with high-quality filter
#     img = Image.open(orig_image_path).convert("RGBA")
#     img = img.resize(TARGET_SIZE, Image.LANCZOS)
#     img.save(resized_image_path, format="PNG")

#     # Mask: keep it single-channel; white = edit, black = keep
#     mask = Image.open(orig_mask_path).convert("L")
#     mask = mask.resize(TARGET_SIZE, Image.NEAREST)  # NEAREST to keep hard edges
#     mask.save(resized_mask_path, format="PNG")


# def main():
#     prepare_image_and_mask()

#     result = client.images.edit(
#         model="dall-e-2",
#         image=open(resized_image_path, "rb"),
#         mask=open(resized_mask_path, "rb"),
#         prompt="Fill the mask with the same texture/material as the rest of the couch",
#         n=1,
#         size="1024x1024",  # <-- must be an explicit allowed size
#     )

#     print(result.data[0].url)


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np

IMAGE_PATH = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/output/kept_with_white_bg.png"       # original couch image
MASK_PATH  = "/Users/davinwinkyi/PreviewAR-V2/PreviewAR/best_image_post_processing/mask_non_overlapping.png"     # white = fill, black = keep

# ---- 1. Load image and mask ----
img  = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if img is None or mask is None:
    raise RuntimeError("Could not load image or mask.")

# Resize mask to image size if needed
if (mask.shape[0] != img.shape[0]) or (mask.shape[1] != img.shape[1]):
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# Binarize mask: 255 = region to FILL, 0 = keep original
_, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

h, w = img.shape[:2]

# ---- 2. Choose patch size & sampling stride ----
PATCH_H = PATCH_W = 128  # adjust depending on image size / couch scale
STRIDE  = PATCH_H // 4   # how densely we search

# ---- 3. Search for a “good” texture patch automatically ----
best_score = None
best_xy = None

# We only consider patches that are fully OUTSIDE the masked region
for y in range(0, h - PATCH_H, STRIDE):
    for x in range(0, w - PATCH_W, STRIDE):
        patch_mask = mask_bin[y:y+PATCH_H, x:x+PATCH_W]

        # skip if this patch overlaps masked region
        if np.any(patch_mask == 255):
            continue

        patch = img[y:y+PATCH_H, x:x+PATCH_W]

        # Convert to gray for scoring
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # Heuristic score:
        #   - low variance (uniform texture)
        #   - low edge energy (fewer seams/edges)
        variance = np.var(gray)

        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edge_energy = np.mean(edges**2)

        # You can tune weights here
        score = variance + 0.5 * edge_energy

        if best_score is None or score < best_score:
            best_score = score
            best_xy = (x, y)

if best_xy is None:
    raise RuntimeError("Could not find any fully unmasked patch. Try smaller PATCH_H/PATCH_W.")

x0, y0 = (700,700)
print(f"Chosen patch top-left: ({x0}, {y0}), score={best_score}")

patch = img[y0:y0+PATCH_H, x0:x0+PATCH_H]  # square patch for simplicity

# ---- 4. Tile the chosen patch over the whole image ----
tile_rows = int(np.ceil(h / PATCH_H))
tile_cols = int(np.ceil(w / PATCH_H))
tiled = np.tile(patch, (tile_rows, tile_cols, 1))
tiled = tiled[:h, :w]

# ---- 5. Use tiled patch ONLY where mask is white ----
result = img.copy()
result[mask_bin == 255] = tiled[mask_bin == 255]

cv2.imwrite("couch_auto_texture_filled.png", result)
print("Saved couch_auto_texture_filled.png")

