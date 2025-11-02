# pip install opencv-python numpy requests pillow

import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

def load_image_cv2_from_url(url, mode="color"):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = np.frombuffer(r.content, np.uint8)
    if mode == "gray":
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def ensure_binary_mask(mask_gray, invert_if_needed=True):
    # Threshold to 0/255
    m = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)[1]

    if invert_if_needed:
        # Heuristic: if most of the mask is white, we might be inverted.
        # We want WHITE = FILL REGION, BLACK = KEEP.
        white_ratio = (m > 127).mean()
        # If "almost everything" is white, try invert (assumes holes are smaller)
        if white_ratio > 0.85:
            m = cv2.bitwise_not(m)
    return m

def inpaint_from_urls(image_url, mask_url, out_path="inpainted_opencv.jpg",
                      inpaint_radius=3, dilate_kernel=(5,5), dilate_iters=1):
    # Load
    img  = load_image_cv2_from_url(image_url, mode="color")
    mask = load_image_cv2_from_url(mask_url,  mode="gray")

    # Make a clean binary mask (white=fill)
    mask = ensure_binary_mask(mask, invert_if_needed=True)

    # Optional: dilate to cover edges
    if dilate_kernel and dilate_iters > 0:
        kernel = np.ones(dilate_kernel, np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iters)

    # Inpaint
    result = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    cv2.imwrite(out_path, result)
    print("Saved:", out_path)


import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

def load_image_cv2_from_url(url, mode="color"):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = np.frombuffer(r.content, np.uint8)
    if mode == "gray":
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def ensure_binary_mask(mask_gray, invert_if_needed=True):
    # Threshold to 0/255
    m = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)[1]

    if invert_if_needed:
        # Heuristic: if most of the mask is white, we might be inverted.
        # We want WHITE = FILL REGION, BLACK = KEEP.
        white_ratio = (m > 127).mean()
        # If "almost everything" is white, try invert (assumes holes are smaller)
        if white_ratio > 0.85:
            m = cv2.bitwise_not(m)
    return m

def inpaint_from_urls(image_url, mask_url, out_path="inpainted_opencv.jpg",
                      inpaint_radius=3, dilate_kernel=(5,5), dilate_iters=1):
    # Load
    img  = load_image_cv2_from_url(image_url, mode="color")
    mask = load_image_cv2_from_url(mask_url,  mode="gray")

    # Make a clean binary mask (white=fill)
    mask = ensure_binary_mask(mask, invert_if_needed=True)

    # Optional: dilate to cover edges
    if dilate_kernel and dilate_iters > 0:
        kernel = np.ones(dilate_kernel, np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iters)

    # Inpaint
    result = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    cv2.imwrite(out_path, result)
    print("Saved:", out_path)

# Example use:
image_url = "https://m.media-amazon.com/images/I/81U1IoVP6YL._AC_SL1500_.jpg"   # <— put your room image URL here
mask_url  = "https://replicate.delivery/xezq/pTiqn9ep7xwkNqHfOaWrK8JKAj049WZRYQVe8hBkiso19UKrA/inverted_mask.jpg"
inpaint_from_urls(image_url, mask_url, out_path="inpainted_opencv.jpg")
