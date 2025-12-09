from rembg import remove
from PIL import Image
import numpy as np
import cv2
import os

def process_largest_object(image_path, highlight_output_path=None, crop_output_path=None):
    # 1. Load original image
    input_image = Image.open(image_path)
    
    # 2. Remove background to isolate objects
    # We keep this as a PIL Image (no_bg_image) to easily crop later
    no_bg_image = remove(input_image) 
    
    # 3. Convert to numpy array to find shapes
    img_array = np.array(no_bg_image)
    
    # Extract the Alpha channel (transparency)
    alpha = img_array[:, :, 3]
    
    # 4. Find Contours
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No objects found!")
        return None

    # 5. Sort contours by Area (Largest to Smallest)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0]
    
    # 6. Get Bounding Box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # --- NEW: CROP & SAVE BLOCK ---
    if crop_output_path:
        crop_dir = os.path.dirname(crop_output_path)
        if crop_dir:  # avoid "" for current dir
            os.makedirs(crop_dir, exist_ok=True)

        # PIL crop syntax is (left, top, right, bottom)
        # We crop the 'no_bg_image' so the result has a transparent background
        cropped_img = no_bg_image.crop((x, y, x + w, y + h))
        cropped_img.save(crop_output_path)
        print(f"Cropped object saved to: {crop_output_path}")
    # ------------------------------

    # --- VISUALIZATION BLOCK ---
    if highlight_output_path:
        highlight_dir = os.path.dirname(highlight_output_path)
        if highlight_dir:
            os.makedirs(highlight_dir, exist_ok=True)
            
        # Convert PIL (RGB) to OpenCV (BGR) for drawing
        opencv_img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        
        # Draw Green Rectangle
        cv2.rectangle(opencv_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Save the highlighted full image
        cv2.imwrite(highlight_output_path, opencv_img)
        print(f"Bounding box visualization saved to: {highlight_output_path}")
    # ---------------------------

    return (x, y, w, h)

# --- Run it ---
bbox = process_largest_object(
    image_path='test.png', 
    highlight_output_path='output/test_highlighted.png', # Saves the full image with a green box
    crop_output_path='output/test_cropped.png'           # Saves just the object (transparent bg)
)

if bbox:
    print(f"Largest Object Box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")