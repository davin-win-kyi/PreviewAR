from PIL import Image
from image_cropper import process_largest_object
import os

from PIL import Image
from image_cropper import process_largest_object
import os

def place_centered_object_width_only(
    image_path: str,
    real_width_inches: float,
    output_path: str,
    canvas_size: int = 1200,
    pixels_per_inch: float = 10.0,
):
    img = Image.open(image_path)

    # Compute new width from inches
    target_width_px = int(real_width_inches * pixels_per_inch)
    # Keep original height unchanged
    target_height_px = img.height

    # ⚠ This will distort horizontally if aspect ratio changes
    resized_img = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)

    # Center on canvas
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
    x_offset = (canvas_size - target_width_px) // 2
    y_offset = (canvas_size - target_height_px) // 2
    canvas.paste(resized_img, (x_offset, y_offset), resized_img)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    canvas.save(output_path)
    print(f"Saved centered (width-only scaled) image to: {output_path}")


def place_centered_object_height_only(
    image_path: str,
    real_height_inches: float,
    output_path: str,
    canvas_size: int = 1200,
    pixels_per_inch: float = 10.0,
):
    img = Image.open(image_path)

    # Compute new height from inches
    target_height_px = int(real_height_inches * pixels_per_inch)
    # Keep original width unchanged
    target_width_px = img.width

    # ⚠ This will distort vertically if aspect ratio changes
    resized_img = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)

    # Center on canvas
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
    x_offset = (canvas_size - target_width_px) // 2
    y_offset = (canvas_size - target_height_px) // 2
    canvas.paste(resized_img, (x_offset, y_offset), resized_img)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    canvas.save(output_path)
    print(f"Saved centered (height-only scaled) image to: {output_path}")


def main(
    real_width_inches: float,
    real_height_inches: float,
    image_path: str = "test.png",
    highlight_output_path: str = "output/test_highlighted.png",
    crop_output_path: str = "output/test_cropped.png",
    scaled_output_path: str = "output_scaled/scaled_couch.png",
):
    """
    Process an image by:
      1. Finding and cropping the largest object.
      2. Scaling it to a given real-world width.
      3. Then scaling it to a given real-world height.
    """
    bbox = process_largest_object(
        image_path=image_path,
        highlight_output_path=highlight_output_path,
        crop_output_path=crop_output_path,
    )

    if bbox is None:
        print("No object found; aborting.")
        return None

    # Only change width (e.g., couch width in inches)
    place_centered_object_width_only(
        image_path=crop_output_path,
        real_width_inches=real_width_inches,
        output_path=scaled_output_path,
    )

    # Only change height (e.g., couch height in inches)
    place_centered_object_height_only(
        image_path=scaled_output_path,
        real_height_inches=real_height_inches,
        output_path=scaled_output_path,
    )

    return scaled_output_path



if __name__ == "__main__":
    # Example usage:
    main(
        real_width_inches=80,
        real_height_inches=35,
        image_path="test.png",
    )
