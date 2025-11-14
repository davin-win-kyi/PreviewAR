import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image, ImageOps, ImageDraw, ImageFont

import numpy as np

# -------- settings --------
# IMPORTANT: use a SEGMENTATION model (e.g. yolo11n-seg.pt), not plain detection
REPO_ID   = "Ultralytics/YOLO11" # or your own repo
WEIGHT_FN = "yolo11n-seg.pt"             # <-- must be a *-seg model

# -------- model loading (cached) --------
_MODEL = None

def load_model():
    global _MODEL
    if _MODEL is None:
        weight_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHT_FN)
        _MODEL = YOLO(weight_path)
        n_cls = len(_MODEL.names)
        if n_cls != 365:
            print(f"[warn] Loaded names has {n_cls} classes (expected 365). "
                  f"Make sure your weights are Objects365-trained.")
    return _MODEL

# -------- JSON-returning method --------
def get_objects_json(image_path: str, conf: float = 0.25, iou: float = 0.45, max_det: int = 300) -> dict:
    """
    Returns:
      {
        "objects": [
          {
            "object_name": str,
            "bounding_box": [x,y,w,h],   # pixels
            "confidence": float,
            "mask_polygon": [[x0,y0], [x1,y1], ...] or null
          },
          ...
        ]
      }

    - bounding_box: top-left (x,y) + width/height in pixels.
    - mask_polygon: list of (x,y) points in pixel coordinates (if a seg model is used).
                    Will be None if no mask is available for that detection.
    """
    model = load_model()
    res = model(image_path, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
    names = model.names

    objects = []

    xyxy   = res.boxes.xyxy.cpu().numpy()   # (N, 4)
    cls    = res.boxes.cls.cpu().numpy()    # (N,)
    confs  = res.boxes.conf.cpu().numpy()   # (N,)

    # Masks (only present for *-seg models)
    if res.masks is not None and hasattr(res.masks, "xy"):
        mask_polys = []
        for poly in res.masks.xy:
            # poly can be a torch.Tensor or a numpy.ndarray depending on version
            if hasattr(poly, "cpu"):  # torch tensor
                poly_np = poly.cpu().numpy()
            else:                     # already numpy
                poly_np = np.asarray(poly)
            mask_polys.append(poly_np.tolist())
    else:
        mask_polys = [None] * len(xyxy)

    for (x1, y1, x2, y2), c, p, poly in zip(xyxy, cls, confs, mask_polys):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        label = names[int(c)]

        objects.append({
            "object_name": label,
            "bounding_box": [x1, y1, w, h],
            "confidence": float(p),
            "mask_polygon": poly,  # may be None if no mask
        })

    return {"objects": objects}

# -------- drawing (optional) --------
def draw_boxes_and_masks_pil(image_path: str, result: dict, out_path: str):
    """
    Draws bounding boxes and (if present) polygon masks on the image.
    """
    im = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    W, H = im.size

    # Use RGBA so we can alpha-blend masks
    im = im.convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw_boxes = ImageDraw.Draw(im)
    draw_masks = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for obj in result.get("objects", []):
        name = obj["object_name"]
        x, y, w, h = obj["bounding_box"]
        poly = obj.get("mask_polygon")

        # clip bbox
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))

        # draw bbox
        draw_boxes.rectangle([x, y, x + w, y + h], outline="red", width=3)

        # label
        label = f"{name} {obj.get('confidence', 0.0):.2f}"
        l, t, r, b = draw_boxes.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - t
        top = y - th - 4 if y - th - 4 >= 0 else y + 2
        draw_boxes.rectangle([x, top, x + tw + 6, top + th + 4], fill="red")
        draw_boxes.text((x + 3, top + 2), label, fill="white", font=font)

        # draw mask polygon if available
        if poly:
            # poly is [[x0,y0], [x1,y1], ...]
            # we clip points to image bounds for safety
            clipped = []
            for px, py in poly:
                px = max(0, min(int(px), W - 1))
                py = max(0, min(int(py), H - 1))
                clipped.append((px, py))

            if len(clipped) >= 3:
                # semi-transparent green mask
                draw_masks.polygon(clipped, fill=(0, 255, 0, 80), outline=(0, 255, 0, 180))

    # alpha-compose mask overlay on top of original
    im = Image.alpha_composite(im, overlay)
    im = im.convert("RGB")  # back to RGB for saving as jpg/png
    im.save(out_path)
    return out_path

# -------- script entry (example) --------
if __name__ == "__main__":
    IMAGE_PATH = "previewar_test#1.jpg"
    OUT_JSON   = "previewar_test#1_yolo11_o365_seg.json"
    OUT_IMAGE  = "previewar_test#1_yolo11_o365_seg.jpg"

    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(Path(IMAGE_PATH).resolve())

    result = get_objects_json(IMAGE_PATH, conf=0.25, iou=0.45)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON → {OUT_JSON}")

    out_img = draw_boxes_and_masks_pil(IMAGE_PATH, result, OUT_IMAGE)
    print(f"Saved annotated image → {out_img}")

    # quick sanity check
    print("Loaded class count:", len(load_model().names))
