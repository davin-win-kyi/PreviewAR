import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image, ImageOps, ImageDraw, ImageFont

# -------- settings --------
REPO_ID   = "NRtred/yolo11n_object365"     # community O365 weights repo (example)
WEIGHT_FN = "yolo11n_object365.pt"         # swap to your own if needed

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
          {"object_name": str, "bounding_box": [x,y,w,h], "confidence": float},
          ...
        ]
      }
    Bounding boxes are in pixels with [x,y,w,h] (top-left + width/height).
    """
    model = load_model()
    res = model(image_path, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
    names = model.names

    objects = []
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls  = res.boxes.cls.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        label = names[int(c)]
        objects.append({
            "object_name": label,
            "bounding_box": [x1, y1, w, h],
            "confidence": float(p),
        })

    return {"objects": objects}

# -------- drawing (optional) --------
def draw_boxes_pil(image_path: str, result: dict, out_path: str):
    im = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for obj in result.get("objects", []):
        name = obj["object_name"]
        x, y, w, h = obj["bounding_box"]

        # clip
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))

        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        label = f"{name} {obj.get('confidence', 0.0):.2f}"
        l, t, r, b = draw.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - t
        top = y - th - 4 if y - th - 4 >= 0 else y + 2
        draw.rectangle([x, top, x + tw + 6, top + th + 4], fill="red")
        draw.text((x + 3, top + 2), label, fill="white", font=font)

    im.save(out_path)
    return out_path

# -------- script entry (example) --------
if __name__ == "__main__":
    IMAGE_PATH = "previewar_test#1.jpg"
    OUT_JSON   = "previewar_test#1_yolo11_o365.json"
    OUT_IMAGE  = "previewar_test#1_yolo11_o365.jpg"

    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(Path(IMAGE_PATH).resolve())

    result = get_objects_json(IMAGE_PATH, conf=0.25, iou=0.45)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON → {OUT_JSON}")

    out_img = draw_boxes_pil(IMAGE_PATH, result, OUT_IMAGE)
    print(f"Saved annotated image → {out_img}")

    # quick sanity check
    from ultralytics.utils import ASSETS
    print("Loaded class count:", len(load_model().names))

