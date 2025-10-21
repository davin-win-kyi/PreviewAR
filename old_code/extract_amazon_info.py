from __future__ import annotations
import json
import re
import sys
from typing import Any, Dict, List, Optional


# ----------------- Text utils -----------------
def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# ----------------- Images -----------------
def _extract_main_image(product: Dict[str, Any]) -> Optional[str]:
    main = product.get("main_image") or {}
    if isinstance(main, dict):
        link = main.get("link")
        if link:
            return str(link)
    return None

def _extract_images(product: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    imgs = product.get("images") or []
    if isinstance(imgs, list):
        urls.extend(str(x.get("link")) for x in imgs if isinstance(x, dict) and x.get("link"))

    if not urls and product.get("images_flat"):
        urls.extend(u.strip() for u in str(product["images_flat"]).split(",") if u.strip())

    # Ensure main image is first if present
    main = _extract_main_image(product)
    if main:
        urls = [main] + urls

    return _unique(urls)


# ----------------- Dimensions -----------------
def _pull_dimension_strings(product: Dict[str, Any]) -> List[str]:
    """Collect likely dimension strings in priority order."""
    candidates: List[str] = []

    # 1) specifications entries that contain 'dimensions' in the name
    specs = product.get("specifications") or []
    if isinstance(specs, list):
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("name", ""))
            value = str(spec.get("value", ""))
            if name and value and re.search(r"dimensions?", name, flags=re.IGNORECASE):
                candidates.append(_normalize_ws(value))

    # 2) product.dimensions (often already "D x W x H")
    pdims = product.get("dimensions")
    if isinstance(pdims, str) and pdims.strip():
        candidates.append(_normalize_ws(pdims))

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def _parse_triplet(raw: str) -> Dict[str, Optional[float]]:
    """
    Parse strings like:
      69.7"D x 108"W x 37.8"H
      L 69.7" x W 108" x H 37.8"
      Depth 69.7 in x Width 108 in x Height 37.8 in
    Returns inches; maps D/Depth and L/Length -> 'length' in final output.
    """
    res: Dict[str, Optional[float]] = {
        "length_in": None, "width_in": None, "height_in": None, "raw": _normalize_ws(raw)
    }

    # split on x / ×
    parts = re.split(r"\s*[x×]\s*", raw, flags=re.IGNORECASE)
    for idx, part in enumerate(parts[:3]):  # only the first three chunks matter
        m = re.search(
            r'(?:(?P<label1>\b[LWDH]\b|\bLength\b|\bWidth\b|\bDepth\b|\bHeight\b)\s*[:\-]?\s*)?'
            r'(?P<val>\d+(?:\.\d+)?)\s*(?:"|in|inch|inches)?\s*'
            r'(?P<label2>\b[LWDH]\b|\bLength\b|\bWidth\b|\bDepth\b|\bHeight\b)?',
            part, flags=re.IGNORECASE
        )
        if not m:
            continue

        val = float(m.group("val"))
        label = (m.group("label1") or m.group("label2") or "").lower()

        if label in ("l", "length", "d", "depth"):
            key = "length_in"
        elif label in ("w", "width"):
            key = "width_in"
        elif label in ("h", "height"):
            key = "height_in"
        else:
            # If unlabeled, infer by position assuming L x W x H
            key = ("length_in", "width_in", "height_in")[idx]

        if res[key] is None:
            res[key] = val

    # Fallback for compact "69.7 108 37.8"
    if any(res[k] is None for k in ("length_in", "width_in", "height_in")):
        nums = re.findall(r"\d+(?:\.\d+)?", raw)
        if len(nums) >= 3:
            if res["length_in"] is None: res["length_in"] = float(nums[0])
            if res["width_in"]  is None: res["width_in"]  = float(nums[1])
            if res["height_in"] is None: res["height_in"] = float(nums[2])

    # Round for cleanliness
    for k in ("length_in", "width_in", "height_in"):
        if isinstance(res[k], (int, float)):
            res[k] = round(float(res[k]), 2)

    return res


# ----------------- Public API -----------------
def extract_media_and_hwl(rainforest_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
    {
      "asin": "...",
      "main_image": "https://...",
      "images": ["https://...", ...],
      "units": "in",
      "length": 69.7,
      "width": 108.0,
      "height": 37.8,
      "source_raw": "69.7\"D x 108\"W x 37.8\"H"
    }
    """
    product = rainforest_json.get("product") or {}
    asin = product.get("asin") or rainforest_json.get("request_parameters", {}).get("asin")

    # Images
    main_image = _extract_main_image(product)
    images = _extract_images(product)

    # Dimensions (pick the first good candidate)
    dim_strings = _pull_dimension_strings(product)
    parsed = {"length_in": None, "width_in": None, "height_in": None, "raw": None}
    for s in dim_strings:
        cand = _parse_triplet(s)
        if any(cand[k] is not None for k in ("length_in", "width_in", "height_in")):
            parsed = cand
            break

    return {
        "asin": asin,
        "main_image": main_image,
        "images": images,
        "units": "in",
        "length": parsed["length_in"],
        "width": parsed["width_in"],
        "height": parsed["height_in"],
        "source_raw": parsed["raw"],
    }


# ----------------- CLI -----------------
if __name__ == "__main__":
    data = json.load(open(sys.argv[1], "r", encoding="utf-8")) if len(sys.argv) > 1 else json.load(sys.stdin)
    result = extract_media_and_hwl(data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
