# product_cropping_pipeline.py

import os
import re
import json
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import cv2
from openai import OpenAI

# Bbox-only YOLO (e.g. Objects365 detector)
import object_detection as od

# Segmentation YOLO (yolo11n-seg)
import object_detection_segmented as ods

import extract_url_info as eui 

from dotenv import load_dotenv

load_dotenv()


# ---------- small helpers ----------

def strip_code_fences(s: str) -> str:
    """
    Remove ``` or ```json fences around a JSON string, if present.
    """
    s = s.strip()
    # remove opening fence like ```json\n or ```\n  (language is optional)
    s = re.sub(r'^\s*```[\w-]*\s*\n', '', s, count=1, flags=re.IGNORECASE)
    # remove closing fence ```
    s = re.sub(r'\n\s*```\s*$', '', s, count=1)
    return s


def _bbox_xyxy_from_xywh(b: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert [x, y, w, h] -> (x1, y1, x2, y2).
    """
    x, y, w, h = b
    return x, y, x + w, y + h


def _bbox_iou_xyxy(b1: List[float], b2: List[float]) -> float:
    """
    IoU between two boxes given as [x, y, w, h].
    """
    x1a, y1a, x2a, y2a = _bbox_xyxy_from_xywh(b1)
    x1b, y1b, x2b, y2b = _bbox_xyxy_from_xywh(b2)

    # intersection
    xi1 = max(x1a, x1b)
    yi1 = max(yi1, y1b) if (yi1 := max(y1a, y1b)) else max(y1a, y1b)  # just to avoid lint complaints
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h

    # areas
    area_a = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
    area_b = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)

    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


class ProductCropper:
    """
    End-to-end:
      URL -> product aliases -> choose YOLO class -> crop all detections of that class.

    Also provides a method to merge:
      - bbox-only JSON (e.g. Objects365 detector; object_detection.py)
      - seg JSON (e.g. YOLO11 segmentation model; object_detection_segmented.py)
    into a per-alias JSON containing bounding_box + mask_polygon.
    """

    def __init__(self, gpt_model: str = "gpt-5"):
        self.client = OpenAI()
        self.gpt_model = gpt_model

    # ---------- public orchestrator ----------
    def run(
        self,
        image_path: str,
        product_url: str,
        out_dir: str = "crops",
        conf: float = 0.25,
        iou: float = 0.45,
        use_gpt: bool = True,
    ) -> Dict:
        """
        Returns a summary dict containing:
          {
            "profile": {...},
            "target_class": str,
            "num_crops_saved": int,
            "saved_files": [paths...],
            "detections": <raw detections dict from od.get_objects_json>,
            "out_dir": str,
          }
        """
        # 1) Get product profile from your URL extractor
        profile = self._get_product_profile(product_url)

        # 2) Detections (bbox-only YOLO)
        det = od.get_objects_json(image_path, conf=conf, iou=iou)

        # 3) Choose target class from bbox model's class space
        class_names = list(od.load_model().names.values())
        target_class = (
            self._map_product_to_yolo_class_gpt(profile, class_names)
            if use_gpt else
            self._map_product_to_yolo_class_fuzzy(profile, class_names)
        )

        # 4) Crop all detections of that class
        saved_files = self._crop_target_objects(image_path, det, target_class, out_dir)

        return {
            "profile": profile,
            "target_class": target_class,
            "num_crops_saved": len(saved_files),
            "saved_files": saved_files,
            "detections": det,
            "out_dir": out_dir,
        }

    # ---------- step 1: product profile ----------
    def _get_product_profile(self, url: str) -> Dict:
        """
        Uses extract_url_info.extract_with_gpt5(url) to get:
          {
            "company_name": str,
            "product_name": [str, ...]
          }
        """
        data = eui.extract_with_gpt5(url)
        print("[extract_url_info] raw:", data)

        # Normalize shape defensively
        company = (data.get("company_name") or "").strip()
        names = data.get("product_name") or []
        if isinstance(names, str):
            names = [names]
        names = [s.strip() for s in names if isinstance(s, str) and s.strip()]
        return {"company_name": company, "product_name": names}

    # ---------- step 3A: GPT mapping ----------
    def _map_product_to_yolo_class_gpt(self, profile: Dict, class_names: List[str]) -> str:
        """
        Ask GPT to choose a single class name from class_names that best matches product aliases.
        Returns an exact string from class_names (with fuzzy fallback).
        """
        company = profile.get("company_name", "")
        aliases = profile.get("product_name", [])
        class_list_str = ", ".join(class_names)

        prompt = (
            "Map a product to one YOLO class name from the provided list.\n"
            'Return ONLY JSON: {"best_match_class": string}, where the value is EXACTLY one of the class names.\n'
            f"Company: {company}\n"
            f"Product aliases: {aliases}\n\n"
            "Candidate class names:\n"
            f"{class_list_str}\n\n"
            "Rules:\n"
            "- Choose a single best class used by typical object-detection datasets.\n"
            "- If multiple are close, pick the most generic household/retail term.\n"
            "- If none is perfect, choose the closest plausible generic class.\n"
        )

        resp = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": prompt}],
        )

        print("[GPT class mapping raw]:", resp.choices[0].message.content)

        content = strip_code_fences(resp.choices[0].message.content)
        data = json.loads(content)
        candidate = (data.get("best_match_class") or "").strip()

        # ensure exact membership; try case-insensitive and fuzzy as fallback
        if candidate in class_names:
            return candidate

        lower_map = {c.lower(): c for c in class_names}
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

        best = difflib.get_close_matches(candidate, class_names, n=1, cutoff=0.0)
        return best[0] if best else class_names[0]

    # ---------- step 3B: Fuzzy-only mapping (no GPT) ----------
    def _map_product_to_yolo_class_fuzzy(self, profile: Dict, class_names: List[str]) -> str:
        """
        No-LLM mapping: take aliases and fuzzy-match to class_names; return best.
        """
        aliases = [a for a in profile.get("product_name", []) if a]
        if not aliases:
            return class_names[0]

        # Try exact / case-insensitive first
        lower_map = {c.lower(): c for c in class_names}
        for a in aliases:
            if a in class_names:
                return a
            if a.lower() in lower_map:
                return lower_map[a.lower()]

        # Fuzzy: try all aliases, keep the best global match
        best_name, best_score = class_names[0], 0.0
        for a in aliases:
            cands = difflib.get_close_matches(a, class_names, n=1, cutoff=0.0)
            if cands:
                score = difflib.SequenceMatcher(None, a, cands[0]).ratio()
                if score > best_score:
                    best_name, best_score = cands[0], score
        return best_name

    # ---------- step 4: crop ----------
    def _crop_target_objects(
        self,
        image_path: str,
        detections: Dict,
        target_class: str,
        out_dir: str,
    ) -> List[str]:
        """
        Crops each detection whose object_name == target_class (case-insensitive).
        Saves JPEGs; returns list of file paths.
        """
        os.makedirs(out_dir, exist_ok=True)

        img_bgr = cv2.imread(image_path)  # OpenCV BGR
        if img_bgr is None:
            raise FileNotFoundError(image_path)
        H, W = img_bgr.shape[:2]

        saved = []
        stem = Path(image_path).stem
        tcl = target_class.lower()

        for i, obj in enumerate(detections.get("objects", [])):
            name = (obj.get("object_name") or "").strip()
            if name.lower() != tcl:
                continue
            x, y, w, h = [int(v) for v in obj.get("bounding_box", [0, 0, 0, 0])]
            # clip to image bounds
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            crop = img_bgr[y:y+h, x:x+w].copy()
            out_path = os.path.join(out_dir, f"{stem}_{tcl}_{i:03d}.jpg")
            cv2.imwrite(out_path, crop)
            saved.append(out_path)
        return saved

    # ---------- merge bbox-only JSON + seg JSON for an alias ----------
    def build_alias_bbox_mask_json(
        self,
        alias: str,
        bbox_json_path: str,
        seg_json_path: str,
        out_path: Optional[str] = None,
        iou_thresh: float = 0.3,
    ) -> Dict:
        """
        Merge a bbox-only JSON and a seg JSON for a given class alias.
        """
        alias_l = alias.lower()

        # ---- load JSONs ----
        with open(bbox_json_path, "r") as f:
            bbox_data = json.load(f)
        with open(seg_json_path, "r") as f:
            seg_data = json.load(f)

        bbox_objs = [
            obj for obj in bbox_data.get("objects", [])
            if (obj.get("object_name") or "").strip().lower() == alias_l
        ]

        seg_objs = [
            obj for obj in seg_data.get("objects", [])
            if (obj.get("object_name") or "").strip().lower() == alias_l
        ]

        merged_objects = []

        # Track which segmentation objects are already matched
        used_seg_idxs = set()

        for i, bobj in enumerate(bbox_objs):
            bb = bobj.get("bounding_box") or [0, 0, 0, 0]

            best_iou = 0.0
            best_j = None

            for j, sobj in enumerate(seg_objs):
                if j in used_seg_idxs:
                    continue
                sb = sobj.get("bounding_box") or [0, 0, 0, 0]
                iou = _bbox_iou_xyxy(bb, sb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j is not None and best_iou >= iou_thresh:
                sobj = seg_objs[best_j]
                used_seg_idxs.add(best_j)
                mask_poly = sobj.get("mask_polygon")  # might be None or list
                mconf = sobj.get("confidence", None)
            else:
                mask_poly = None
                mconf = None

            merged_objects.append({
                "object_name": bobj.get("object_name"),
                "bounding_box": bb,
                "mask_polygon": mask_poly,
                "bbox_confidence": bobj.get("confidence", None),
                "mask_confidence": mconf,
                "iou_match": float(best_iou),
            })

        out = {
            "alias": alias,
            "objects": merged_objects,
            "source_bbox_json": str(bbox_json_path),
            "source_seg_json": str(seg_json_path),
            "iou_threshold": iou_thresh,
        }

        if out_path is not None:
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[merge] Saved alias bbox+mask JSON → {out_path}")

        return out


# ---------- new callable "main" pipeline function ----------

def run_product_cropping_pipeline(
    image_path: str,
    product_url: str,
    out_dir: str = "crops",
    conf: float = 0.25,
    iou: float = 0.45,
    use_gpt: bool = True,
) -> Dict[str, Any]:
    """
    High-level pipeline you can call from other code.

    It:
      1) Runs ProductCropper.run (bbox-only detection + cropping)
      2) Ensures bbox + seg JSONs exist
      3) Merges bbox + seg JSONs for the chosen alias
      4) Returns a dict with all relevant outputs
    """
    pc = ProductCropper(gpt_model="gpt-5")

    # 1) main pipeline (bbox-only detection via object_detection.py)
    summary = pc.run(
        image_path=image_path,
        product_url=product_url,
        out_dir=out_dir,
        conf=conf,
        iou=iou,
        use_gpt=use_gpt,
    )

    print(json.dumps({
        "company_name": summary["profile"]["company_name"],
        "product_aliases": summary["profile"]["product_name"],
        "target_class": summary["target_class"],
        "num_crops_saved": summary["num_crops_saved"],
        "out_dir": summary["out_dir"]
    }, indent=2))

    # 2) bbox-only JSON
    stem = Path(image_path).stem
    bbox_json = f"{stem}_yolo11_o365.json"
    if not Path(bbox_json).exists():
        det = od.get_objects_json(image_path, conf=conf, iou=iou)
        with open(bbox_json, "w") as f:
            json.dump(det, f, indent=2)
        print(f"[bbox] Saved → {bbox_json}")

    # 3) segmentation JSON + annotated mask image
    seg_json = f"{stem}_yolo11_o365_seg.json"
    seg_img  = f"{stem}_yolo11_o365_seg.jpg"
    if not Path(seg_json).exists():
        seg_res = ods.get_objects_json(image_path, conf=conf, iou=iou)
        with open(seg_json, "w") as f:
            json.dump(seg_res, f, indent=2)
        print(f"[seg] Saved JSON → {seg_json}")

        seg_out_img = ods.draw_boxes_and_masks_pil(image_path, seg_res, seg_img)
        print(f"[seg] Saved annotated image → {seg_out_img}")
    else:
        seg_out_img = seg_img

    # 4) merge bbox + seg JSON for the chosen target class alias
    alias = summary["target_class"]
    merged_out = f"{stem}_yolo11_o365_merged_alias.json"

    merged = pc.build_alias_bbox_mask_json(
        alias=alias,
        bbox_json_path=bbox_json,
        seg_json_path=seg_json,
        out_path=merged_out,
        iou_thresh=0.3,
    )

    return {
        "summary": summary,
        "bbox_json": bbox_json,
        "seg_json": seg_json,
        "seg_annotated_image": seg_out_img,
        "merged_json": merged_out,
        "merged_data": merged,
    }


# ---------- example usage via __main__ ----------

if __name__ == "__main__":
    # Example values—replace with your own image + product URL
    IMAGE_PATH = "previewar_test#1.jpg"
    PRODUCT_URL = (
        "https://www.amazon.com/MODNEST-Modular-Sectional-Boneless-Assembly/dp/"
        "B0FCYC86TT/ref=sr_1_5_sspa?crid=3RIV4C6CTWYQA&dib=eyJ2IjoiMSJ9."
        "mCR5SjVr3IuoofQI97UxVmePO3nKmQbrqGkH6q7BhCWvXZfA2gaJDgyWsF100Jp3IznRSrEL8WKrwF2Xtlr-Q6YdVwugk_h-"
        "vhluo-EvhxJBqShb2gctTmjV71AXRyHOoPE6xC5K1iS8ITO3gdrhSf93HanYw7yk5iuIDU0gQFvfQLiPHo05ZX5PuYYk5As943eAeCxhe_d7i07UPtixVaCT_4yDty6lWukpMHvQmtssdgjG_"
        "zKvPqz8uqzZ5oAjIljfU3T1fj2lJKgKnqjlWxjA504G0RVwfRlQuUKr5bM.bAdATDJIW3OxMs6M16JiHRG9zAvYg7_B-SoRvPV4i5s&dib_tag=se&keywords=couch&qid=1762565138&"
        "sprefix=cou%2Caps%2C264&sr=8-5-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"
    )
    OUT_DIR = "crops"

    # Call the new main pipeline function
    results = run_product_cropping_pipeline(
        image_path=IMAGE_PATH,
        product_url=PRODUCT_URL,
        out_dir=OUT_DIR,
        conf=0.25,
        iou=0.45,
        use_gpt=True,
    )

    # Optional: pretty-print top-level results
    print(json.dumps({
        "company_name": results["summary"]["profile"]["company_name"],
        "product_aliases": results["summary"]["profile"]["product_name"],
        "target_class": results["summary"]["target_class"],
        "num_crops_saved": results["summary"]["num_crops_saved"],
        "out_dir": results["summary"]["out_dir"],
        "bbox_json": results["bbox_json"],
        "seg_json": results["seg_json"],
        "merged_json": results["merged_json"],
    }, indent=2))

