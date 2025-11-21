# product_cropping_pipeline.py

import os
import re
import json
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import cv2
from openai import OpenAI

# Segmentation YOLO (yolo11n-seg) – we will use this for everything
import best_image_post_processing.object_detection_segmented as ods

import best_image_post_processing.extract_url_info as eui

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


class ProductCropper:
    """
    End-to-end:
      URL -> product aliases -> choose YOLO class (from segmentation output) -> crop all detections of that class.

    This version is segmentation-only:
      - Uses `object_detection_segmented` (yolo11n-seg) to detect objects.
      - Uses the class names present in the segmentation JSON as the candidate classes.
      - Crops using the bounding_box fields provided by the segmentation model.
    """

    def __init__(self, gpt_model: str = "gpt-5"):
        self.client = OpenAI()
        self.gpt_model = gpt_model

    # ---------- public orchestrator ----------
    def run(
        self,
        image_path: str,
        product_url: str,
        out_dir: str = "best_image_post_processing/crops",
        conf: float = 0.25,
        iou: float = 0.45,
        use_gpt: bool = True,
        save_crops: bool = True,   # <--- NEW
    ) -> Dict:
        """
        Returns a summary dict containing:
          {
            "profile": {...},
            "target_class": str,
            "num_crops_saved": int,
            "saved_files": [paths...],
            "detections": <raw detections dict from ods.get_objects_json>,
            "out_dir": str,
          }

        All detections are from the segmentation model (yolo11n-seg).
        """
        # 1) Get product profile from your URL extractor
        profile = self._get_product_profile(product_url)

        # 2) Detections (segmentation YOLO)
        det = ods.get_objects_json(image_path, conf=conf, iou=iou)

        # 3) Candidate class names from segmentation JSON
        class_names = sorted({
            (obj.get("object_name") or "").strip()
            for obj in det.get("objects", [])
            if (obj.get("object_name") or "").strip()
        })

        if not class_names:
            raise RuntimeError(
                "Segmentation model returned no objects with object_name; "
                "cannot choose a target class."
            )

        # Choose target class via GPT or fuzzy-only, among classes present in this image
        target_class = (
            self._map_product_to_yolo_class_gpt(profile, class_names)
            if use_gpt else
            self._map_product_to_yolo_class_fuzzy(profile, class_names)
        )

        # 4) Crop all detections of that class (from segmentation detections)
        if save_crops:
            saved_files = self._crop_target_objects(image_path, det, target_class, out_dir)
        else:
            saved_files = []

        return {
            "profile": profile,
            "target_class": target_class,
            "num_crops_saved": len(saved_files),
            "saved_files": saved_files,
            "detections": det,   # segmentation detections
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
            "Candidate class names (from the segmentation model detections in this image):\n"
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

    # ---------- step 4: crop (using segmentation detections) ----------
    def _crop_target_objects(
        self,
        image_path: str,
        detections: Dict,
        target_class: str,
        out_dir: str,
    ) -> List[str]:
        """
        Crops each detection whose object_name == target_class (case-insensitive).
        Uses bounding_box coordinates from the segmentation JSON.
        Saves JPEGs; returns list of file paths.
        """
        os.makedirs(out_dir, exist_ok=True)

        img_bgr = cv2.imread(str(image_path))  # OpenCV BGR
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
            print("******************************OUTPUT PATH: ", out_path)
            cv2.imwrite(out_path, crop)
            saved.append(out_path)
        return saved


# ---------- new callable "main" pipeline function ----------

def run_product_cropping_pipeline(
    image_path: str,
    product_url: str,
    out_dir: str = "best_image_post_processing/crops",
    conf: float = 0.25,
    iou: float = 0.45,
    use_gpt: bool = True,
    save_crops: bool = True,   # <--- NEW
) -> Dict[str, Any]:
    """
    High-level pipeline you can call from other code.

    It:
      1) Runs ProductCropper.run (segmentation-only) to:
         - decide target_class
         - (optionally) make crops of the target_class
         - get segmentation detections
      2) Saves seg JSON + annotated seg image
      3) Filters seg JSON so it ONLY contains objects whose object_name == target_class
      4) Returns summary + filtered seg_json path + annotated image path
    """
    pc = ProductCropper(gpt_model="gpt-5")

    # 1) main pipeline (segmentation-only)
    summary = pc.run(
        image_path=image_path,
        product_url=product_url,
        out_dir=out_dir,
        conf=conf,
        iou=iou,
        use_gpt=use_gpt,
        save_crops=save_crops,   # <--- pass through
    )

    print(json.dumps({
        "company_name": summary["profile"]["company_name"],
        "product_aliases": summary["profile"]["product_name"],
        "target_class": summary["target_class"],
        "num_crops_saved": summary["num_crops_saved"],
        "out_dir": summary["out_dir"]
    }, indent=2))

    # Alias / target class name (used for filtering)
    alias = summary["target_class"]
    alias_lower = alias.lower()

    # Segmentation detections from ProductCropper (already computed)
    seg_res = summary["detections"]

    # 2) Write segmentation JSON + annotated mask image
    stem = Path(image_path).stem
    seg_json = f"best_image_post_processing/{stem}_yolo11_o365_seg.json"
    seg_img  = f"best_image_post_processing/{stem}_yolo11_o365_seg.jpg"

    # Save full segmentation results to JSON
    with open(seg_json, "w") as f:
        json.dump(seg_res, f, indent=2)
    print(f"[seg] Saved JSON → {seg_json}")

    # Draw boxes + masks from seg_res
    seg_out_img = ods.draw_boxes_and_masks_pil(image_path, seg_res, seg_img)
    print(f"[seg] Saved annotated image → {seg_out_img}")

    # 3) Filter seg_res to ONLY include objects of the alias/target class
    filtered_objects = [
        obj for obj in seg_res.get("objects", [])
        if (obj.get("object_name") or "").strip().lower() == alias_lower
    ]

    seg_res_filtered = dict(seg_res)
    seg_res_filtered["objects"] = filtered_objects

    # Overwrite seg_json with filtered objects only
    with open(seg_json, "w") as f:
        json.dump(seg_res_filtered, f, indent=2)

    print(
        f"[seg] Overwrote {seg_json} with only objects of class '{alias}' "
        f"({len(filtered_objects)} objects)"
    )

    return {
        "summary": summary,
        "seg_json": seg_json,               # filtered to only alias objects
        "seg_annotated_image": seg_out_img,
        "filtered_objects": filtered_objects,
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

