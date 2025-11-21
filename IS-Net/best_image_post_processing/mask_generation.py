#!/usr/bin/env python3
import os
import json
import glob
import requests
from pathlib import Path
from typing import Dict, List, Any

import cv2
import replicate

# Your existing modules
import best_image_post_processing.object_detection as od
from best_image_post_processing.target_object import ProductCropper


def json_safe(obj: Any) -> Any:
    """
    Recursively convert arbitrary Python objects (including Replicate stream objects)
    into something json.dumps can handle.
    """
    # primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # bytes -> summarized
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": True, "length": len(obj)}

    # lists / tuples / sets
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]

    # dicts
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    # Replicate FileOutput (or similar) with a url attribute
    # Avoid importing internal classes; duck-type instead.
    url = getattr(obj, "url", None)
    if isinstance(url, str):
        return {"type": obj.__class__.__name__, "url": url}

    # Last resort: string repr (prevents crashes)
    return repr(obj)


import os
import json
import base64
from typing import List, Dict, Any

from openai import OpenAI

# Make sure OPENAI_API_KEY is set in your environment.
client = OpenAI()


def encode_image_to_base64(image_path: str) -> str:
    """
    Read an image from disk and return a base64-encoded data URL string.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    # You can tweak the mime type if you know it (jpeg/png).
    return f"data:image/jpeg;base64,{b64}"


def strip_code_fences(s: str) -> str:
    """
    Remove ``` or ```json ... ``` fences around JSON, if present.
    """
    s = s.strip()
    if s.startswith("```"):
        # remove starting fence line
        s = s.split("\n", 1)[-1]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()


def gpt_objects_not_alias(
    image_path: str,
    alias_name: str,
    model: str = "gpt-5",
    max_objects: int = 20,
) -> Dict[str, Any]:
    """
    Ask GPT (vision) to look at an image, list all objects,
    then (via instructions) exclude the target alias class.

    Returns:
      {
        "all_objects": [...],          # everything GPT listed
        "non_alias_objects": [...],    # filtered list (no alias / synonyms)
        "negative_prompt": "a, b, c"   # comma-separated string
      }
    """
    data_url = encode_image_to_base64(image_path)

    # We tell GPT to:
    #  - List distinct object categories (not pixel-precise labels)
    #  - Exclude the alias and its obvious synonyms
    #  - Return JSON only
    user_text = f"""
        You are helping build a negative prompt for an image segmentation model.

        You are given:
        - A single photo (a product scene).
        - The target object alias: "{alias_name}".

        Your job:
        1. Look at the photo.
        2. Identify up to {max_objects} visually distinct object categories you clearly see
        in the scene (for example: "pillow", "blanket", "floor", "wall", "lamp",
        "coffee table", "plant").
        3. DO NOT include the main target object type ("{alias_name}") or any obvious
        synonyms / variations of the same object type (e.g. "sofa", "couch",
        "sectional" if the alias is "sofa", etc.).
        4. Use lowercase, English, singular nouns (e.g. "pillow", not "Pillows").
        5. Remove very generic words like "thing", "object", "stuff".

        Return ONLY valid JSON of the form:

        {{
        "all_objects": ["..."],
        "non_alias_objects": ["..."]
        }}
        """

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
    )

    raw = resp.choices[0].message.content
    raw = strip_code_fences(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # If something goes wrong, just fall back to an empty result.
        data = {"all_objects": [], "non_alias_objects": []}

    def _norm_list(key: str) -> List[str]:
        vals = data.get(key, [])
        out: List[str] = []
        seen = set()
        for v in vals:
            v = v.strip().lower()
            if not v:
                continue
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    all_objects = _norm_list("all_objects")
    non_alias_objects = _norm_list("non_alias_objects")

    # Safety check: if non_alias_objects came back empty but all_objects is non-empty,
    # we can derive non_alias_objects by filtering anything that obviously
    # matches the alias string.
    if not non_alias_objects and all_objects:
        alias_lower = alias_name.lower()
        non_alias_objects = [
            o for o in all_objects
            if alias_lower not in o  # simple filter; GPT already tried to exclude though
        ]

    negative_prompt = ", ".join(non_alias_objects) if non_alias_objects else "none"

    return {
        "all_objects": all_objects,
        "non_alias_objects": non_alias_objects,
        "negative_prompt": negative_prompt,
    }


def build_negative_prompt_for_image(
    image_path: str,
    alias_name: str,
    model: str = "gpt-5",
) -> Dict[str, Any]:
    """
    Convenience wrapper: returns alias + negative prompt data.
    """
    result = gpt_objects_not_alias(
        image_path=image_path,
        alias_name=alias_name,
        model=model,
    )
    return {
        "alias": alias_name,
        "all_objects": result["all_objects"],
        "non_alias_objects": result["non_alias_objects"],
        "negative_prompt": result["negative_prompt"],
    }


class MaskGenerationRunner:
    """
    End-to-end helper:
      - runs your ProductCropper to create crops/
      - takes first crop
      - re-detects classes on that crop with YOLO
      - calls Replicate grounded_sam with mask_prompt / negative_mask_prompt
      - saves all streamed outputs into output/ and writes a JSON manifest
    """

    def __init__(
        self,
        replicate_model: str = "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
        crops_dir: str = "best_image_post_processing/crops",
        output_dir: str = "best_image_post_processing/output",
    ):
        self.replicate_model = replicate_model
        self.crops_dir = crops_dir
        self.output_dir = output_dir
        os.makedirs(self.crops_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------- public API ----------
    def run(
        self,
        image_path: str,
        product_url: str,
        conf: float = 0.25,
        iou: float = 0.45,
        use_gpt: bool = True,
        adjustment_factor: int = -15,
    ) -> Dict:
        """
        Orchestrates the whole flow and returns a summary dict.
        """
        # 1) Create crops for the target product via your earlier pipeline
        pc = ProductCropper(gpt_model="gpt-5")
        summary = pc.run(
            image_path=image_path,
            product_url=product_url,
            out_dir=self.crops_dir,
            conf=conf,
            iou=iou,
            use_gpt=use_gpt,
            save_crops=False
        )
        target_class = summary["target_class"]

        # 2) Pick the first crop
        crop_path = self._first_crop_path(self.crops_dir)
        if crop_path is None:
            return {
                "error": "No crops found; detection may have missed the target.",
                "target_class": target_class,
                "profile": summary["profile"],
                "detections": summary["detections"],
                "crops_dir": self.crops_dir,
                "output_dir": self.output_dir,
            }

        # 3) (optional) Run YOLO on the crop to list present classes
        classes_in_crop = self._detect_classes_on_image(crop_path)

        # 4) Prepare prompts
        mask_prompt = target_class

        if use_gpt:
            # GPT looks at the crop image and builds a negative prompt
            neg_info = build_negative_prompt_for_image(
                image_path=crop_path,
                alias_name=target_class,
                model="gpt-5",  # or "gpt-5" once vision is enabled there
            )
            negative_mask_prompt = neg_info["negative_prompt"]
            gpt_non_alias_objects = neg_info["non_alias_objects"]
        else:
            # fall back to YOLO-based list
            gpt_non_alias_objects = sorted(
                {c for c in classes_in_crop if c.lower() != target_class.lower()}
            )
            negative_mask_prompt = ",".join(gpt_non_alias_objects) or "none"

        # 5) Replicate call + manifest
        rep_out = self._call_replicate_grounded_sam(
            image=crop_path,
            mask_prompt=mask_prompt,
            negative_mask_prompt=negative_mask_prompt,
            adjustment_factor=adjustment_factor,
        )

        return {
            "target_class": target_class,
            "classes_in_crop": classes_in_crop,
            "mask_prompt": mask_prompt,
            "negative_mask_prompt": negative_mask_prompt,
            "gpt_non_alias_objects": gpt_non_alias_objects,
            "output_files": rep_out["saved_files"],
            "manifest_path": rep_out["manifest_path"],
            "crops_dir": self.crops_dir,
            "output_dir": self.output_dir,
        }

    # ---------- helpers ----------
    def _first_crop_path(self, crops_dir: str) -> str | None:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        candidates: List[str] = []
        for e in exts:
            candidates.extend(glob.glob(os.path.join(crops_dir, e)))
        candidates.sort()
        return candidates[0] if candidates else None

    def _detect_classes_on_image(self, image_path: str) -> List[str]:
        """
        Uses the same YOLO11 (Objects365) model to find which classes exist in the crop.
        Returns a unique, sorted list of class names.
        """
        model = od.load_model()
        res = model(image_path, conf=0.10, iou=0.45, max_det=200, verbose=False)[0]
        names = model.names
        classes = []
        if res.boxes is not None and res.boxes.cls is not None:
            for c in res.boxes.cls.cpu().numpy().tolist():
                classes.append(names[int(c)])
        return sorted(list(set(classes)))

    def _call_replicate_grounded_sam(
        self,
        image: str,
        mask_prompt: str,
        negative_mask_prompt: str,
        adjustment_factor: int = -15,
    ) -> Dict[str, str | list]:
        """
        Calls Replicate's schananas/grounded_sam on the local image file.
        - Saves any image outputs to self.output_dir
        - Writes a JSON manifest with stream details (all items are JSON-safe)
        Returns: { "saved_files": [...], "manifest_path": "<path>" }
        """
        saved: List[str] = []
        items_manifest: List[Dict] = []

        # 🔐 Make sure output_dir exists (in case CWD changed or __init__ wasn't called)
        os.makedirs(self.output_dir, exist_ok=True)

        print("Mask prompt: ", mask_prompt)
        print("negative mask prompt: ", negative_mask_prompt)

        with open(image, "rb") as f:
            output = replicate.run(
                self.replicate_model,
                input={
                    "image": f,
                    "mask_prompt": mask_prompt,
                    "negative_mask_prompt": negative_mask_prompt,
                    "adjustment_factor": adjustment_factor,
                },
            )

            for idx, item in enumerate(output):
                urls = self._extract_image_urls(item)
                saved_this = []
                for j, url in enumerate(urls):
                    out_path = os.path.join(
                        self.output_dir, f"grounded_sam_{idx:03d}_{j:02d}.png"
                    )
                    try:
                        if isinstance(url, (bytes, bytearray)):
                            with open(out_path, "wb") as wf:
                                wf.write(url)
                        else:
                            self._download(url, out_path)
                        saved.append(out_path)
                        saved_this.append(out_path)
                    except Exception as e:
                        saved_this.append(f"[save-error] {e}")

                items_manifest.append({
                    "step_index": idx,
                    "raw_item": json_safe(item),
                    "extracted_urls_count": len(urls),
                    "saved_files": saved_this,
                })

        manifest = {
            "replicate_model": self.replicate_model,
            "source_image": str(image),
            "mask_prompt": mask_prompt,
            "negative_mask_prompt": negative_mask_prompt,
            "adjustment_factor": adjustment_factor,
            "output_dir": self.output_dir,
            "num_files_saved": len(saved),
            "items": items_manifest,
        }

        manifest_path = os.path.join(self.output_dir, "grounded_sam_output.json")

        # 🔐 Ensure the directory for the manifest exists
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        with open(manifest_path, "w", encoding="utf-8") as jf:
            json.dump(manifest, jf, ensure_ascii=False, indent=2)

        return {"saved_files": saved, "manifest_path": manifest_path}


    def _extract_image_urls(self, item) -> List[str | bytes]:
        """
        The Replicate output schema can vary (URL strings, dicts with image URLs, FileOutput objects, etc.).
        Collect anything that looks like an image URL or raw bytes.
        """
        urls: List[str | bytes] = []

        # strings (likely URLs)
        if isinstance(item, str):
            if item.startswith(("http://", "https://")):
                urls.append(item)
            return urls

        # bytes-like -> save directly
        if isinstance(item, (bytes, bytearray)):
            urls.append(item)
            return urls

        # objects with a .url attribute (e.g., FileOutput)
        u = getattr(item, "url", None)
        if isinstance(u, str) and u.startswith(("http://", "https://")):
            urls.append(u)

        # dicts and lists
        if isinstance(item, dict):
            for v in item.values():
                urls.extend(self._extract_image_urls(v))
        elif isinstance(item, list):
            for v in item:
                urls.extend(self._extract_image_urls(v))

        return urls

    def _download(self, url: str, out_path: str):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)


# -------- example CLI usage --------
if __name__ == "__main__":
    """
    Example:
      export OPENAI_API_KEY=sk-...
      export REPLICATE_API_TOKEN=...
      python mask_generation.py
    """
    IMAGE_PATH = "previewar_test#1.jpg"
    PRODUCT_URL = "https://www.amazon.com/MODNEST-Modular-Sectional-Boneless-Assembly/dp/B0FCYC86TT/"

    runner = MaskGenerationRunner()
    summary = runner.run(
        image_path=IMAGE_PATH,
        product_url=PRODUCT_URL,
        conf=0.25,
        iou=0.45,
        use_gpt=True,
        adjustment_factor=-15,
    )

    print(json.dumps({
        "target_class": summary.get("target_class"),
        "classes_in_crop": summary.get("classes_in_crop"),
        "mask_prompt": summary.get("mask_prompt"),
        "negative_mask_prompt": summary.get("negative_mask_prompt"),
        "num_outputs": len(summary.get("output_files", [])),
        "output_dir": summary.get("output_dir"),
        "manifest_path": summary.get("manifest_path"),
    }, indent=2))
