# best_image_selector.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import time

# ⬇️ Your existing module/class with the method already implemented.
# Expecting: ExtractAmazonInfo().extract_media_and_hwl(json_path) -> dict-like structure containing images
import extract_amazon_info

from dotenv import load_dotenv

# looks for a .env in the current working directory (or parents)
load_dotenv()

try:
    # OpenAI >= 1.x client
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class GPTSelection:
    best_index: int
    best_url: str
    reason: str
    details: List[Dict[str, Any]]


class BestImageSelector:
    """
    Usage:

        selector = BestImageSelector(
            model="gpt-4o-mini",  # or "gpt-4o", "gpt-4.1-mini", etc. (any vision-capable model)
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        result = selector.get_best_image_from_amazon_json(
            r"outputs\amazon\B0FB2PSYPK_20251020-163146.json"
        )

        print(result.best_url, result.reason)
    """

    DEFAULT_CRITERIA = (
        "Pick the best image for an ecommerce PDP. "
        "Must be sharp, subject centered, neutral background, even lighting, "
        "and at least ~1200 px on the short edge. Avoid watermarks/borders."
    )

    def __init__(
        self,
        model: str = "gpt-5",
        openai_api_key: Optional[str] = None,
        request_timeout: int = 60,
        max_retries: int = 2,
    ):
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        # Lazily init client to avoid import errors in environments without openai
        self._client = None
        
        self._openai_api_key = os.getenv("OPENAI_API_KEY")

    # -----------------------------
    # Public: main entry points
    # -----------------------------
    def get_best_image_from_amazon_json(
        self,
        json_path: str | Path,
        criteria: Optional[str] = None,
    ) -> GPTSelection:
        """
        1) Calls your ExtractAmazonInfo.extract_media_and_hwl(json_path).
        2) Extracts a flat list of image URLs.
        3) Sends those image URLs to GPT to select the best one.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")


        # Use your existing extractor
        data = json.load(open(json_path, "r", encoding="utf-8"))
        media_info = extract_amazon_info.extract_media_and_hwl(data)

        image_urls = self._pluck_image_urls(media_info)
        if not image_urls:
            # Fallback: try to parse the raw JSON directly (in case structure differs)
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            image_urls = self._pluck_image_urls(raw)

        image_urls = self._unique(image_urls)
        if not image_urls:
            raise ValueError("No image URLs found from extract_media_and_hwl or JSON.")

        return self.select_best_image_via_gpt(image_urls, criteria or self.DEFAULT_CRITERIA)

    def select_best_image_via_gpt(
        self,
        image_urls: List[str],
        criteria: str = DEFAULT_CRITERIA,
    ) -> GPTSelection:
        """
        Calls a vision-capable GPT model with the list of images and returns the chosen best one.
        """
        if len(image_urls) == 1:
            return GPTSelection(best_index=0, best_url=image_urls[0], reason="Only one image.", details=[{"index": 0, "url": image_urls[0], "score": 1.0}])

        client = self._get_openai_client()

        # Build a JSON-only response instruction
        schema_hint = {
            "type": "object",
            "properties": {
                "best_index": {"type": "integer", "minimum": 0, "maximum": len(image_urls) - 1},
                "reason": {"type": "string"},
                "details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "url": {"type": "string"},
                            "score": {"type": "number"},
                            "notes": {"type": "string"},
                        },
                        "required": ["index", "url", "score"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["best_index", "reason", "details"],
            "additionalProperties": False,
        }

        system_msg = (
            "You are an ecommerce image judge. "
            "Pick EXACTLY ONE best image for a product detail page (PDP) using the user's criteria. "
            "Prefer crisp, centered, evenly lit images on neutral backgrounds. "
            "Penalize watermarks, borders, collages, low resolution, clutter, and awkward crops."
        )

        # Present the images with indices to the model
        text_listing = "Images to consider (index: url):\n" + "\n".join(
            f"{i}: {u}" for i, u in enumerate(image_urls)
        )

        user_text = (
            f"Criteria:\n{criteria}\n\n"
            "Return STRICT JSON only matching the provided schema (no prose). "
            "Use 'best_index' to indicate your final choice.\n\n"
            f"{text_listing}"
        )

        # Vision prompt with image_url blocks (one per image)
        content_blocks = [{"type": "text", "text": user_text}]
        for url in image_urls:
            content_blocks.append({"type": "image_url", "image_url": {"url": url}})

        # Robust retries
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_schema", "json_schema": {"name": "image_choice", "schema": schema_hint, "strict": True}},
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": content_blocks},
                    ],
                    timeout=self.request_timeout,
                )

                raw = resp.choices[0].message.content
                data = self._safe_json_loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("Model did not return a JSON object.")

                best_index = int(data.get("best_index", -1))
                if not (0 <= best_index < len(image_urls)):
                    raise ValueError("best_index out of range.")

                reason = str(data.get("reason", "")).strip() or "No reason provided."
                details = data.get("details", [])
                best_url = image_urls[best_index]

                return GPTSelection(
                    best_index=best_index,
                    best_url=best_url,
                    reason=reason,
                    details=details,
                )

            except Exception as e:
                last_err = e
                time.sleep(0.6)  # brief backoff

        # If GPT fails, use a simple heuristic fallback
        fallback_idx = self._fallback_index(image_urls)
        return GPTSelection(
            best_index=fallback_idx,
            best_url=image_urls[fallback_idx],
            reason=f"Fallback selection due to GPT error: {last_err}",
            details=[{"index": i, "url": u, "score": 1.0 if i == fallback_idx else 0.5} for i, u in enumerate(image_urls)],
        )

    # -----------------------------
    # Helpers
    # -----------------------------
    def _get_openai_client(self):
        if self._client is None:
            if OpenAI is None:
                raise ImportError("openai package not installed. `pip install openai`")
            if not self._openai_api_key:
                raise EnvironmentError("OPENAI_API_KEY not set. Pass openai_api_key=... or set env var.")
            self._client = OpenAI(api_key=self._openai_api_key)
        return self._client

    def _pluck_image_urls(self, data: Any) -> List[str]:
        """
        Extract a flat list of image URLs from a variety of plausible shapes.
        Tries common keys your extractor might return (e.g., 'media', 'images', 'gallery', etc.).
        """
        urls: List[str] = []

        def maybe_add(u: Any):
            if isinstance(u, str) and self._looks_like_url(u):
                urls.append(u)

        def walk(x: Any):
            if isinstance(x, dict):
                for k, v in x.items():
                    lk = k.lower()
                    if lk in {"images", "media", "image_urls", "gallery", "hires", "image_list", "variant_images"} and isinstance(v, (list, dict)):
                        walk(v)
                    elif lk in {"main_image", "primary_image"}:
                        maybe_add(v if isinstance(v, str) else v.get("url") if isinstance(v, dict) else None)
                    else:
                        walk(v)
            elif isinstance(x, list):
                for item in x:
                    walk(item)
            else:
                maybe_add(x)

        walk(data)

        # Also check for known Amazon patterns inside dict items
        if not urls and isinstance(data, dict):
            # Sometimes images under: data.get("media", {}).get("images", [])
            media = data.get("media") if isinstance(data.get("media"), (list, dict)) else None
            if media:
                walk(media)

        # Deduplicate and keep http/https only
        urls = [u for u in self._unique(urls) if u.startswith("http")]
        return urls

    def _fallback_index(self, image_urls: List[str]) -> int:
        """
        Simple heuristic fallback:
          - Prefer URLs mentioning 'main' or 'primary'
          - Otherwise prefer ones that look largest (e.g., contain 1500, 2000, etc.)
          - Otherwise choose index 0
        """
        # 1) Look for hints in URL
        for i, u in enumerate(image_urls):
            if re.search(r"(main|primary|hero)", u, re.I):
                return i

        # 2) Guess by pixel hints in URL (e.g., ..._1500x1500_)
        def size_hint(u: str) -> int:
            nums = re.findall(r"(\d{3,5})[xX_](\d{3,5})", u)
            if nums:
                try:
                    w, h = map(int, nums[-1])
                    return min(w, h)
                except:
                    return 0
            # Also check standalone big numbers
            nums2 = [int(n) for n in re.findall(r"(\d{3,5})", u)]
            return max(nums2) if nums2 else 0

        sizes = [size_hint(u) for u in image_urls]
        if any(sizes):
            return max(range(len(sizes)), key=lambda i: sizes[i])

        # 3) Default
        return 0

    @staticmethod
    def _safe_json_loads(s: str) -> Any:
        import json
        return json.loads(s)

    @staticmethod
    def _unique(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    @staticmethod
    def _looks_like_url(u: str) -> bool:
        return isinstance(u, str) and u.startswith(("http://", "https://")) and ("." in u)


# -----------------------------
# Quick CLI hook (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pick best ecommerce image via GPT from Amazon JSON.")
    parser.add_argument("json_path", help="Path to Amazon JSON (e.g., outputs\\amazon\\B0FB2PSYPK_20251020-163146.json)")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--criteria", default=BestImageSelector.DEFAULT_CRITERIA)
    args = parser.parse_args()

    selector = BestImageSelector(model=args.model)
    sel = selector.get_best_image_from_amazon_json(args.json_path, args.criteria)
    print(json.dumps({
        "best_index": sel.best_index,
        "best_url": sel.best_url,
        "reason": sel.reason,
        "details": sel.details
    }, indent=2))

    best_url = sel.best_url

    # pass this into the DIS
