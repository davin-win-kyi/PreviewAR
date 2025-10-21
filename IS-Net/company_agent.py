#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

# These must exist in your project:
# decision_agent.resolve(url) -> dict with your schema
# amazon_api.fetch_rainforest_product(asin: str) -> Any
from desicion_agent import resolve  # <-- fixed typo
from amazon_api import fetch_rainforest_product

from dotenv import load_dotenv

# looks for a .env in the current working directory (or parents)
load_dotenv()

def _normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Lowercase keys and replace spaces with underscores (handles 'api call link')."""
    return {str(k).lower().replace(" ", "_"): v for k, v in d.items()}

def _validate_schema(d: Dict[str, Any]) -> None:
    required = ["agent", "company", "identifier"]
    missing = [k for k in required if k not in d or d[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

def _save_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def process_url(url: str, out_path: str | None = None):
    """
    Resolve URL -> schema via decision_agent.resolve, then dispatch.
    For Amazon, calls amazon_api.fetch_rainforest_product(asin=identifier).
    If out_path is provided, saves JSON there; otherwise auto-generates a file name.

    Returns a dict with keys: {"company", "asin", "saved_to", "data"}.
    """
    # 1) Resolve schema
    raw_schema = resolve(url)
    if not isinstance(raw_schema, dict):
        try:
            raw_schema = json.loads(raw_schema)  # type: ignore[arg-type]
        except Exception as e:
            raise TypeError(
                f"resolve(url) must return dict or JSON string; got {type(raw_schema)}"
            ) from e

    # 2) Normalize + validate
    schema = _normalize_keys(raw_schema)
    _validate_schema(schema)

    company = str(schema["company"]).strip().lower()
    asin = str(schema["identifier"]).strip()

    # 3) Dispatch per company
    if company == "amazon":
        data = fetch_rainforest_product(asin=asin)

        # 4) Decide output path
        if out_path:
            path = Path(out_path)
        else:
            # ./outputs/amazon/{ASIN}_{YYYYmmdd-HHMMSS}.json
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = Path("outputs") / "amazon" / f"{asin}_{ts}.json"

        saved = _save_json(data, path)
        return out_path

    # Add more companies in future as elif blocks.
    raise NotImplementedError(f"No agent implemented for company='{schema['company']}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python route_decision.py <product_url> [output_json_path]")
        sys.exit(1)

    product_url = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    try:
        result = process_url(product_url, out_path=output_path)
        # Print a short confirmation plus the saved path.
        print(json.dumps({
            "company": result["company"],
            "asin": result["asin"],
            "saved_to": result["saved_to"]
        }, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(2)
