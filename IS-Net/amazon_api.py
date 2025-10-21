#!/usr/bin/env python3
"""
rf_raw_json_vars.py

Fetch a Rainforest API product JSON using ONLY the ASIN parameter.

Env vars:
- RAINFOREST_API_KEY  (required)
- AMAZON_DOMAIN       (optional, default: "amazon.com")
"""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def fetch_rainforest_product(asin: str, *, timeout: int = 25) -> dict:
    """
    Returns the raw JSON dict from Rainforest for the given ASIN.
    Uses env vars:
      - RAINFOREST_API_KEY (required)
      - AMAZON_DOMAIN (optional, default 'amazon.com')
    Raises:
      - RuntimeError if API key missing
      - requests.HTTPError on non-2xx responses
      - ValueError on invalid JSON
    """
    api_key = os.getenv("RAINFOREST_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Rainforest API key. Set RAINFOREST_API_KEY.")

    amazon_domain = os.getenv("AMAZON_DOMAIN", "amazon.com").strip() or "amazon.com"

    url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": api_key,
        "type": "product",
        "amazon_domain": amazon_domain,
        "asin": asin,
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

# ---------------------------
# Put your variables here:
ASIN = "B0FB2PSYPK"   # e.g., "B0FB2PSYPK"
OUTPUT_PATH = None    # e.g., "product.json" or None to just print
# ---------------------------

if __name__ == "__main__":
    try:
        data = fetch_rainforest_product(ASIN)
        if OUTPUT_PATH:
            save_json(data, OUTPUT_PATH)
            print(json.dumps({"ok": True, "saved_to": OUTPUT_PATH}))
        else:
            print(json.dumps(data, ensure_ascii=False))
    except requests.HTTPError as e:
        print(json.dumps({"error": f"HTTP error: {e}"}))
    except ValueError:
        print(json.dumps({"error": "Invalid JSON returned by Rainforest"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
