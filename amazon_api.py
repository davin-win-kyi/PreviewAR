#!/usr/bin/env python3
"""
rf_raw_json_vars.py

Pass ASIN, Amazon domain, and Rainforest API key as variables.
Makes a call to Rainforest API (type=product) and returns the raw JSON.

- Do NOT hardcode real keys in code committed to repos.
- Prefer reading api_key from an env var in real projects.
"""

import json
import os
import requests

from dotenv import load_dotenv

load_dotenv()

def fetch_rainforest_product(asin: str, amazon_domain: str, api_key: str, *, timeout: int = 25) -> dict:
    """
    Returns the raw JSON dict from Rainforest for the given ASIN and domain.
    Raises a requests.HTTPError on non-2xx or ValueError on invalid JSON.
    """
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
ASIN = "B0FB2PSYPK"          # e.g., "B0FB2PSYPK"
AMAZON_DOMAIN = "amazon.com" # e.g., "amazon.co.uk", "amazon.de", etc.
RAINFOREST_API_KEY = os.getenv("RAINFOREST_API_KEY")  # replace or set env var
OUTPUT_PATH = None           # e.g., "product.json" or None to just print
# ---------------------------

if __name__ == "__main__":
    try:
        if not RAINFOREST_API_KEY or RAINFOREST_API_KEY == "YOUR_RAINFOREST_KEY":
            raise RuntimeError("Missing Rainforest API key. Set RAINFOREST_API_KEY or replace RAINFOREST_API_KEY variable.")
        data = fetch_rainforest_product(ASIN, AMAZON_DOMAIN, RAINFOREST_API_KEY)
        if OUTPUT_PATH:
            save_json(data, OUTPUT_PATH)
            print(json.dumps({"ok": True, "saved_to": OUTPUT_PATH}))
        else:
            # Print the raw Rainforest JSON (single line)
            print(json.dumps(data, ensure_ascii=False))
    except requests.HTTPError as e:
        print(json.dumps({"error": f"HTTP error: {e}"}))
    except ValueError:
        print(json.dumps({"error": "Invalid JSON returned by Rainforest"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))