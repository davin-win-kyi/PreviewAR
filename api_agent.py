# make_company_fetcher.py
# Usage:
#   export OPENAI_API_KEY=sk-...
#   python make_company_fetcher.py --json '{"agent":"API agent","company":"Amazon","identifier":"B0FB2PSYPK","api call link":"https://webservices.amazon.com/paapi5/getitems"}'
#   # This writes fetch_<company>.py (e.g., fetch_Amazon.py)
#   # Then run the generated script:
#   python fetch_Amazon.py --json '{"agent":"API agent","company":"Amazon","identifier":"B0FB2PSYPK","api call link":"https://webservices.amazon.com/paapi5/getitems"}'
#
# Notes:
# - The generated script MUST use official APIs only (no scraping).
# - It outputs a single JSON line: {"images":[...], "L":..., "W":..., "H":..., "notes":"..."}
# - For Amazon PA-API v5, you’ll need env vars; the generated script will document them and fail gracefully if missing.

import argparse, json, os, re, sys
from pathlib import Path
from openai import OpenAI

MODEL = "gpt-5"  # do not set temperature; this model supports only the default

SYSTEM_PROMPT = """You are a senior API integrator.
Write ONE fully runnable Python 3.10+ script (no placeholders like <code here>).
The script must be *company-specific* based on the provided JSON and MUST:
- Accept input via:
  A) --input <path_to_json_file>  OR
  B) --json '<raw JSON string>'
- Expected JSON schema (keys exact):
  {"agent": <string>, "company": <string>, "identifier": <string or 'unknown'>, "api call link": <string>}
- Print ONLY a single JSON object to stdout with keys:
  {"images": [<url strings>], "L": <float_or_null>, "W": <float_or_null>, "H": <float_or_null>, "notes": <string>}
  *Dimensions must represent product size (in inches if possible). If only cm/mm are available, convert: inches = cm/2.54, inches = mm/25.4.
  *If dimensions unavailable, set L/W/H to null and explain in "notes".
  *Prefer highest-resolution product images.

- Use OFFICIAL public APIs for the given company:
  * Amazon → Product Advertising API v5 (PA-API v5) with GetItems (identifier is ASIN).
  * eBay → Buy APIs (product content endpoints).
  * Walmart, BestBuy, etc. → their official product APIs.
  * If the company has no suitable public API or credentials are missing, return empty images and null dims with a helpful "notes".

- Read any required credentials from environment variables and *document them in a header comment*, e.g.:
  For Amazon PA-API v5 (typical):
    PAAPI_ACCESS_KEY
    PAAPI_SECRET_KEY
    PAAPI_PARTNER_TAG
    PAAPI_REGION   (e.g., us-east-1)
    PAAPI_HOST     (e.g., webservices.amazon.com)
  For eBay Buy APIs:
    EBAY_BEARER_TOKEN
- Fail gracefully without raising uncaught exceptions. Never print anything except the final JSON.

- Minimal dependencies only: argparse, json, os, typing, re, time, hashlib/hmac if needed for signing, requests.
- Encapsulate company logic in a function like fetch_amazon(), fetch_ebay(), etc., and switch on the "company" value.
- If identifier == "unknown" but the api call link (or any URL string) looks like a product URL, attempt to parse a valid ID
  (e.g., Amazon ASIN from /dp/<ASIN> or /gp/product/<ASIN>).
- Name entrypoint main(); guard with if __name__ == "__main__":.
"""

TEMPLATE_USER_PROMPT = """Write the script now. It must specifically support the company in this JSON and use its official API:
Input instance (for your own internal testing while writing code):
{schema_instance}

Important output contract:
- Print exactly one line of JSON: {{"images": [...], "L": <float_or_null>, "W": <float_or_null>, "H": <float_or_null>, "notes": "<string>"}}
- Do NOT print logs, errors, or extra lines—only the JSON.
- The script itself should be fully runnable after paste to a single file.
Return ONLY the code (no backticks, no commentary).
"""

def parse_args():
    p = argparse.ArgumentParser(description="Generate a company-specific product fetcher via GPT.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--json", type=str, help="Raw JSON string matching the schema.")
    g.add_argument("--input", type=str, help="Path to JSON file matching the schema.")
    p.add_argument("--out", type=str, help="Output filename (defaults to fetch_<company>.py).")
    return p.parse_args()

def load_schema(args):
    if args.json:
        return json.loads(args.json)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            return json.load(f)

def canonical_company_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "Company"
    return f"fetch_{safe}.py"

def main():
    args = parse_args()
    schema = load_schema(args)

    # Basic validation
    for key in ["agent", "company", "identifier", "api call link"]:
        if key not in schema:
            print(json.dumps({
                "images": [],
                "L": None, "W": None, "H": None,
                "notes": f"Missing required key '{key}' in input JSON."
            }))
            return

    out_file = args.out or canonical_company_filename(schema["company"])

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_prompt = TEMPLATE_USER_PROMPT.format(
        schema_instance=json.dumps(schema, ensure_ascii=False)
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # No temperature param: model uses default
    )

    code = resp.choices[0].message.content
    if not code or "import" not in code:
        print(json.dumps({
            "images": [],
            "L": None, "W": None, "H": None,
            "notes": "Failed to generate a valid script from the model."
        }))
        return

    Path(out_file).write_text(code, encoding="utf-8")
    print(json.dumps({
        "images": [],
        "L": None, "W": None, "H": None,
        "notes": f"Wrote {out_file}. Now run it with your JSON to get images and dimensions."
    }))

if __name__ == "__main__":
    main()