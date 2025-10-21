import os, re, json
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# get the env vriables as nessecary:
from dotenv import load_dotenv
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
TIMEOUT = int(os.getenv("TIMEOUT", "30"))
UA = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# (optional) common ID helpers for sanity fixes
ASIN_RX = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})(?:[/?]|$)")
EBAY_ITEM_RX = re.compile(r"/itm/(?:.*?/)?(\d+)(?:[/?]|$)")
ETSY_LISTING_RX = re.compile(r"/listing/(\d+)(?:[/?]|$)")

def _maybe_fix_identifier(url: str, company: str, identifier: str | None) -> str | None:
    c = (company or "").lower()
    if c == "amazon":
        m = ASIN_RX.search(url);  return m.group(1) if m else identifier
    if c == "ebay":
        m = EBAY_ITEM_RX.search(url); return m.group(1) if m else identifier
    if c == "etsy":
        m = ETSY_LISTING_RX.search(url); return m.group(1) if m else identifier
    return identifier

def gpt_decide(url: str) -> dict:
    """
    Ask GPT: does this URL belong to a site with an official, public, product-read API?
    If YES and certain: return {"company","identifier","api call link"}.
    Else: return {"decision":"web"}.
    """
    host = urlparse(url).netloc.lower()
    system = (
        "You are a strict JSON-only router. "
        "Given a product URL, decide if the SITE exposes an OFFICIAL, PUBLIC, PRODUCT-READ API "
        "(intended for third parties to retrieve product details/images/specs). "
        "Examples: Amazon Product Advertising API, eBay Browse/Buy API, Etsy v3 listings. "
        "Exclude partner/seller back-office APIs and private/internal endpoints. "
        "If NOT CERTAIN, choose web.\n\n"
        "OUTPUT (JSON only):\n"
        "1) If an official public product-read API exists and you are certain how to call it:\n"
        "   {\"company\": <company name>, \"identifier\": <extracted id or 'unknown'>, \"api call link\": <canonical endpoint or templated URL>}\n"
        "   - Extract an ID from the URL if possible (e.g., ASIN/item_id/listing_id). "
        "     If no ID is present, set identifier to \"unknown\" and provide a canonical/templated endpoint.\n"
        "2) Otherwise (no public product-read API or uncertain):\n"
        "   {\"decision\":\"web\"}"
    )
    user = f"URL: {url}\nHost: {host}\nRespond with JSON ONLY per the rules."
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=1.0
    )
    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except Exception:
        return {"decision": "web"}

# -------------------------
# Improved HTML fetch + extraction (no evasion)
# -------------------------

BLOCK_PATTERNS = [
    "captcha", "are you a robot", "robot check", "bot protection",
    "perimeterx", "cloudflare", "akamai", "distil_r", "sensor_data",
    "unusual traffic", "human verification"
]
NUM_UNIT = re.compile(r'([\d\.]+)\s*(in|inch|inches|cm|mm|m)\b', re.I)

def _redirect_chain_info(resp: requests.Response) -> list[dict]:
    chain = []
    for h in getattr(resp, "history", []) or []:
        chain.append({
            "status": h.status_code,
            "url": h.url,
            "location": h.headers.get("Location")
        })
    return chain

def _looks_blocked(resp: requests.Response, low_html: str) -> bool:
    if resp.status_code in (401, 403, 429):
        return True
    return any(pat in low_html for pat in BLOCK_PATTERNS)

def _short_text_preview(html: str, limit: int = 260) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = " ".join(soup.stripped_strings)
        return (text[:limit] + "…") if len(text) > limit else text
    except Exception:
        return html[:limit] + ("…" if len(html) > limit else "")

def _q(value):
    """Parse '12 in', {'value':12,'unitText':'in'}, 12, etc."""
    if value is None: return None
    if isinstance(value, (int, float)): return {"value": float(value), "unit": None}
    if isinstance(value, dict):
        raw = value.get("value")
        unit = (value.get("unitText") or value.get("unitCode") or value.get("unit"))
        if isinstance(raw, (int, float)):
            return {"value": float(raw), "unit": (unit.lower() if isinstance(unit, str) else unit)}
        if isinstance(raw, str):
            m = NUM_UNIT.search(raw)
            return {"value": float(m.group(1)), "unit": m.group(2).lower()} if m else {"raw": raw}
    if isinstance(value, str):
        m = NUM_UNIT.search(value)
        return {"value": float(m.group(1)), "unit": m.group(2).lower()} if m else {"raw": value}
    return None

def _to_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def _extract_schemaorg_product(html: str) -> dict:
    """
    Parse schema.org JSON-LD Product blocks and return a compact summary.
    """
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all("script", attrs={"type": "application/ld+json"})
    products = []

    def walk(obj):
        if isinstance(obj, dict):
            t = obj.get("@type")
            types = _to_list(t)
            if any(isinstance(tt, str) and tt.lower() == "product" for tt in types):
                products.append(obj)
            if "@graph" in obj:
                for g in _to_list(obj["@graph"]):
                    walk(g)
            for v in obj.values():
                if isinstance(v, (dict, list)): walk(v)
        elif isinstance(obj, list):
            for it in obj: walk(it)

    for tag in blocks:
        txt = (tag.string or tag.text or "").strip()
        if not txt: continue
        try:
            data = json.loads(txt)
        except Exception:
            continue
        walk(data)

    if not products:
        return {"name": None, "brand": None, "images": [], "dimensions": {"normalized": {}, "raw": {}}}

    p = products[0]
    # name/brand
    name = p.get("name")
    brand = p.get("brand")
    if isinstance(brand, dict): brand = brand.get("name")

    # images
    images = []
    imgs = p.get("image")
    if isinstance(imgs, str): images.append(imgs)
    elif isinstance(imgs, list): images.extend([i for i in imgs if isinstance(i, str)])

    # dimensions (direct + additionalProperty)
    dims_raw = {}
    for k in ("width", "height", "depth", "length", "size"):
        if k in p:
            v = p[k].get("value") if isinstance(p[k], dict) else p[k]
            q = _q(v)
            if q: dims_raw[k] = q
    for ap in _to_list(p.get("additionalProperty")):
        if not isinstance(ap, dict): continue
        n = str(ap.get("name", "")).lower()
        q = _q(ap.get("value"))
        if not q: continue
        for key in ("width", "height", "depth", "length"):
            if key in n and key not in dims_raw:
                dims_raw[key] = q

    dims_norm = {"L": dims_raw.get("length") or dims_raw.get("depth"),
                 "W": dims_raw.get("width"),
                 "H": dims_raw.get("height")}

    return {"name": name, "brand": brand, "images": images, "dimensions": {"normalized": dims_norm, "raw": dims_raw}}

def _parse_specs_table_dimensions(html: str) -> dict:
    """
    Heuristic: look for table/dl label cells with width/height/depth/length and parse numeric+unit.
    """
    soup = BeautifulSoup(html, "html.parser")
    dims = {}
    def capture(label, val):
        label = (label or "").lower(); val = val or ""
        for key in ("width","height","depth","length"):
            if key in label and key not in dims:
                q = _q(val)
                if q: dims[key] = q

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            th = tr.find(["th","td"])
            tds = tr.find_all("td")
            if th and tds:
                capture(th.get_text(" ", strip=True), tds[-1].get_text(" ", strip=True))

    for dl in soup.find_all("dl"):
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                capture(dt.get_text(" ", strip=True), dd.get_text(" ", strip=True))

    return {"normalized": {
                "L": dims.get("length") or dims.get("depth"),
                "W": dims.get("width"),
                "H": dims.get("height")
            },
            "raw": dims}

def _extract_images_fallbacks(html: str, base_url: str | None) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    imgs = []

    # Open Graph
    for m in soup.find_all("meta", attrs={"property": "og:image"}):
        c = m.get("content")
        if c: imgs.append(c)

    # <img> tags
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if src: imgs.append(src)

    # absolutize & dedup
    out, seen = [], set()
    for u in imgs:
        full = urljoin(base_url, u) if base_url and not str(u).startswith(("http://","https://")) else u
        if full and full not in seen:
            seen.add(full); out.append(full)
    return out

def fetch_html(url: str) -> dict:
    """
    Fetch page without any bot-evasion.
    - If blocked: return structured block info (final_url, status, preview).
    - If ok: return product extraction (schema.org + spec-table) and image fallbacks.
    """
    try:
        with requests.Session() as s:
            r = s.get(url, headers=UA, timeout=TIMEOUT, allow_redirects=True)
    except Exception as e:
        return {"agent":"Web scraping agent",
                "blocked": False,
                "error": f"fetch_error: {str(e)}",
                "requested_url": url}

    html = r.text or ""
    low = html.lower()
    final_url = r.url
    chain = _redirect_chain_info(r)

    # page title (best-effort)
    title = None
    try:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    except Exception:
        pass

    if _looks_blocked(r, low):
        return {
            "agent":"Web scraping agent",
            "blocked": True,
            "status_code": r.status_code,
            "requested_url": url,
            "final_url": final_url,
            "redirect_chain": chain,
            "page_title": title,
            "preview": _short_text_preview(html),
            "note": "Access appears restricted (bot wall / CAPTCHA detected). No bypass attempted."
        }

    # Not blocked → extract product data
    schema = _extract_schemaorg_product(html)
    images = list(schema.get("images") or [])
    # add fallbacks if needed
    if not images:
        images = _extract_images_fallbacks(html, final_url)

    dims = schema.get("dimensions") or {"normalized": {}, "raw": {}}
    if not any(dims["normalized"].values()) and not dims["raw"]:
        dims = _parse_specs_table_dimensions(html)

    return {
        "agent":"Web scraping agent",
        "blocked": False,
        "status_code": r.status_code,
        "requested_url": url,
        "final_url": final_url,
        "redirect_chain": chain,
        "page_title": title,
        "schema_org": schema,         # includes name, brand, images (if present), dimensions
        "images": images,             # final image list after fallbacks
        "dimensions": dims,           # final dims after fallback
        "html": html                  # keep or remove if you want smaller payloads
    }

def resolve(url: str) -> str:
    """
    Returns one of:
    - {"agent":"API agent","company":"...","identifier":"...","api call link":"..."}
    - {"agent":"Web scraping agent", ... rich fields ...}
    """
    decision = gpt_decide(url)

    # API path
    if all(k in decision for k in ("company","identifier","api call link")):
        fixed = _maybe_fix_identifier(url, decision["company"], decision.get("identifier"))
        if fixed:
            decision["identifier"] = fixed
        out = {
            "agent": "API agent",
            "company": decision["company"],
            "identifier": decision["identifier"],
            "api call link": decision["api call link"],
        }
        return json.dumps(out, ensure_ascii=False)

    # Web path
    return json.dumps(fetch_html(url), ensure_ascii=False)

# CLI
if __name__ == "__main__":
    import sys
    url = os.environ.get("PRODUCT_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not url:
        print("Usage: python universal_router_agent.py <product_url>\n"
              "   or set PRODUCT_URL in the environment.")
        raise SystemExit(1)
    print(resolve(url))