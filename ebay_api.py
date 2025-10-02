"""
This is the API link to use: GET https://api.ebay.com/commerce/catalog/v1_beta/product/{epid}

First step: 
- overall want to get a token, and that is done with the token link
- and then you want to be able to then get the token with the use of the Client ID and Client Secret

Second step: 
- then you will want to get the product information with the catalog link 
- and you will also want to use the token that you recieved from step 1


Side note: you can get the following two cases
- it is one item 
- it is a group of items


Third step: 
- parse the information to get the following information from the returned json: 
    - aspects: this is where you can get the product details, in this case
               you will want to get ask GPT to get the aspects that are related
               to the width, height and the length of the product
    - image: this is the main image which you will be recieving which you will 
             want to use 
    - additionalImages: this is where you will get any additional images that 
             are related to the product
"""


import os, base64, requests
from dotenv import load_dotenv

load_dotenv()

LEGACY_ID = "365839133746"
CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")

# --- 1) OAuth token (PRODUCTION) ---
basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
tok = requests.post(
    "https://api.ebay.com/identity/v1/oauth2/token",
    headers={"Content-Type":"application/x-www-form-urlencoded",
             "Authorization": f"Basic {basic}"},
    data={"grant_type":"client_credentials",
          "scope":"https://api.ebay.com/oauth/api_scope"},
    timeout=30
)
tok.raise_for_status()
access_token = tok.json()["access_token"]

BASE = "https://api.ebay.com/buy/browse/v1"
HEADERS = {
    "Authorization": f"Bearer {access_token}",
    "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
    "Accept": "application/json",
}

def looks_like_group_error(err_json: dict) -> bool:
    if not err_json or "errors" not in err_json:
        return False
    for e in err_json["errors"]:
        if e.get("errorId") == 11006:
            return True
        if "item group" in (e.get("message") or "").lower():
            return True
    return False

# --- 2) Try item-by-legacy-id, else fallback to item-group ---
params = {
    "legacy_item_id": LEGACY_ID,
    "fieldgroups": "PRODUCT",
    "quantity_for_shipping_estimate": "1",
}
r = requests.get(f"{BASE}/item/get_item_by_legacy_id",
                 params=params, headers=HEADERS, timeout=30)

if r.ok:
    print(r.json())
else:
    try:
        err = r.json()
    except Exception:
        r.raise_for_status()

    if looks_like_group_error(err):
        r2 = requests.get(f"{BASE}/item/get_items_by_item_group",
                          params={"item_group_id": LEGACY_ID},
                          headers=HEADERS, timeout=30)
        r2.raise_for_status()
        print(r2.json())
    else:
        # bubble the original error
        r.raise_for_status()
