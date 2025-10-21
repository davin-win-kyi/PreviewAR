# pip install openai python-dotenv
import os, json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use any multimodal vision model you have (text+image URLs)
MODEL = "gpt-5"  # swap if you prefer/are allowed to use "gpt-5" etc.

SYSTEM_MSG = (
    "You are an image selection assistant. "
    "Given criteria and a set of image URLs, you will return VALID JSON ONLY, with no extra text. "
    "Shape:\n"
    "{"
    "  \"winner\": \"<uri>\","
    "}"
)

def choose_top_image_simple(image_urls: List[str], criteria_prompt: str) -> Dict[str, Any]:
    """
    Minimal version: assumes the model replies with clean JSON (no code fences, no prose).
    Returns: dict with keys winner, rationale, ranking[].
    """
    user_content = [
        {
            "type": "text",
            "text": (
                "Evaluate each image against the criteria and select ONE best image.\n"
                "Return JSON ONLY in the exact shape described. No markdown or extra text.\n\n"
                f"Criteria:\n{criteria_prompt}\n\nImages follow:"
            )
        }
    ] + [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_content},
        ],
        # keep it minimal; omit params like temperature if your org/model restricts them
    )

    print("Response: ", resp.choices[0].message.content)

    return ""

    # return json.loads(resp.choices[0].message.content)

# ---------- Example ----------
if __name__ == "__main__":
    urls = [
        "https://m.media-amazon.com/images/I/81sk-iwQP7L._AC_SL1500_.jpg",
        "https://m.media-amazon.com/images/I/81ArZs5qX4L._AC_SL1500_.jpg",
        "https://m.media-amazon.com/images/I/81wAFBcL1vL._AC_SL1500_.jpg",
        "https://m.media-amazon.com/images/I/81lsFpuDPaL._AC_SL1500_.jpg",
        "https://m.media-amazon.com/images/I/916kka0RERL._AC_SL1500_.jpg",
        "https://m.media-amazon.com/images/I/81sZZZOuJdL._AC_SL1500_.jpg",
        "https://m.media-amazon.com/images/I/91HK0XpwXzL._AC_SL1500_.jpg",
        "https://m.media-amazon.com/images/I/714AxVWFR2L._AC_SL1500_.jpg",
        "https://m.media-amazon.com/images/I/817gWvybxxL._AC_SL1500_.jpg"
    ]
    criteria = (
        "Pick the best hero image for an ecommerce PDP. "
        "Must be sharp, subject centered, neutral background, even lighting, "
        "and at least ~1200 px on the short edge. Avoid watermarks/borders."
    )

    result = choose_top_image_simple(urls, criteria)
    # print(json.dumps(result, indent=2))
    # print("\nTOP IMAGE:", result["winner"])
