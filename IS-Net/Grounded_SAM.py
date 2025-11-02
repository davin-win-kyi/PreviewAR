#!/usr/bin/env python3
import os
import json
import replicate
from typing import Any, Dict, List

"""
Prereqs:
  pip install replicate

Auth:
  export REPLICATE_API_TOKEN=YOUR_TOKEN
"""

MODEL = "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c"

if __name__ == "__main__":
    # Example usage — replace with your actual values:
    IMAGE = "https://m.media-amazon.com/images/I/81U1IoVP6YL._AC_SL1500_.jpg"
    MASK_PROMPT = "sectional,sofa,couch,pillow"
    NEGATIVE_MASK = "sky"

    output = replicate.run(
        "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
        input={
            "image": "https://m.media-amazon.com/images/I/81U1IoVP6YL._AC_SL1500_.jpg",
            "mask_prompt": "sectional,sofa,couch,pillow",
            "adjustment_factor": -15,
            "negative_mask_prompt": "sky"
        }
    )

    # The schananas/grounded_sam model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    for item in output:
        # https://replicate.com/schananas/grounded_sam/api#output-schema
        print(item)
