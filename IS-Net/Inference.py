from __future__ import annotations

import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import *


import os
from pathlib import Path
import requests

from best_image_agent import get_best_image

# parameters for downloading and shrinking an image

from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image, ImageOps

import io as bytesio


def download_image(
    url: str,
    out_path: str | Path,
    timeout: int = 20,
    *,
    # Shrink options (pick one style):
    max_size: Optional[Tuple[int, int]] = None,   # (max_width, max_height)
    max_side: Optional[int] = None,               # longest side limit (e.g., 1200)
    scale: Optional[float] = 0.5,                # e.g., 0.5 to shrink to 50%
    # Encoding options:
    out_format: Optional[str] = None,             # e.g., "JPEG", "PNG", "WEBP" (auto by filename if None)
    quality: int = 85,                            # used for JPEG/WEBP
) -> Path:
    """
    Download an image from `url`, shrink it, and save to `out_path`.

    Shrink rules (applied in this priority if provided):
      1) `scale`  -> scale by factor
      2) `max_side` -> limit the longest side to this value
      3) `max_size` -> fit inside (max_width, max_height)

    - Keeps aspect ratio and won't upscale smaller images.
    - Applies EXIF orientation.
    - Uses high-quality LANCZOS resampling.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "image" not in ctype:
            raise ValueError(f"URL did not return an image (Content-Type={ctype!r})")
        # Load the response body into PIL without writing original to disk
        raw = bytesio.BytesIO(r.content)

    with Image.open(raw) as im:
        # Normalize orientation (e.g., camera-rotated JPEGs)
        im = ImageOps.exif_transpose(im)

        # Figure out target size
        w, h = im.size
        new_w, new_h = w, h

        if scale is not None and scale > 0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
        elif max_side is not None and max_side > 0:
            # Limit the longest side
            if max(w, h) > max_side:
                if w >= h:
                    new_w = max_side
                    new_h = int(round(h * (max_side / w)))
                else:
                    new_h = max_side
                    new_w = int(round(w * (max_side / h)))
        elif max_size is not None:
            max_w, max_h = max_size
            # Fit inside the given box
            ratio = min(max_w / w, max_h / h)
            if ratio < 1.0:
                new_w = int(round(w * ratio))
                new_h = int(round(h * ratio))

        # Resize only if shrinking
        if (new_w, new_h) != (w, h):
            im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Decide output format (by explicit arg, else by file extension, else original)
        fmt = (
            out_format
            or (out_path.suffix[1:].upper() if out_path.suffix else None)
            or getattr(im, "format", None)
            or "JPEG"
        )
        fmt = "JPG" if fmt.upper() == "JPEG" else fmt  # allow ".jpg" nicely

        # Handle alpha if saving to JPEG/JPG (fill with white)
        save_kwargs = {}
        if fmt.upper() in {"JPG", "JPEG"}:
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1] if im.mode != "P" else im.convert("RGBA").split()[-1])
                im = bg
            else:
                im = im.convert("RGB")

            save_kwargs.update(
                dict(quality=quality, optimize=True, progressive=True, subsampling="4:2:0")
            )

        elif fmt.upper() == "PNG":
            # Let PNG be palette/rgba as-is; optimize flag helps reduce size
            save_kwargs.update(dict(optimize=True))

        elif fmt.upper() == "WEBP":
            # Lossy by default; set quality; preserves alpha
            save_kwargs.update(dict(quality=quality, method=6))

        # Finally save
        im.save(out_path, format=None if fmt is None else fmt, **save_kwargs)

    return out_path




def dis(json_path):

    # before calling the rest, call the method that 
    # will save the json for a certain url

    dataset_path="../demo_datasets/your_dataset/sofa2.jpg"  #Your dataset path
    model_path="../saved_models/IS-Net/isnet-general-use.pth"  # the model path
    result_path="../demo_datasets/your_dataset_result/test"  #The folder path that you want to save the results
    
    url = get_best_image(json_path)
    
    dataset_path = download_image(url, "best_image.png")

    
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net.eval()
    im_list = [dataset_path] if os.path.isfile(dataset_path) else glob(os.path.join(dataset_path, '*'))
    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("im_path: ", im_path)
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear", align_corners=False).type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            
            result = net(image)
            result = torch.squeeze(F.interpolate(result[0][0],im_shp,mode='bilinear', align_corners=False))
            
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)

            alpha_mask = result.cpu().numpy()[:, :, np.newaxis]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            if im.shape[2] > 3:
                im = im[:,:,:3]

            white_bg = np.full_like(im, 255, dtype=np.uint8)
            final_image = (im * alpha_mask + white_bg * (1 - alpha_mask)).astype(np.uint8)

            im_name, _ = os.path.splitext(os.path.basename(im_path))
            io.imsave(os.path.join(result_path,im_name+".png"), final_image)

if __name__ == "__main__":
    dis()