"""Portrait alpha matting via RobustVideoMatting (RVM).

RVM is loaded through torch.hub (PeterL1n/RobustVideoMatting). Inference is
sequential across the frame sequence so the recurrent state is preserved.
The resulting alpha is written as grayscale JPEG at the native frame
resolution, matching the pipeline convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _load_rvm(variant: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(
        "PeterL1n/RobustVideoMatting", variant, trust_repo=True
    )
    return model.to(device).eval()


def run_matting(imgs_dir: Path, out_dir: Path, variant: str = "mobilenetv3",
                downsample_ratio: float | None = None, device: str = "cuda",
                overwrite: bool = False) -> int:
    """Emit alpha/XXXXX.jpg for every frame in `imgs_dir`.

    `variant` is one of 'mobilenetv3' or 'resnet50'. `downsample_ratio`
    controls RVM's internal downsample; None picks a sane default by
    resolution (per RVM docs).
    """
    imgs_dir = Path(imgs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(imgs_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no *.jpg frames under {imgs_dir}")

    dev = torch.device(device)
    model = _load_rvm(variant, dev)

    rec = [None] * 4  # recurrent state
    done = 0

    for frame in tqdm(frames, desc="matting"):
        out_path = out_dir / f"{frame.stem}.jpg"
        if not overwrite and out_path.exists():
            done += 1
            continue

        img = Image.open(frame).convert("RGB")
        w, h = img.size
        x = torch.from_numpy(np.asarray(img)).float().div(255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(dev)  # (1, 3, H, W)

        if downsample_ratio is None:
            longest = max(h, w)
            dr = 0.25 if longest >= 1920 else (0.4 if longest >= 1280 else 1.0)
        else:
            dr = downsample_ratio

        with torch.no_grad():
            fgr, pha, *rec = model(x, *rec, downsample_ratio=dr)

        alpha = (pha[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(alpha, mode="L").save(out_path, quality=95)
        done += 1

    return done
