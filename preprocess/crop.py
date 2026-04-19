"""Final formatting stage: square crop + resize, with camera K adjustment.

This is the only stage that interprets `--crop/--no-crop`. Upstream stages
(parsing, matting, tracker) always run at native resolution; this stage takes
their outputs together with the tracker's `.frame` files, applies a single
square bbox and resize, and writes matching 4-tuples into the final dataset
directory so `scene.Scene_mica` can consume them directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


@dataclass
class Bbox:
    """Axis-aligned square bbox in source image coordinates.

    Attributes are all in pixels on the source frame. `size` is the square
    side length (== w == h)."""
    x0: int
    y0: int
    size: int

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size


def _union_head_bbox(parsing_dir: Path, stride: int = 50,
                     pad: float = 0.15) -> tuple[int, int, int, int] | None:
    """Compute the (x0, y0, x1, y1) bbox that covers *_neckhead.png pixels
    across a striding sample of frames, expanded by `pad` (fractional).

    Returns None if no neckhead masks are present or they are all empty.
    """
    masks = sorted(parsing_dir.glob("*_neckhead.png"))[::max(stride, 1)]
    if not masks:
        return None
    xs0, ys0, xs1, ys1 = [], [], [], []
    for m in masks:
        arr = np.array(Image.open(m))
        ys, xs = np.where(arr > 0)
        if xs.size == 0:
            continue
        xs0.append(int(xs.min()))
        ys0.append(int(ys.min()))
        xs1.append(int(xs.max()))
        ys1.append(int(ys.max()))
    if not xs0:
        return None
    x0, y0 = min(xs0), min(ys0)
    x1, y1 = max(xs1), max(ys1)
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w / 2, y0 + h / 2
    side = max(w, h) * (1.0 + pad)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = int(round(cx + side / 2))
    y1 = int(round(cy + side / 2))
    return x0, y0, x1, y1


def compute_bbox(parsing_dir: Path | None, image_size: tuple[int, int],
                 crop: bool) -> Bbox:
    """Pick the square bbox to apply to every downstream output.

    `image_size` is (W, H) of the source frames.
    """
    W, H = image_size
    if crop:
        if parsing_dir is None:
            raise ValueError("crop=True requires a parsing_dir to derive the bbox")
        rect = _union_head_bbox(parsing_dir)
        if rect is None:
            raise RuntimeError(
                f"no non-empty *_neckhead.png in {parsing_dir}; cannot derive crop")
        x0, y0, x1, y1 = rect
    else:
        side = min(W, H)
        x0 = (W - side) // 2
        y0 = (H - side) // 2
        x1 = x0 + side
        y1 = y0 + side

    size = max(x1 - x0, y1 - y0)
    x0 = max(0, min(W - size, x0))
    y0 = max(0, min(H - size, y0))
    if size > min(W, H):
        raise RuntimeError(
            f"bbox side {size} exceeds shortest image dim {min(W, H)}; "
            f"the frame cannot contain a square of that size")
    return Bbox(x0=int(x0), y0=int(y0), size=int(size))


def _crop_and_resize(img: Image.Image, bbox: Bbox, size: int,
                     resample) -> Image.Image:
    cropped = img.crop((bbox.x0, bbox.y0, bbox.x1, bbox.y1))
    if cropped.size != (size, size):
        cropped = cropped.resize((size, size), resample=resample)
    return cropped


def apply_to_images(src: Path, dst: Path, bbox: Bbox, size: int,
                    resample=Image.BILINEAR, overwrite: bool = False) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    frames = sorted(src.glob("*.jpg"))
    for frame in tqdm(frames, desc=f"crop/{src.name}"):
        out = dst / frame.name
        if not overwrite and out.exists():
            continue
        _crop_and_resize(Image.open(frame), bbox, size, resample).save(
            out, quality=95)
    return len(frames)


def apply_to_parsing(src: Path, dst: Path, bbox: Bbox, size: int,
                     overwrite: bool = False) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    masks = sorted(src.glob("*.png"))
    for mask in tqdm(masks, desc="crop/parsing"):
        out = dst / mask.name
        if not overwrite and out.exists():
            continue
        _crop_and_resize(Image.open(mask), bbox, size, Image.NEAREST).save(out)
    return len(masks)


def apply_to_alpha(src: Path, dst: Path, bbox: Bbox, size: int,
                   overwrite: bool = False) -> int:
    return apply_to_images(src, dst, bbox, size,
                           resample=Image.BILINEAR, overwrite=overwrite)


def adjusted_K(K: np.ndarray | torch.Tensor, bbox: Bbox, size: int):
    """Return a new K accounting for `(bbox -> size x size)`.

    Works for either numpy arrays or torch tensors of shape (..., 3, 3).
    """
    s = size / bbox.size
    if isinstance(K, torch.Tensor):
        new = K.clone().to(torch.float64)
        new[..., 0, 2] = (new[..., 0, 2] - bbox.x0) * s
        new[..., 1, 2] = (new[..., 1, 2] - bbox.y0) * s
        new[..., 0, 0] = new[..., 0, 0] * s
        new[..., 1, 1] = new[..., 1, 1] * s
        return new.to(K.dtype)
    arr = np.array(K, dtype=np.float64, copy=True)
    arr[..., 0, 2] = (arr[..., 0, 2] - bbox.x0) * s
    arr[..., 1, 2] = (arr[..., 1, 2] - bbox.y0) * s
    arr[..., 0, 0] = arr[..., 0, 0] * s
    arr[..., 1, 1] = arr[..., 1, 1] * s
    return arr


def adjust_frame_files(src_dir: Path, dst_dir: Path, bbox: Bbox, size: int) -> int:
    """Rewrite tracker `.frame` files with updated K and img_size.

    `src_dir` / `dst_dir` may be the same path; if different, `dst_dir` is
    created and populated with adjusted copies.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(src_dir.glob("*.frame"))
    if not frames:
        raise FileNotFoundError(f"no *.frame files under {src_dir}")

    for fp in tqdm(frames, desc="crop/.frame"):
        payload = torch.load(fp, map_location="cpu", weights_only=False)
        K = payload["opencv"]["K"]
        payload["opencv"]["K"] = adjusted_K(K, bbox, size)
        payload["img_size"] = (size, size)
        torch.save(payload, dst_dir / fp.name)
    return len(frames)
