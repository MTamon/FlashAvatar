"""SMIRK-based FLAME tracker, an alternative to metrical-tracker.

Runs MTamon/smirk (release/cuda128) over every frame under
`dataset/<idname>/raw/imgs/` and writes FlashAvatar-compatible `.frame`
files under `metrical-tracker/output/<idname>/checkpoint_raw/` — same
downstream contract as the metrical-tracker path, so
`preprocess finalize` doesn't know or care which tracker produced them.

SMIRK is more robust than metrical-tracker against large head rotations
and motion blur, at the cost of a weak-perspective camera model we
approximate as perspective with a large focal length.

See docs/smirk.md for the full design + feature compatibility matrix.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# --- Public entry ----------------------------------------------------------

@dataclass
class SmirkConfig:
    smirk_root: Path                 # external/smirk
    checkpoint: Path                 # .../pretrained_models/SMIRK_em1.pt
    device: str = "cuda"
    crop_scale: float = 1.4          # matches SMIRK demo_video crop
    crop_size: int = 224             # SMIRK encoder input size
    focal_px: float = 5000.0         # synthesized K focal, in full-frame px
    shape_frames: int = 150          # canonicalize shape over this many frames
    batch_size: int = 8
    overwrite: bool = False
    # Eye pose source: "zero" (default, identity eye pose for every frame —
    # matches the pre-eye-pose behaviour) or "blendshapes" (derives per-eye
    # yaw/pitch from MediaPipe Face Landmarker ARKit blendshape coefficients
    # so the trained avatar tracks gaze).
    eye_mode: str = "zero"


def run(cfg: SmirkConfig, raw_imgs: Path, ckpt_out: Path,
        verify_dir: Path | None = None) -> int:
    """Drive the whole tracker: detect+crop+encode+convert+write.

    Returns the number of `.frame` files written.
    """
    # Lazy import: these modules live inside external/smirk and we don't want
    # to force users who never touch SMIRK to have it installed.
    _ensure_on_pythonpath(cfg.smirk_root)
    from smirk_runtime import SmirkRunner  # local module, see below

    runner = SmirkRunner(cfg)
    frames = sorted(raw_imgs.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames under {raw_imgs}")

    ckpt_out.mkdir(parents=True, exist_ok=True)

    # Pass 1: encode every frame, collect per-frame payloads.
    from preprocess.smirk_convert import (
        FramePayload, canonicalize_shape, to_flashavatar_frame,
    )
    payloads: list[FramePayload] = []
    img_size: tuple[int, int] | None = None
    for i in tqdm(range(0, len(frames), cfg.batch_size), desc="smirk/encode"):
        batch = frames[i:i + cfg.batch_size]
        imgs = [np.array(Image.open(p).convert("RGB")) for p in batch]
        if img_size is None:
            h, w = imgs[0].shape[:2]
            img_size = (w, h)
        results = runner.encode_batch(imgs)
        for j, r in enumerate(results):
            payloads.append(FramePayload(
                idx=i + j, src_path=batch[j], result=r,
            ))

    # Pass 2: canonicalize FLAME shape over a subset of frames; write.
    shape = canonicalize_shape(payloads, cfg.shape_frames)
    n_written = 0
    for p in tqdm(payloads, desc="smirk/write"):
        dst = ckpt_out / f"{p.idx:05d}.frame"
        if dst.exists() and not cfg.overwrite:
            continue
        frame_dict = to_flashavatar_frame(
            p.result, shape=shape, img_size=img_size,
            focal_px=cfg.focal_px, eye_mode=cfg.eye_mode,
        )
        torch.save(frame_dict, dst)
        n_written += 1

    if verify_dir is not None:
        from preprocess.smirk_verify import dump_verification
        dump_verification(payloads, shape, img_size, cfg, verify_dir)

    return n_written


def _ensure_on_pythonpath(smirk_root: Path) -> None:
    """Add `external/smirk` AND this package's `preprocess/_smirk/` to sys.path.

    The SMIRK repo ships as a loose `src/` tree; we put the smirk root on the
    path so `from src.smirk_encoder import SmirkEncoder` works. The runtime /
    convert / verify helpers live under this package so users don't have to
    install FlashAvatar to run the tracker.
    """
    smirk_root = Path(smirk_root).resolve()
    if not (smirk_root / "src" / "smirk_encoder.py").is_file():
        raise FileNotFoundError(
            f"SMIRK not found at {smirk_root} (missing src/smirk_encoder.py).\n"
            f"Run scripts/setup_smirk.sh to clone + install it."
        )
    p = str(smirk_root)
    if p not in sys.path:
        sys.path.insert(0, p)
    # Also expose our private runtime module.
    here = Path(__file__).resolve().parent / "_smirk"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


# --- Path helpers used by the CLI wrappers --------------------------------

def checkpoint_raw_dir(repo_root: Path, idname: str) -> Path:
    return (Path(repo_root) / "metrical-tracker" / "output" / idname
            / "checkpoint_raw")


def raw_imgs_dir(repo_root: Path, idname: str) -> Path:
    return Path(repo_root) / "dataset" / idname / "raw" / "imgs"
