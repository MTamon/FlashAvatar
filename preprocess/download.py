"""Fetch pretrained weights for preprocessing models.

BiSeNet (face-parsing): the original checkpoint `79999_iter.pth` ships via
Google Drive from zllrunning/face-parsing.PyTorch. There is no stable direct
URL, so we delegate to `gdown` when available and otherwise instruct the user
to place the file manually.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

BISENET_GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
BISENET_FILENAME = "79999_iter.pth"


def default_weights_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "preprocess_weights"


def ensure_bisenet_weights(dest: Path | None = None) -> Path:
    """Return the path to a usable BiSeNet checkpoint.

    If the file is already present, return it. Otherwise try `gdown`.
    Raise FileNotFoundError with a helpful message if retrieval fails.
    """
    dest_dir = Path(dest) if dest else default_weights_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    ckpt = dest_dir / BISENET_FILENAME
    if ckpt.is_file() and ckpt.stat().st_size > 0:
        return ckpt

    if shutil.which("gdown") is not None:
        subprocess.run(
            ["gdown", "--id", BISENET_GDRIVE_ID, "-O", str(ckpt)],
            check=True,
        )
        if ckpt.is_file() and ckpt.stat().st_size > 0:
            return ckpt

    raise FileNotFoundError(
        f"BiSeNet checkpoint not found at {ckpt}.\n"
        f"Download manually from face-parsing.PyTorch (Google Drive id "
        f"{BISENET_GDRIVE_ID}) and place it at that path, or install gdown "
        f"(`pip install gdown`) and re-run."
    )
