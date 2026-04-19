"""Video -> JPEG frame sequence via ffmpeg.

Frames are written as 5-digit zero-padded JPEGs starting at 00001.jpg, which
matches the convention consumed by `scene.Scene_mica`.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_frames(video_path: Path, out_dir: Path, qscale: int = 2,
                   fps: float | None = None, overwrite: bool = False) -> int:
    """Extract frames from `video_path` into `out_dir` as 00001.jpg, 00002.jpg...

    Returns the number of frames written.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available on PATH")

    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            existing = sorted(out_dir.glob("*.jpg"))
            if existing:
                return len(existing)
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video_path)]
    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]
    cmd += ["-q:v", str(qscale), "-start_number", "1",
            str(out_dir / "%05d.jpg")]
    subprocess.run(cmd, check=True)

    return len(list(out_dir.glob("*.jpg")))
