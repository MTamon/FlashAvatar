"""Metrical-tracker integration.

The tracker itself lives outside this repository and typically requires its
own environment. This module provides a thin subprocess wrapper plus small
helpers to verify / locate the resulting `.frame` checkpoints.

If `metrical-tracker/output/<id>/checkpoint/` already contains .frame files
(matching the expected count), the tracker step is skipped.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def checkpoint_dir(repo_root: Path, idname: str) -> Path:
    return Path(repo_root) / "metrical-tracker" / "output" / idname / "checkpoint"


def count_frames(ckpt_dir: Path) -> int:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return 0
    return len(list(ckpt_dir.glob("*.frame")))


def run_tracker(tracker_cmd: str, imgs_dir: Path, ckpt_dir: Path,
                idname: str) -> None:
    """Invoke an external metrical-tracker command.

    `tracker_cmd` is a shell command template. These placeholders are expanded
    before execution:
        {imgs_dir} {ckpt_dir} {idname}
    e.g. `tracker_cmd =
        "python /opt/metrical-tracker/tracker.py --input_dir {imgs_dir} "
        "--output_dir metrical-tracker/output/{idname}"`
    """
    cmd_str = tracker_cmd.format(
        imgs_dir=str(imgs_dir), ckpt_dir=str(ckpt_dir), idname=idname,
    )
    subprocess.run(shlex.split(cmd_str), check=True)


def verify_or_hint(ckpt_dir: Path, expected: int) -> None:
    found = count_frames(ckpt_dir)
    if found == 0:
        raise FileNotFoundError(
            f"metrical-tracker output not found at {ckpt_dir}.\n"
            f"Run the tracker manually (see docs/preprocessing.md) or pass "
            f"--tracker-cmd to invoke it from this script."
        )
    if expected and found < expected:
        print(f"[tracker] warning: found {found} .frame files, expected "
              f"{expected}; training will use min(count, frames).")
