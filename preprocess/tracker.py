"""Helpers for consuming metrical-tracker output.

The tracker itself runs in a separate conda environment (see
`scripts/setup_metrical_tracker.sh`). This module only provides utilities
for locating and validating its output from within the FlashAvatar env.
A small subprocess wrapper (`run_tracker`) is kept for callers that want
to shell out to the tracker's `python tracker.py ...` entry point.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def checkpoint_dir(repo_root: Path, idname: str) -> Path:
    return Path(repo_root) / "metrical-tracker" / "output" / idname / "checkpoint"


def raw_checkpoint_dir(repo_root: Path, idname: str) -> Path:
    return Path(repo_root) / "metrical-tracker" / "output" / idname / "checkpoint_raw"


def count_frames(ckpt_dir: Path) -> int:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return 0
    return len(list(ckpt_dir.glob("*.frame")))


def run_tracker(tracker_cmd: str, imgs_dir: Path, ckpt_dir: Path,
                idname: str) -> None:
    """Invoke an external metrical-tracker command (optional helper).

    `tracker_cmd` is a shell template. Placeholders `{imgs_dir}`,
    `{ckpt_dir}`, `{idname}` are substituted before execution.
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
            f"Set up the tracker env with:\n"
            f"    bash scripts/setup_metrical_tracker.sh\n"
            f"then run it (conda activate tracker; python tracker.py ...) or\n"
            f"    bash scripts/run_tracker.sh <idname>\n"
            f"before re-running `preprocess finalize`."
        )
    if expected and found < expected:
        print(f"[tracker] warning: found {found} .frame files, expected "
              f"{expected}; training will use min(count, frames).")
