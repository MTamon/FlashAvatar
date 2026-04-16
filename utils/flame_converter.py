"""Convert face-tracker outputs to FlashAvatar's .frame format.

Supported trackers: DECA, EMOCA, SMIRK, SPARK.

Design goals
------------
- **Real-time capable**: every function is stateless and operates on a
  single frame.  No file I/O, no accumulated statistics.
- **Deterministic**: all conversions are closed-form math (no learned
  components).
- **Minimal dependencies**: only torch and (optionally) pytorch3d.

Usage
-----
In-memory (real-time pipeline)::

    from utils.flame_converter import FlameConverter

    converter = FlameConverter(tracker="deca")
    frame_dict = converter.convert(deca_output, img_size=(512, 512))
    # frame_dict is ready for DeformModel.decode()

Batch file generation (offline)::

    python utils/flame_converter.py \\
        --tracker deca \\
        --input_dir /path/to/deca_outputs \\
        --output_dir metrical-tracker/output/myface/checkpoint \\
        --img_size 512,512

See ``docs/conversion_rationale.md`` for the mathematical justification
of each conversion step.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Rotation utilities (self-contained to avoid hard pytorch3d dependency)
# ---------------------------------------------------------------------------

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (B, 3) to rotation matrix (B, 3, 3).

    Uses Rodrigues' formula.  Equivalent to
    ``pytorch3d.transforms.axis_angle_to_matrix`` but dependency-free.
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    k = axis_angle / angle  # unit axis (B, 3)

    K = torch.zeros(*k.shape[:-1], 3, 3, device=k.device, dtype=k.dtype)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]

    sin = torch.sin(angle).unsqueeze(-1)
    cos = torch.cos(angle).unsqueeze(-1)
    I = torch.eye(3, device=k.device, dtype=k.dtype).expand_as(K)
    R = I + sin * K + (1 - cos) * (K @ K)
    return R


def axis_angle_to_rot6d(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (B, 3) to 6D continuous rotation (B, 6).

    Takes the first two columns of the rotation matrix and flattens.
    See Zhou et al., CVPR 2019.
    """
    R = axis_angle_to_matrix(aa)          # (B, 3, 3)
    return R[:, :, :2].reshape(-1, 6)     # (B, 6)


def identity_rot6d(batch_size: int = 1,
                   device: torch.device | str = "cpu",
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the 6D representation of the identity rotation (B, 6)."""
    rot = torch.zeros(batch_size, 6, device=device, dtype=dtype)
    rot[:, 0] = 1.0  # col0 = [1, 0, 0]
    rot[:, 4] = 1.0  # col1 = [0, 1, 0]
    return rot


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

def weak_perspective_to_full(
    cam: torch.Tensor,
    img_w: int,
    img_h: int,
    focal_scale: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert DECA-style weak-perspective camera to OpenCV (K, R, t).

    Parameters
    ----------
    cam : (B, 3) — [scale, tx, ty]
    img_w, img_h : image resolution
    focal_scale : heuristic multiplier for focal length estimation

    Returns
    -------
    K : (B, 3, 3) intrinsic matrix
    R : (B, 3, 3) rotation (identity — DECA encodes pose in FLAME params)
    t : (B, 3) translation
    """
    B = cam.shape[0]
    device, dtype = cam.device, cam.dtype
    scale = cam[:, 0]
    tx = cam[:, 1]
    ty = cam[:, 2]

    focal = focal_scale * img_w * scale
    cx = torch.full((B,), img_w / 2.0, device=device, dtype=dtype)
    cy = torch.full((B,), img_h / 2.0, device=device, dtype=dtype)

    K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    K[:, 0, 0] = focal
    K[:, 1, 1] = focal
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0

    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    t = torch.zeros(B, 3, device=device, dtype=dtype)
    t[:, 0] = tx
    t[:, 1] = ty
    t[:, 2] = 1.0 / scale.clamp(min=1e-8)

    return K, R, t


# ---------------------------------------------------------------------------
# Tracker-specific key mappings
# ---------------------------------------------------------------------------

@dataclass
class TrackerConfig:
    """Describes how a specific tracker stores FLAME parameters."""
    name: str
    shape_key: str = "shape"
    exp_key: str = "exp"
    pose_key: str = "pose"
    cam_key: str = "cam"
    shape_dim: int = 100
    exp_dim: int = 50
    jaw_slice: tuple[int, int] = (3, 6)
    global_pose_slice: tuple[int, int] = (0, 3)
    has_eyes: bool = False
    eyes_key: str = ""
    has_eyelids: bool = False
    eyelids_key: str = ""


TRACKER_CONFIGS: dict[str, TrackerConfig] = {
    "deca": TrackerConfig(name="DECA"),
    "emoca": TrackerConfig(name="EMOCA"),
    "smirk": TrackerConfig(name="SMIRK"),
    "spark": TrackerConfig(
        name="SPARK",
        shape_dim=300,
        exp_dim=50,
    ),
}


# ---------------------------------------------------------------------------
# FlameConverter
# ---------------------------------------------------------------------------

class FlameConverter:
    """Stateless converter from tracker output dicts to FlashAvatar format.

    Parameters
    ----------
    tracker : str
        One of "deca", "emoca", "smirk", "spark".
    device : str or torch.device
        Target device for output tensors.
    """

    FLASH_SHAPE_DIM = 300
    FLASH_EXP_DIM = 100
    FLASH_JAW_DIM = 6
    FLASH_EYES_DIM = 12
    FLASH_EYELIDS_DIM = 2

    def __init__(self, tracker: str = "deca", device: str = "cpu"):
        tracker = tracker.lower()
        if tracker not in TRACKER_CONFIGS:
            raise ValueError(
                f"Unknown tracker '{tracker}'. "
                f"Supported: {list(TRACKER_CONFIGS)}"
            )
        self.cfg = TRACKER_CONFIGS[tracker]
        self.device = torch.device(device)

    # ----- public API -----

    def convert(
        self,
        tracker_output: dict[str, torch.Tensor],
        img_size: tuple[int, int] = (512, 512),
        camera_K: Optional[torch.Tensor] = None,
        camera_R: Optional[torch.Tensor] = None,
        camera_t: Optional[torch.Tensor] = None,
    ) -> dict:
        """Convert a single-frame tracker output to FlashAvatar .frame dict.

        Parameters
        ----------
        tracker_output : dict
            Raw output dict from the tracker (e.g. DECA codedict).
        img_size : (width, height)
        camera_K, camera_R, camera_t : optional
            If provided, used as-is instead of converting the tracker's
            weak-perspective camera.  Preferred in real-time pipelines
            where an actual camera calibration is available.

        Returns
        -------
        dict matching the .frame file format expected by
        ``scene/__init__.py`` / ``Scene_mica``.
        """
        shape = self._convert_shape(tracker_output)
        exp = self._convert_expression(tracker_output)
        jaw = self._convert_jaw(tracker_output)
        eyes = self._convert_eyes(tracker_output)
        eyelids = self._convert_eyelids(tracker_output)

        if camera_K is not None and camera_R is not None and camera_t is not None:
            K = camera_K.to(self.device)
            R = camera_R.to(self.device)
            t = camera_t.to(self.device)
        else:
            K, R, t = self._convert_camera(tracker_output, img_size)

        if K.dim() == 2:
            K = K.unsqueeze(0)
        if R.dim() == 2:
            R = R.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(0)

        return {
            "flame": {
                "shape": shape,
                "exp": exp,
                "jaw": jaw,
                "eyes": eyes,
                "eyelids": eyelids,
            },
            "opencv": {
                "K": K,
                "R": R,
                "t": t,
            },
            "img_size": img_size,
        }

    def convert_flame_params_only(
        self,
        tracker_output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert only FLAME parameters (no camera). For real-time use.

        Returns
        -------
        codedict compatible with ``DeformModel.decode()``:
            shape (1, 300), expr (1, 100), jaw_pose (1, 6),
            eyes_pose (1, 12), eyelids (1, 2).
        """
        return {
            "shape": self._convert_shape(tracker_output),
            "expr": self._convert_expression(tracker_output),
            "jaw_pose": self._convert_jaw(tracker_output),
            "eyes_pose": self._convert_eyes(tracker_output),
            "eyelids": self._convert_eyelids(tracker_output),
        }

    # ----- internal conversions -----

    def _get(self, d: dict, key: str) -> torch.Tensor:
        return d[key].to(self.device).float()

    def _convert_shape(self, d: dict) -> torch.Tensor:
        shape = self._get(d, self.cfg.shape_key)
        if shape.dim() == 1:
            shape = shape.unsqueeze(0)
        pad = self.FLASH_SHAPE_DIM - shape.shape[-1]
        if pad > 0:
            shape = F.pad(shape, (0, pad), value=0.0)
        return shape[:, :self.FLASH_SHAPE_DIM]

    def _convert_expression(self, d: dict) -> torch.Tensor:
        exp = self._get(d, self.cfg.exp_key)
        if exp.dim() == 1:
            exp = exp.unsqueeze(0)
        pad = self.FLASH_EXP_DIM - exp.shape[-1]
        if pad > 0:
            exp = F.pad(exp, (0, pad), value=0.0)
        return exp[:, :self.FLASH_EXP_DIM]

    def _convert_jaw(self, d: dict) -> torch.Tensor:
        pose = self._get(d, self.cfg.pose_key)
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
        s, e = self.cfg.jaw_slice
        jaw_aa = pose[:, s:e]                    # (B, 3) axis-angle
        return axis_angle_to_rot6d(jaw_aa)        # (B, 6)

    def _convert_eyes(self, d: dict) -> torch.Tensor:
        if self.cfg.has_eyes and self.cfg.eyes_key in d:
            eyes_raw = self._get(d, self.cfg.eyes_key)
            if eyes_raw.dim() == 1:
                eyes_raw = eyes_raw.unsqueeze(0)
            if eyes_raw.shape[-1] == 6:
                left_aa = eyes_raw[:, :3]
                right_aa = eyes_raw[:, 3:6]
                left_6d = axis_angle_to_rot6d(left_aa)
                right_6d = axis_angle_to_rot6d(right_aa)
                return torch.cat([left_6d, right_6d], dim=-1)
            return eyes_raw[:, :self.FLASH_EYES_DIM]
        B = 1
        return identity_rot6d(B, self.device).repeat(1, 2)  # (1, 12)

    def _convert_eyelids(self, d: dict) -> torch.Tensor:
        if self.cfg.has_eyelids and self.cfg.eyelids_key in d:
            eyelids = self._get(d, self.cfg.eyelids_key)
            if eyelids.dim() == 1:
                eyelids = eyelids.unsqueeze(0)
            return eyelids[:, :self.FLASH_EYELIDS_DIM]
        return torch.zeros(1, self.FLASH_EYELIDS_DIM,
                           device=self.device, dtype=torch.float32)

    def _convert_camera(
        self, d: dict, img_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.cam_key in d:
            cam = self._get(d, self.cfg.cam_key)
            if cam.dim() == 1:
                cam = cam.unsqueeze(0)
            return weak_perspective_to_full(cam, img_size[0], img_size[1])
        K = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0)
        R = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0)
        t = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        return K, R, t


# ---------------------------------------------------------------------------
# CLI: batch-convert saved tracker outputs to .frame files
# ---------------------------------------------------------------------------

def _load_tracker_output(path: Path) -> dict[str, torch.Tensor]:
    """Load a single tracker output file (pickle / .pt / .pth)."""
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected type in {path}: {type(data)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-convert tracker outputs to FlashAvatar .frame files."
    )
    parser.add_argument(
        "--tracker", required=True,
        choices=list(TRACKER_CONFIGS),
        help="Source tracker name.",
    )
    parser.add_argument(
        "--input_dir", required=True, type=Path,
        help="Directory containing tracker output files (.pkl / .pt).",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Output directory for .frame files "
             "(typically metrical-tracker/output/<id>/checkpoint).",
    )
    parser.add_argument(
        "--img_size", default="512,512",
        help="Image size as W,H (default: 512,512).",
    )
    parser.add_argument(
        "--ext", default=".pkl,.pt,.pth",
        help="Comma-separated file extensions to scan (default: .pkl,.pt,.pth).",
    )
    args = parser.parse_args()

    w, h = (int(x) for x in args.img_size.split(","))
    exts = set(args.ext.split(","))
    converter = FlameConverter(tracker=args.tracker)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix in exts
    )

    if not input_files:
        print(f"No files with extensions {exts} found in {args.input_dir}")
        sys.exit(1)

    print(f"Converting {len(input_files)} files ({args.tracker} → .frame)")

    for idx, path in enumerate(input_files):
        tracker_output = _load_tracker_output(path)
        frame_dict = converter.convert(tracker_output, img_size=(w, h))
        out_path = args.output_dir / f"{idx:05d}.frame"
        torch.save(frame_dict, str(out_path))

        if (idx + 1) % 100 == 0 or idx == len(input_files) - 1:
            print(f"  [{idx + 1}/{len(input_files)}] → {out_path.name}")

    print(f"Done. {len(input_files)} .frame files written to {args.output_dir}")


if __name__ == "__main__":
    main()
