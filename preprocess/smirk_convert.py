"""Convert SMIRK encoder outputs into FlashAvatar `.frame` payloads.

Handles the five awkward impedance mismatches between SMIRK and FlashAvatar:

1. Expression dim (50 vs 100)      -> zero-pad last 50
2. Jaw representation (aa vs 6D)   -> axis_angle_to_matrix -> matrix_to_rot6d
3. Eye pose                        -> SMIRK produces none; we derive it from
                                      MediaPipe Face Landmarker blendshapes
                                      (`eye-mode blendshapes`, default) or
                                      write identity (`eye-mode zero`).
4. Global pose (aa + OpenGL)       -> rot matrix, then diag(1,-1,-1) y/z flip
                                      for OpenCV (y-down, +z forward) convention
5. Camera (weak-persp in 224 crop) -> perspective K/R/t at FULL raw frame res
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

if TYPE_CHECKING:
    from preprocess._smirk.smirk_runtime import FrameResult


@dataclass
class FramePayload:
    idx: int
    src_path: Path
    result: "FrameResult"


def canonicalize_shape(payloads: list[FramePayload], n: int) -> np.ndarray:
    """Collapse per-frame SMIRK shape codes to a single identity.

    Uses the median over the first `n` high-detection frames (robust to the
    occasional misdetection in the early sequence). Falls back to the mean
    if fewer than `n` detections are available.
    """
    stack = []
    for p in payloads:
        if not p.result.detected:
            continue
        stack.append(p.result.shape_params)
        if len(stack) >= n:
            break
    if not stack:
        # No high-confidence frames — fall back to all of them.
        stack = [p.result.shape_params for p in payloads]
    arr = np.stack(stack, axis=0)
    return np.median(arr, axis=0).astype(np.float32)


def to_flashavatar_frame(
    r: "FrameResult",
    *,
    shape: np.ndarray,
    img_size: tuple[int, int],
    focal_px: float,
    eye_mode: str = "zero",
) -> dict:
    """Build the dict saved as `.frame` by `torch.save`.

    `img_size` is (W, H) of the full raw frame; `shape` is the canonicalized
    300-dim FLAME identity. `eye_mode` is one of:

      "zero"        (default) — write identity eye pose (legacy behaviour;
                                the trained avatar renders with static eyes).
      "blendshapes"           — derive eye_pose from MediaPipe Face Landmarker
                                ARKit-style blendshape coefficients captured
                                alongside each frame's detection. Enables
                                real gaze tracking.

    Falls back to identity whenever blendshapes are unavailable for a frame,
    so callers never need to special-case missing detections.
    """
    w, h = img_size

    # --- FLAME params --------------------------------------------------
    shape_100 = _pad_expression(r.expression_params, 100)   # (100,)
    jaw_6d = _axis_angle_to_rot6d(r.jaw_params)             # (6,)
    if eye_mode == "blendshapes":
        eyes_12 = eye_pose_6d_from_blendshapes(r.blendshapes)
    elif eye_mode == "zero":
        eyes_12 = _default_eye_pose_6d()
    else:
        raise ValueError(f"unknown eye_mode: {eye_mode!r}; "
                         "expected 'blendshapes' or 'zero'.")
    flame_dict = {
        "shape": torch.from_numpy(shape).float().unsqueeze(0),       # (1, 300)
        "exp": torch.from_numpy(shape_100).float().unsqueeze(0),     # (1, 100)
        "jaw": torch.from_numpy(jaw_6d).float().unsqueeze(0),        # (1, 6)
        "eyes": torch.from_numpy(eyes_12).float().unsqueeze(0),      # (1, 12)
        "eyelids": torch.from_numpy(
            np.clip(r.eyelid_params.astype(np.float32), 0.0, 1.0),
        ).unsqueeze(0),                                              # (1, 2)
    }

    # --- OpenCV camera (K, R, t) at full-frame resolution --------------
    K = _build_K(w, h, focal_px)
    R = _build_R(r.pose_params)
    t = _build_t(r.cam, r.bbox_center, r.bbox_size, w, h, focal_px)

    return {
        "flame": flame_dict,
        "opencv": {
            "K": torch.from_numpy(K).float().unsqueeze(0),   # (1, 3, 3)
            "R": torch.from_numpy(R).float().unsqueeze(0),   # (1, 3, 3)
            "t": torch.from_numpy(t).float().unsqueeze(0),   # (1, 3)
        },
        "img_size": (w, h),
        # Debug / provenance, not read by FlashAvatar:
        "_smirk": {
            "cam": r.cam.astype(np.float32),
            "tform": r.tform_matrix.astype(np.float32),
            "bbox_center": r.bbox_center.astype(np.float32),
            "bbox_size": float(r.bbox_size),
            "focal_px": float(focal_px),
            "detected": bool(r.detected),
        },
    }


# --- low-level building blocks -------------------------------------------

def _pad_expression(exp50: np.ndarray, target: int) -> np.ndarray:
    out = np.zeros((target,), dtype=np.float32)
    n = min(exp50.shape[0], target)
    out[:n] = exp50[:n]
    return out


def _axis_angle_to_rot6d(aa: np.ndarray) -> np.ndarray:
    aa_t = torch.from_numpy(aa.astype(np.float32)).unsqueeze(0)  # (1, 3)
    R = axis_angle_to_matrix(aa_t)                               # (1, 3, 3)
    return matrix_to_rotation_6d(R)[0].numpy()                   # (6,)


def _default_eye_pose_6d() -> np.ndarray:
    # Identity 6D per eye in pytorch3d's row convention (first two rows of
    # I_3: [1,0,0, 0,1,0]). `rotation_6d_to_matrix` reconstructs I exactly.
    eye = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    return np.concatenate([eye, eye], axis=0)


# ARKit blendshape names consumed by `eye_pose_6d_from_blendshapes`.
_ARKIT_EYE_KEYS = (
    "eyeLookUpLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
)

# Max per-axis eye rotation, in radians (~34deg). ARKit coefficients are
# already in [0, 1], so this scales the (down-up) / (in-out) difference
# directly into an axis-angle magnitude.
_EYE_MAX_RAD = 0.6


def eye_pose_6d_from_blendshapes(bs: dict | None) -> np.ndarray:
    """ARKit eye-look blendshapes -> FLAME eye_pose (12,) in pytorch3d rot6d.

    Mapping (per-eye axis-angle in the FLAME y-up frame):

        pitch = (eyeLookDown{L,R} - eyeLookUp{L,R}) * _EYE_MAX_RAD   # +X = down
        yaw   = (eyeLookInLeft    - eyeLookOutLeft) * _EYE_MAX_RAD   # +Y = subject-right
        yaw   = (eyeLookOutRight  - eyeLookInRight) * _EYE_MAX_RAD   # mirrored: +Y = subject-right
        aa    = [pitch, yaw, 0]

    Each axis-angle is converted to a 3x3 rotation via
    `pytorch3d.transforms.axis_angle_to_matrix` and then to 6D via
    `matrix_to_rotation_6d` (row-major convention, matching the
    `rotation_6d_to_matrix` call inside `flame/lbs.py`).

    Falls back to identity when `bs` is None or missing every ARKit eye key —
    so the caller doesn't need to special-case frames without a MediaPipe
    detection.
    """
    if not bs or not any(k in bs for k in _ARKIT_EYE_KEYS):
        return _default_eye_pose_6d()

    def g(k: str) -> float:
        return float(bs.get(k, 0.0))

    left_pitch = (g("eyeLookDownLeft") - g("eyeLookUpLeft")) * _EYE_MAX_RAD
    left_yaw = (g("eyeLookInLeft") - g("eyeLookOutLeft")) * _EYE_MAX_RAD
    right_pitch = (g("eyeLookDownRight") - g("eyeLookUpRight")) * _EYE_MAX_RAD
    right_yaw = (g("eyeLookOutRight") - g("eyeLookInRight")) * _EYE_MAX_RAD

    left_aa = np.array([left_pitch, left_yaw, 0.0], dtype=np.float32)
    right_aa = np.array([right_pitch, right_yaw, 0.0], dtype=np.float32)
    return np.concatenate(
        [_axis_angle_to_rot6d(left_aa), _axis_angle_to_rot6d(right_aa)],
        axis=0,
    )


def _build_K(w: int, h: int, f_px: float) -> np.ndarray:
    cx = w / 2.0
    cy = h / 2.0
    return np.array([
        [f_px, 0.0, cx],
        [0.0, f_px, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# OpenGL (y-up, -z forward) -> OpenCV (y-down, +z forward).
_GL_TO_CV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _build_R(pose_aa: np.ndarray) -> np.ndarray:
    """World-to-camera rotation. SMIRK's pose rotates the FLAME mesh in an
    OpenGL-ish frame (y-up) before the renderer flips y for rasterization.
    FlashAvatar's scene/__init__.py stores opencv.R as a world-to-camera
    rotation used by the Gaussian rasterizer (via getWorld2View2)."""
    aa_t = torch.from_numpy(pose_aa.astype(np.float64)).unsqueeze(0)
    R_gl = axis_angle_to_matrix(aa_t)[0].numpy().astype(np.float64)  # (3, 3)
    return _GL_TO_CV @ R_gl


def _build_t(cam: np.ndarray, bbox_center: np.ndarray, bbox_size: float,
             w: int, h: int, f_px: float) -> np.ndarray:
    """Back-project SMIRK's weak-perspective [s, tx, ty] + the MediaPipe
    bbox to an OpenCV translation in the FULL frame.

    This is an explicit bbox_center / bbox_size decomposition of what used
    to go through the crop's similarity `tform` matrix. Equivalent math,
    but each term carries a clear physical meaning — so jitter sources
    can be attributed unambiguously:

        u_full = bbox_center_x + SMIRK_offset_x
        v_full = bbox_center_y + SMIRK_offset_y
        Z      = f_px / apparent_face_scale_full_px

    Derivation:
      SMIRK's weak-perspective renders orthographically inside the 224
      crop: the FLAME origin lands at crop pixel
        u_crop = 112 + 112 * s * tx
        v_crop = 112 - 112 * s * ty         (y flipped to y-down)
      The crop covers a square of `bbox_size` full-frame pixels centred on
      `bbox_center`. The crop-to-full pixel scale is therefore
        k = bbox_size / (CROP_SIZE - 1)     (~= bbox_size / 223)
      i.e. one crop pixel spans `k` full-frame pixels. The FLAME origin's
      full-frame pixel is then
        u_full = bbox_center_x + (u_crop - HALF) * k
               = bbox_center_x + 112 * s * tx * k
               ~~ bbox_center_x + (s * tx) * (bbox_size / 2)

      Depth: SMIRK's scale `s` is "crop-NDC units per FLAME unit", so one
      FLAME unit spans `s * HALF` crop pixels, i.e. `s * HALF * k` full-
      frame pixels. For a perspective camera at depth Z, one FLAME unit
      spans `f_px / Z` full-frame pixels. Equating:
          Z = f_px / (s * HALF * k)
            ~~ 2 * f_px / (s * bbox_size)

    Note this Z formula couples `s` (from SMIRK encoder) and `bbox_size`
    (from MediaPipe landmarks). Independent jitter in either amplifies
    multiplicatively into Z, and then into t_x, t_y (through Z * (u-cx)
    / f_px). Smoothing either one attacks the Z-jitter source directly.
    """
    CROP_SIZE = 224
    # SMIRK's weak-perspective renderer maps NDC=0 to crop pixel
    # CROP_SIZE/2 (= 112). The skimage similarity transform in `_crop_224`
    # maps bbox corners to dst pixels 0 and (CROP_SIZE-1). Keeping both
    # conventions — HALF_PROJ = 112 for SMIRK renderer, and divisor
    # (CROP_SIZE - 1) for the crop scale — preserves the exact numerical
    # output of the original `tform`-based implementation so the `.frame`
    # files don't silently change under this refactor.
    HALF_PROJ = CROP_SIZE / 2.0  # 112.0
    denom = max(CROP_SIZE - 1, 1)  # 223

    s, tx, ty = float(cam[0]), float(cam[1]), float(cam[2])
    bx, by = float(bbox_center[0]), float(bbox_center[1])

    # Crop-pixel-per-full-pixel ratio: (size - 1) / bbox_size. Its inverse
    # k turns crop pixels into full-frame pixels.
    k = float(bbox_size) / denom

    # Full-frame pixel of the projected FLAME origin (bbox position +
    # SMIRK-relative offset, plus a half-pixel bias inherited from the
    # 112 vs 111.5 convention mismatch above).
    u_full = bx + k * (HALF_PROJ + HALF_PROJ * s * tx) - bbox_size / 2.0
    v_full = by - k * (HALF_PROJ * s * ty - HALF_PROJ) - bbox_size / 2.0

    # Apparent face scale in full-frame pixels per FLAME unit, using
    # SMIRK's HALF_PROJ convention so `Z` matches the original formula.
    apparent_px_per_flame = s * HALF_PROJ * k
    Z = f_px / max(apparent_px_per_flame, 1e-6)

    cx = w / 2.0
    cy = h / 2.0
    return np.array([
        (u_full - cx) * Z / f_px,
        (v_full - cy) * Z / f_px,
        Z,
    ], dtype=np.float64)
