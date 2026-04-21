"""Thin wrapper over SmirkEncoder that handles detection + crop + batching.

Kept separate from `preprocess/smirk_tracker.py` so the crop/tform math
is easy to unit-test without touching file I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

# Imported lazily from the SMIRK repo root (added to sys.path by
# preprocess.smirk_tracker._ensure_on_pythonpath).


@dataclass
class FrameResult:
    # SMIRK encoder outputs, detached CPU tensors (B=1 removed):
    shape_params: np.ndarray      # (300,)
    expression_params: np.ndarray # (50,)
    pose_params: np.ndarray       # (3,) axis-angle global head rotation
    jaw_params: np.ndarray        # (3,) axis-angle
    eyelid_params: np.ndarray     # (2,)
    cam: np.ndarray               # (3,) weak-perspective [s, tx, ty]
    # Crop geometry (similarity tform mapping full-frame -> 224 crop):
    tform_matrix: np.ndarray      # (3, 3) full-frame px -> 224 crop px
    bbox_center: np.ndarray       # (2,) full-frame center of the crop bbox
    bbox_size: float              # full-frame side length of the crop bbox
    detected: bool = True         # False => we re-used the previous crop


class SmirkRunner:
    """Initializes SmirkEncoder + MediaPipe and runs per-batch inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._load_encoder()
        self._load_mediapipe()
        self._prev_landmarks: np.ndarray | None = None

    def _load_encoder(self) -> None:
        from src.smirk_encoder import SmirkEncoder  # type: ignore
        enc = SmirkEncoder().to(self.device).eval()
        ckpt = torch.load(str(self.cfg.checkpoint), map_location=self.device,
                          weights_only=False)
        # SMIRK checkpoints bundle {smirk_encoder, smirk_generator, ...};
        # strip the prefix.
        state = {}
        for k, v in ckpt.items():
            if k.startswith("smirk_encoder."):
                state[k[len("smirk_encoder."):]] = v
        enc.load_state_dict(state)
        self.encoder = enc

    def _load_mediapipe(self) -> None:
        # Try the SMIRK-bundled wrapper first (uses Face Landmarker .task);
        # fall back to the legacy FaceMesh API if the .task file is absent.
        try:
            from utils.mediapipe_utils import run_mediapipe  # type: ignore
            self._mediapipe_fn = run_mediapipe
            self._mediapipe_kind = "smirk"
        except Exception:
            import mediapipe as mp
            self._mp_fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            )
            self._mediapipe_fn = None
            self._mediapipe_kind = "legacy"

    # ---- public -------------------------------------------------------

    def encode_batch(self, imgs_rgb: List[np.ndarray]) -> list[FrameResult]:
        crops = []
        tforms = []
        centers = []
        sizes = []
        detected_flags = []
        for img in imgs_rgb:
            lm = self._detect(img)
            if lm is None:
                # Re-use last good landmarks so the crop stays roughly stable;
                # mark as not-detected so a caller can filter if desired.
                lm = self._prev_landmarks
                detected = False
            else:
                detected = True
                self._prev_landmarks = lm
            if lm is None:
                raise RuntimeError(
                    "No face detected in the very first frame. "
                    "SMIRK needs a valid MediaPipe detection at least once "
                    "to initialize the crop.")
            crop, tform, center, size = self._crop_224(img, lm)
            crops.append(crop)
            tforms.append(tform)
            centers.append(center)
            sizes.append(size)
            detected_flags.append(detected)

        batch = np.stack(crops, axis=0).astype(np.float32) / 255.0
        batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            out = self.encoder(batch_t)
        # SMIRK encoder outputs a dict with 'shape_params' etc; gather.
        out_np = {k: v.detach().cpu().numpy() for k, v in out.items()
                  if isinstance(v, torch.Tensor)}

        results: list[FrameResult] = []
        for i in range(len(imgs_rgb)):
            results.append(FrameResult(
                shape_params=out_np["shape_params"][i],
                expression_params=out_np["expression_params"][i],
                pose_params=out_np["pose_params"][i],
                jaw_params=out_np["jaw_params"][i],
                eyelid_params=out_np["eyelid_params"][i],
                cam=out_np["cam"][i],
                tform_matrix=tforms[i],
                bbox_center=centers[i],
                bbox_size=sizes[i],
                detected=detected_flags[i],
            ))
        return results

    # ---- landmark detection + crop -----------------------------------

    def _detect(self, img_rgb: np.ndarray) -> np.ndarray | None:
        if self._mediapipe_kind == "smirk" and self._mediapipe_fn is not None:
            try:
                lm = self._mediapipe_fn(img_rgb)
            except Exception:
                lm = None
            if lm is None:
                return None
            if isinstance(lm, tuple):
                # SMIRK helper returns (landmarks, extra) in some forks.
                lm = lm[0]
            lm = np.asarray(lm)
            if lm.ndim == 3:
                lm = lm[0]
            return lm[:, :2] if lm.shape[-1] >= 2 else None
        # legacy FaceMesh fallback
        res = self._mp_fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        h, w = img_rgb.shape[:2]
        lm = np.array([[p.x * w, p.y * h]
                       for p in res.multi_face_landmarks[0].landmark])
        return lm

    def _crop_224(self, img_rgb: np.ndarray, lm: np.ndarray):
        """Replicates SMIRK's demo `crop_face`. Returns (crop224, tform, c, s).

        `tform` is a 3x3 affine matrix mapping full-frame pixel -> crop pixel
        (homogeneous coords). Its inverse un-warps crop-space back to full.
        """
        from skimage.transform import estimate_transform, warp

        size = self.cfg.crop_size
        scale = self.cfg.crop_scale
        left, right = float(lm[:, 0].min()), float(lm[:, 0].max())
        top, bottom = float(lm[:, 1].min()), float(lm[:, 1].max())
        old_size = (right - left + bottom - top) / 2.0
        cx = (right + left) / 2.0
        cy = (bottom + top) / 2.0
        s = old_size * scale  # side length of the square crop, in full px
        src = np.array([
            [cx - s / 2.0, cy - s / 2.0],
            [cx - s / 2.0, cy + s / 2.0],
            [cx + s / 2.0, cy - s / 2.0],
        ], dtype=np.float64)
        dst = np.array([[0, 0], [0, size - 1], [size - 1, 0]], dtype=np.float64)
        tform = estimate_transform("similarity", src, dst)
        crop = warp(img_rgb, tform.inverse, output_shape=(size, size),
                    preserve_range=True, order=1).astype(np.uint8)
        return crop, tform.params.astype(np.float64), \
               np.array([cx, cy], dtype=np.float64), float(s)
