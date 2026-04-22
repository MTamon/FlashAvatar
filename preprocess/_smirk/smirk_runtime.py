"""Thin wrapper over SmirkEncoder that handles detection + crop + batching.

Kept separate from `preprocess/smirk_tracker.py` so the crop/tform math
is easy to unit-test without touching file I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    # MediaPipe landmarks in full-frame pixels (N, 2). Kept around so the
    # demo/verify tools can draw them without re-running MediaPipe. May be
    # the previous frame's landmarks when `detected` is False.
    landmarks: np.ndarray | None = None
    # MediaPipe Face Landmarker blendshape scores (dict[str, float]) used by
    # `preprocess.smirk_convert.eye_pose_6d_from_blendshapes` to synthesize
    # eye_pose. None when the `.task` file isn't available and we fell back
    # to legacy FaceMesh, or when no face was detected on this frame.
    blendshapes: dict = field(default_factory=dict)


class SmirkRunner:
    """Initializes SmirkEncoder + MediaPipe and runs per-batch inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._load_encoder()
        self._load_mediapipe()
        self._prev_landmarks: np.ndarray | None = None
        self._prev_blendshapes: dict = {}
        # Monotonically-increasing timestamp (ms) for MediaPipe VIDEO mode.
        # Tasks API's FaceLandmarker rejects non-increasing timestamps with
        # a RuntimeError, so we increment here rather than relying on a
        # caller-supplied frame index.
        self._mp_timestamp_ms: int = 0

    def _load_encoder(self) -> None:
        # SMIRK (MTamon/smirk@release/cuda128) ships as a proper `smirk`
        # package with `smirk/__init__.py` and `smirk/src/__init__.py`, so
        # we import via the dotted path. `_ensure_on_pythonpath` puts the
        # parent of the SMIRK clone on sys.path, which keeps us out of
        # FlashAvatar's own top-level `src/` namespace.
        from smirk.src.smirk_encoder import SmirkEncoder  # type: ignore
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
        """Build a MediaPipe detector that can surface ARKit blendshapes.

        Preferred path: instantiate `mediapipe.tasks.python.vision.FaceLandmarker`
        against the `.task` file that SMIRK's `quick_install.sh` downloads to
        `<smirk_root>/assets/face_landmarker.task`. This exposes
        `face_blendshapes`, which we feed into
        `eye_pose_6d_from_blendshapes` downstream.

        Fallback: the legacy `mp.solutions.face_mesh.FaceMesh` API. It still
        gives us landmarks for the crop but does NOT produce blendshapes — so
        `eye_mode blendshapes` silently degrades to identity eye pose for
        every frame. A one-line warning is emitted so the operator can fix
        the install (typically by re-running `bash external/smirk/quick_install.sh`).
        """
        self._landmarker = None
        self._mp_fm = None

        task_path = self._find_face_landmarker_task()
        if task_path is not None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision

                base = mp_python.BaseOptions(
                    model_asset_path=str(task_path),
                    delegate=mp_python.BaseOptions.Delegate.CPU,
                )
                # VIDEO mode enables MediaPipe's built-in tracking: after the
                # first successful detection, subsequent `detect_for_video`
                # calls seed the landmark regressor with the previous frame's
                # result, which reduces per-frame landmark jitter
                # substantially. Stateless IMAGE mode (the default) runs a
                # fresh detection on every frame and amplifies jitter that
                # then leaks into the SMIRK crop tform.
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=0.1,
                    min_face_presence_confidence=0.1,
                    min_tracking_confidence=0.1,
                )
                self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                self._mp_image_cls = mp.Image
                self._mp_image_format = mp.ImageFormat.SRGB
                return
            except Exception as e:
                print(
                    f"[smirk/runtime] MediaPipe FaceLandmarker init failed "
                    f"({e.__class__.__name__}: {e}); falling back to legacy "
                    f"FaceMesh — eye blendshapes will be unavailable.",
                )

        # Legacy fallback (no blendshapes).
        import mediapipe as mp
        self._mp_fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        )
        print(
            "[smirk/runtime] face_landmarker.task not found under "
            f"{self.cfg.smirk_root}/assets/ — using legacy FaceMesh. "
            "`--eye-mode blendshapes` will degrade to identity eye pose; "
            "run `bash external/smirk/quick_install.sh` to enable blendshapes.",
        )

    def _find_face_landmarker_task(self) -> Path | None:
        # SMIRK's quick_install.sh drops it at <smirk_root>/assets/face_landmarker.task.
        # An operator can also override with $MP_FACE_LANDMARKER_TASK.
        import os
        env = os.environ.get("MP_FACE_LANDMARKER_TASK")
        if env:
            p = Path(env)
            if p.is_file():
                return p
        cand = Path(self.cfg.smirk_root) / "assets" / "face_landmarker.task"
        return cand if cand.is_file() else None

    # ---- public -------------------------------------------------------

    def encode_batch(self, imgs_rgb: List[np.ndarray]) -> list[FrameResult]:
        crops = []
        tforms = []
        centers = []
        sizes = []
        detected_flags = []
        blendshape_dicts: list[dict] = []
        landmarks_list: list[np.ndarray] = []
        for img in imgs_rgb:
            lm, bs = self._detect(img)
            if lm is None:
                # Re-use last good landmarks so the crop stays roughly stable;
                # mark as not-detected so a caller can filter if desired.
                lm = self._prev_landmarks
                bs = self._prev_blendshapes
                detected = False
            else:
                detected = True
                self._prev_landmarks = lm
                self._prev_blendshapes = bs or {}
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
            blendshape_dicts.append(bs or {})
            landmarks_list.append(lm.astype(np.float32))

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
                landmarks=landmarks_list[i],
                blendshapes=blendshape_dicts[i],
            ))
        return results

    # ---- landmark detection + crop -----------------------------------

    def _detect(self, img_rgb: np.ndarray):
        """Return (landmarks (N,2) in full-frame px, blendshapes dict | None).

        Landmarks are what the SMIRK crop needs; blendshapes feed
        `eye_pose_6d_from_blendshapes`. Either can be None when the backend
        doesn't support it (legacy FaceMesh never surfaces blendshapes).
        """
        if self._landmarker is not None:
            mp_image = self._mp_image_cls(
                image_format=self._mp_image_format, data=img_rgb,
            )
            # VIDEO mode requires strictly-monotonic timestamps. The absolute
            # value doesn't matter (frames don't have to align to a real clock);
            # we just need each call > the previous. 33 ms/step corresponds to
            # ~30 fps but MediaPipe doesn't use the delta for anything other
            # than ordering, so we don't need to know the true FPS here.
            self._mp_timestamp_ms += 33
            try:
                res = self._landmarker.detect_for_video(
                    mp_image, self._mp_timestamp_ms,
                )
            except Exception:
                return None, None
            if not res.face_landmarks:
                return None, None
            h, w = img_rgb.shape[:2]
            lm = np.array(
                [[p.x * w, p.y * h] for p in res.face_landmarks[0]],
                dtype=np.float64,
            )
            bs: dict = {}
            if res.face_blendshapes:
                for cat in res.face_blendshapes[0]:
                    bs[cat.category_name] = float(cat.score)
            return lm, bs

        # legacy FaceMesh fallback
        res = self._mp_fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return None, None
        h, w = img_rgb.shape[:2]
        lm = np.array([[p.x * w, p.y * h]
                       for p in res.multi_face_landmarks[0].landmark])
        return lm, None

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
