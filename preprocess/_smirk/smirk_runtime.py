"""Thin wrapper over SmirkEncoder that handles detection + crop + batching.

Kept separate from `preprocess/smirk_tracker.py` so the crop/tform math
is easy to unit-test without touching file I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

# Imported lazily from the SMIRK repo root (added to sys.path by
# preprocess.smirk_tracker._ensure_on_pythonpath).


@dataclass
class _Prepared:
    """Per-frame encoder input assembled by the orchestrator.

    Populated either directly (legacy / online bbox) or from a precomputed
    smoothed tform list (offline bbox). Kept internal to the runtime layer.
    """
    crop: np.ndarray            # uint8 (224, 224, 3) RGB
    tform_matrix: np.ndarray    # (3, 3) full-frame px -> crop px
    bbox_center: np.ndarray     # (2,) full-frame center of the crop bbox
    # **Side length of the actual 224 crop region in full-frame pixels**,
    # i.e. legacy `_crop_224`'s `s = old_size * crop_scale`. NOT the bare
    # face extent — `_build_t` and `_draw_bbox` both depend on this unit.
    # SMIRK PR #8's `extract_bbox_center_size` returns just `face_extent
    # * size_calibration` (without `crop_scale`, because
    # `build_similarity_tform` multiplies by `scale` internally), so the
    # online / offline encode loops MUST multiply by `cfg.crop_scale`
    # before constructing this record.
    bbox_size: float
    landmarks: np.ndarray | None
    blendshapes: dict
    detected: bool


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
    # Same convention as `_Prepared.bbox_size`: full-frame side length of
    # the actual 224 crop region (= `face_extent * crop_scale`), NOT the
    # bare face extent. See `_Prepared` docstring for details.
    bbox_size: float
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
        """Legacy path: detect + legacy crop + encode for each image.

        Bbox is derived from the min/max of every MediaPipe landmark, which
        is what we shipped before bbox stabilization landed in SMIRK
        (release/cuda128). Kept as the default for backward-compatibility
        with existing `.frame` outputs. For stabilized bbox, callers should
        drive the runner via `detect_with_fallback` + `encode_prepared`
        (see `preprocess.smirk_tracker.run` for the orchestration).
        """
        prepared: list[_Prepared] = []
        for img in imgs_rgb:
            lm, bs, detected = self.detect_with_fallback(img)
            crop, tform, center, size = self._crop_224(img, lm)
            prepared.append(_Prepared(
                crop=crop, tform_matrix=tform,
                bbox_center=center, bbox_size=size,
                landmarks=lm, blendshapes=bs, detected=detected,
            ))
        return self.encode_prepared(prepared)

    def detect_with_fallback(
        self, img_rgb: np.ndarray,
    ) -> tuple[np.ndarray, dict, bool]:
        """Run MediaPipe on one frame; reuse the previous landmarks on miss.

        Returns ``(landmarks (N, 2), blendshapes, detected)``. Exists as a
        public method so the tracker orchestrator can detect landmarks in an
        offline pre-pass (before cropping) without reimplementing the
        first-frame / fallback semantics.

        Raises `RuntimeError` when the very first frame has no detection —
        we need at least one landmark set to seed the crop.
        """
        lm, bs = self._detect(img_rgb)
        if lm is None:
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
        return lm, bs or {}, detected

    def crop_224(self, img_rgb: np.ndarray, landmarks: np.ndarray):
        """Public wrapper for `_crop_224` (legacy all-landmarks bbox).

        Returns `(crop_uint8, tform_matrix_3x3, bbox_center_xy, bbox_size)`.
        """
        return self._crop_224(img_rgb, landmarks)

    def crop_224_with_tform(self, img_rgb: np.ndarray, tform) -> np.ndarray:
        """Warp `img_rgb` to a 224x224 crop given a precomputed skimage
        similarity transform.

        Online / offline bbox stabilization builds the similarity tform
        externally (via SMIRK's `build_similarity_tform`) and feeds the
        resulting crop to the encoder; this helper centralises the
        `skimage.transform.warp` invocation so the bilinear order and the
        `preserve_range` flag match what `_crop_224` uses in legacy mode.
        """
        from skimage.transform import warp

        size = self.cfg.crop_size
        crop = warp(
            img_rgb, tform.inverse,
            output_shape=(size, size), preserve_range=True, order=1,
        ).astype(np.uint8)
        return crop

    def encode_prepared(
        self, prepared: Sequence,
    ) -> list[FrameResult]:
        """Run the SMIRK encoder on a batch of pre-cropped inputs.

        Takes a sequence of record objects (either `_Prepared` from this
        module or any object with the same attributes — a `SimpleNamespace`
        works) and returns `FrameResult`s ready for the downstream `.frame`
        writer.

        The accepted duck-type is:

            record.crop          : np.uint8 (224, 224, 3) RGB
            record.tform_matrix  : np.ndarray (3, 3)
            record.bbox_center   : np.ndarray (2,)
            record.bbox_size     : float
            record.landmarks     : np.ndarray (N, 2) or None
            record.blendshapes   : dict
            record.detected      : bool

        Duck-typing matters here because this module is loaded twice at
        runtime (once as the top-level `smirk_runtime` module via
        `_ensure_on_pythonpath` — the name `SmirkRunner` sees — and once
        as `preprocess._smirk.smirk_runtime` via the package path).
        Importing `_Prepared` across those two paths would yield two
        distinct class objects and break `isinstance` checks; accepting
        any attribute-compatible record sidesteps the hazard.
        """
        if not prepared:
            return []
        batch_np = np.stack([p.crop for p in prepared], axis=0).astype(np.float32)
        batch_np /= 255.0
        batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            out = self.encoder(batch_t)
        out_np = {k: v.detach().cpu().numpy() for k, v in out.items()
                  if isinstance(v, torch.Tensor)}

        results: list[FrameResult] = []
        for i, p in enumerate(prepared):
            results.append(FrameResult(
                shape_params=out_np["shape_params"][i],
                expression_params=out_np["expression_params"][i],
                pose_params=out_np["pose_params"][i],
                jaw_params=out_np["jaw_params"][i],
                eyelid_params=out_np["eyelid_params"][i],
                cam=out_np["cam"][i],
                tform_matrix=np.asarray(p.tform_matrix, dtype=np.float64),
                bbox_center=np.asarray(p.bbox_center, dtype=np.float64),
                bbox_size=float(p.bbox_size),
                detected=bool(p.detected),
                landmarks=(p.landmarks.astype(np.float32)
                           if p.landmarks is not None else None),
                blendshapes=p.blendshapes or {},
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
