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
from types import SimpleNamespace
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
    # --- bbox stabilization (MTamon/smirk@release/cuda128, PR #7) ---
    # How to derive the per-frame 224 crop's bbox. Mirrors SMIRK's demo
    # `--bbox_mode` but carries one extra constraint: FlashAvatar's
    # preprocess pipeline is frame-indexed and file-driven, so:
    #   "legacy"  — original all-landmarks min/max bbox (DEFAULT for
    #               backward-compatibility with existing `.frame` files).
    #   "online"  — stable-landmark-subset bbox + per-frame One-Euro filter.
    #               Single-pass. Mirrors what a real-time / webcam pipeline
    #               would produce, so it's the right choice if the `.frame`
    #               consumer expects online-smoothed inputs.
    #   "offline" — stable-landmark-subset bbox + zero-phase FIR low-pass
    #               over the full size series in a pre-pass. Two-pass
    #               (detect all → filter → encode on smoothed crops).
    #               Best quality for teacher-data preparation.
    # Online and offline both require `bbox_fps` so the filter cutoff can
    # be normalised against Nyquist.
    bbox_mode: str = "legacy"
    bbox_all_landmarks: bool = False  # opt out of the stable subset
    bbox_fps: float | None = None     # required for online / offline
    # One-Euro parameters (only used in online mode).
    online_size_min_cutoff: float = 1.0
    online_size_beta: float = 0.02
    online_center_cutoff: float | None = None
    online_center_beta: float = 0.02
    # Offline FIR parameters (only used in offline mode).
    offline_size_cutoff: float = 2.5
    offline_size_taps: int = 61
    offline_center_cutoff: float | None = None
    # Stable-subset size calibration (SMIRK release/cuda128 PR #8). Without
    # this, the stable-landmark subset — which intentionally drops mouth /
    # jaw / eyebrows / forehead — produces a bbox whose vertical extent is
    # only ~30% of the full face, so `--bbox-mode online/offline` crops a
    # region ~60% the size of `legacy` and every projected FLAME vertex
    # appears shrunk in the overlay / rendered output. `None` (the default)
    # defers to SMIRK's own default constant `STABLE_LANDMARK_SIZE_CALIBRATION`
    # (currently 1.55), which matches the legacy crop extent empirically.
    # Pass `1.0` to disable the compensation for A/B diagnostics.
    size_calibration: float | None = None


def run(cfg: SmirkConfig, raw_imgs: Path, ckpt_out: Path,
        verify_dir: Path | None = None,
        demo_path: Path | None = None,
        demo_fps: float = 25.0,
        demo_lock_bbox: bool = False,
        demo_smooth_bbox: int = 0,
        demo_lbs_pose: bool = True,
        demo_ext_pose: bool = False,
        demo_vertex_stride: int = 8,
        demo_vertex_radius: int = 1,
        demo_vertex_radius_rel: float | None = None,
        lpf_cfg=None) -> int:
    """Drive the whole tracker: detect+crop+encode+convert+write.

    Returns the number of `.frame` files written.

    `lpf_cfg` is an optional `preprocess.smirk_lpf.LpfConfig`. When
    supplied, a zero-phase FIR LPF is applied across the frame sequence
    to the configured FLAME channels before `.frame` files are written
    AND before the demo / verify overlays are rendered, so the smoothed
    values feed into every downstream consumer in one place.
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

    # Pass 1: encode every frame, collect per-frame payloads. The bbox-mode
    # branches below all populate `payloads` / `img_size` with identical
    # semantics so downstream (LPF, canonicalize_shape, .frame writer,
    # verify, demo) are mode-agnostic.
    from preprocess.smirk_convert import (
        FramePayload, canonicalize_shape, to_flashavatar_frame,
    )
    payloads: list[FramePayload] = []
    img_size: tuple[int, int] | None = None

    bbox_mode = getattr(cfg, "bbox_mode", "legacy")
    if bbox_mode == "legacy":
        img_size, payloads = _run_legacy(cfg, runner, frames)
    elif bbox_mode == "online":
        img_size, payloads = _run_online(cfg, runner, frames)
    elif bbox_mode == "offline":
        img_size, payloads = _run_offline(cfg, runner, frames)
    else:
        raise ValueError(
            f"unknown bbox_mode {bbox_mode!r}; "
            f"expected one of: legacy, online, offline")

    # Pass 1.5 (optional): temporal LPF on selected FLAME channels.
    # Runs after encode so we have the full sequence in memory, but
    # before shape canonicalization (shape is identity-constant and isn't
    # in the LPF channel set; order doesn't affect it) and before the
    # .frame writer so the smoothed values land in both the on-disk
    # .frame files and the demo overlays.
    if lpf_cfg is not None:
        from preprocess.smirk_lpf import smooth_payloads
        smooth_payloads(payloads, lpf_cfg)

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

    if demo_path is not None:
        from preprocess.smirk_demo import dump_demo
        dump_demo(payloads, shape, img_size, cfg, demo_path, fps=demo_fps,
                  lock_bbox=demo_lock_bbox, smooth_bbox=demo_smooth_bbox,
                  use_lbs_pose=demo_lbs_pose,
                  use_ext_pose=demo_ext_pose,
                  mesh_stride=demo_vertex_stride,
                  vertex_radius=demo_vertex_radius,
                  vertex_radius_rel=demo_vertex_radius_rel)

    return n_written


def _ensure_on_pythonpath(smirk_root: Path) -> None:
    """Add `external/` AND this package's `preprocess/_smirk/` to sys.path.

    SMIRK ships as a proper `smirk` package (MTamon/smirk@release/cuda128
    has `external/smirk/__init__.py` and `external/smirk/src/__init__.py`),
    so we put the *parent* of the SMIRK clone on sys.path and import it
    as `smirk.src.smirk_encoder`. Going via `smirk.src.*` sidesteps the
    FlashAvatar top-level `src/` package (which would otherwise shadow a
    bare `src.X` import). The runtime / convert / verify helpers live
    under `preprocess/_smirk/` so users don't have to pip-install
    FlashAvatar to run the tracker.
    """
    smirk_root = Path(smirk_root).resolve()
    if not (smirk_root / "src" / "smirk_encoder.py").is_file():
        raise FileNotFoundError(
            f"SMIRK not found at {smirk_root} (missing src/smirk_encoder.py).\n"
            f"Run scripts/setup_smirk.sh to clone + install it."
        )
    if not (smirk_root / "__init__.py").is_file() or \
       not (smirk_root / "src" / "__init__.py").is_file():
        raise RuntimeError(
            f"SMIRK checkout at {smirk_root} is missing __init__.py — you're "
            f"on an old SMIRK revision. Update with:\n"
            f"    (cd {smirk_root} && git pull --ff-only origin release/cuda128)"
        )
    parent = str(smirk_root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    # Also expose our private runtime module.
    here = Path(__file__).resolve().parent / "_smirk"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


# --- Per-bbox-mode encode loops ------------------------------------------
#
# Each `_run_*` returns `(img_size_wh, payloads)`. They all:
#   * open every frame at disk-read order (sorted glob), in batches of
#     `cfg.batch_size`,
#   * advance MediaPipe VIDEO tracking in strict monotonic frame order
#     (critical — Tasks API rejects non-increasing timestamps),
#   * produce payloads indexed by position in the sorted list (not by
#     filename number), which the existing writer and downstream tools
#     already assume.


def _run_legacy(cfg: SmirkConfig, runner, frames: list[Path]
                ) -> tuple[tuple[int, int], list]:
    """All-landmarks min/max bbox, no temporal smoothing (pre-PR#7 behaviour)."""
    from preprocess.smirk_convert import FramePayload

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
    assert img_size is not None  # frames is non-empty by caller's invariant
    return img_size, payloads


def _run_online(cfg: SmirkConfig, runner, frames: list[Path]
                ) -> tuple[tuple[int, int], list]:
    """Per-frame One-Euro bbox — same order / semantics as a real-time loop.

    The filter is stateful and MUST see frames in monotonic order, so we
    slice into batches only for encoder throughput; the detection + filter
    step still walks frame-by-frame within each batch.
    """
    from preprocess._smirk.bbox_stabilizer import load_bbox_tracker
    from preprocess.smirk_convert import FramePayload

    if cfg.bbox_fps is None:
        raise ValueError(
            "bbox_mode='online' requires bbox_fps (source video fps). "
            "The One-Euro cutoff is specified in Hz and normalised against "
            "sample rate — no fps, no filter.")

    bbox_mod = load_bbox_tracker(cfg.smirk_root)
    tracker = bbox_mod.OnlineBBoxTracker(
        fps=float(cfg.bbox_fps),
        image_size=cfg.crop_size,
        scale=cfg.crop_scale,
        use_stable_subset=not cfg.bbox_all_landmarks,
        size_calibration=cfg.size_calibration,
        size_min_cutoff=cfg.online_size_min_cutoff,
        size_beta=cfg.online_size_beta,
        center_min_cutoff=cfg.online_center_cutoff,
        center_beta=cfg.online_center_beta,
    )

    payloads: list[FramePayload] = []
    img_size: tuple[int, int] | None = None
    for i in tqdm(range(0, len(frames), cfg.batch_size),
                  desc="smirk/encode (online)"):
        batch = frames[i:i + cfg.batch_size]
        imgs = [np.array(Image.open(p).convert("RGB")) for p in batch]
        if img_size is None:
            h, w = imgs[0].shape[:2]
            img_size = (w, h)
        # `SmirkRunner` loads as a top-level module (see _ensure_on_pythonpath)
        # so importing its `_Prepared` dataclass by either path would create
        # two distinct class identities. Duck-type instead — `encode_prepared`
        # only does attribute access on these records, not isinstance checks.
        prepared: list = []
        for img in imgs:
            lm, bs, detected = runner.detect_with_fallback(img)
            tform, center, size = tracker.update(lm)
            crop = runner.crop_224_with_tform(img, tform)
            # `bbox_size` semantics: legacy `_crop_224` stores `old_size *
            # crop_scale` (= the actual crop region's full-frame side
            # length, padding included), and `_build_t` / `_draw_bbox`
            # downstream are written against that convention. SMIRK's
            # `extract_bbox_center_size` returns the bare face extent
            # (calibration applied, but `crop_scale` NOT applied — see
            # `build_similarity_tform`, which multiplies internally). We
            # restore the legacy convention here so the same `bbox_size`
            # semantics flow through `_build_t` regardless of bbox_mode.
            # Without this multiplication, online/offline `bbox_size` is
            # ~71% of legacy and the back-projected FLAME mesh appears
            # at ~71% scale (`Z = f_px / (s * HALF_PROJ * (bbox_size /
            # 223))` => Z grows by 1/0.71, mesh shrinks by 0.71).
            bbox_size_full = float(size) * float(cfg.crop_scale)
            prepared.append(SimpleNamespace(
                crop=crop,
                tform_matrix=tform.params.astype(np.float64),
                bbox_center=np.asarray(center, dtype=np.float64),
                bbox_size=bbox_size_full,
                landmarks=lm, blendshapes=bs, detected=detected,
            ))
        results = runner.encode_prepared(prepared)
        for j, r in enumerate(results):
            payloads.append(FramePayload(
                idx=i + j, src_path=batch[j], result=r,
            ))
    assert img_size is not None
    return img_size, payloads


def _run_offline(cfg: SmirkConfig, runner, frames: list[Path]
                 ) -> tuple[tuple[int, int], list]:
    """Two-pass: detect + filter the size/center series, then encode.

    Pass 1 walks every frame to collect landmarks and raw (center, size) via
    SMIRK's stable-landmark subset. Pass 2 re-reads each frame (cheap — the
    OS page cache is warm after pass 1), applies the smoothed tform, and
    runs the encoder. We do NOT re-detect in pass 2: MediaPipe's VIDEO
    tracking state would advance twice and the two pass-1 / pass-2 landmark
    streams would diverge; we cache the pass-1 result and thread it through.
    """
    from preprocess._smirk.bbox_stabilizer import load_bbox_tracker
    from preprocess.smirk_convert import FramePayload

    if cfg.bbox_fps is None:
        raise ValueError(
            "bbox_mode='offline' requires bbox_fps (source video fps). "
            "The FIR cutoff is specified in Hz and normalised against "
            "sample rate — no fps, no filter.")

    bbox_mod = load_bbox_tracker(cfg.smirk_root)
    use_subset = not cfg.bbox_all_landmarks

    # Pass 1: detect landmarks + collect raw bbox series.
    lm_cache: list[np.ndarray] = []
    bs_cache: list[dict] = []
    detected_cache: list[bool] = []
    raw_centers_list: list[np.ndarray] = []
    raw_sizes_list: list[float] = []
    img_size: tuple[int, int] | None = None
    for p in tqdm(frames, desc="smirk/bbox pre-pass"):
        img = np.array(Image.open(p).convert("RGB"))
        if img_size is None:
            h, w = img.shape[:2]
            img_size = (w, h)
        lm, bs, detected = runner.detect_with_fallback(img)
        # `size_calibration=None` defers to SMIRK's default 1.55 when the
        # stable subset is in use (PR #8), which rescales the subset-derived
        # size to cover the same face extent as the legacy all-landmarks
        # bbox. Skipping this would shrink the crop (and therefore every
        # downstream FLAME vertex projection) by ~40%.
        c, s = bbox_mod.extract_bbox_center_size(
            lm, use_stable_subset=use_subset,
            size_calibration=cfg.size_calibration,
        )
        lm_cache.append(lm)
        bs_cache.append(bs)
        detected_cache.append(detected)
        raw_centers_list.append(np.asarray(c, dtype=np.float64))
        raw_sizes_list.append(float(s))

    raw_centers = np.stack(raw_centers_list, axis=0)
    raw_sizes = np.asarray(raw_sizes_list, dtype=np.float64)

    # Filter the size series. Centers are left raw unless the operator
    # explicitly asks for them to be smoothed (same default as SMIRK: the
    # head should still track fast translations faithfully).
    smooth_sizes = bbox_mod.fir_lowpass_offline(
        raw_sizes, fps=float(cfg.bbox_fps),
        cutoff_hz=cfg.offline_size_cutoff, taps=cfg.offline_size_taps,
    )
    smooth_centers = raw_centers.copy()
    if cfg.offline_center_cutoff is not None and cfg.offline_center_cutoff > 0:
        smooth_centers[:, 0] = bbox_mod.fir_lowpass_offline(
            raw_centers[:, 0], fps=float(cfg.bbox_fps),
            cutoff_hz=cfg.offline_center_cutoff, taps=cfg.offline_size_taps,
        )
        smooth_centers[:, 1] = bbox_mod.fir_lowpass_offline(
            raw_centers[:, 1], fps=float(cfg.bbox_fps),
            cutoff_hz=cfg.offline_center_cutoff, taps=cfg.offline_size_taps,
        )

    precomputed_tforms = [
        bbox_mod.build_similarity_tform(
            smooth_centers[i], float(smooth_sizes[i]),
            scale=cfg.crop_scale, image_size=cfg.crop_size,
        )
        for i in range(len(frames))
    ]

    # Pass 2: encode with the smoothed crops.
    payloads: list[FramePayload] = []
    for i in tqdm(range(0, len(frames), cfg.batch_size),
                  desc="smirk/encode (offline)"):
        batch = frames[i:i + cfg.batch_size]
        imgs = [np.array(Image.open(p).convert("RGB")) for p in batch]
        prepared: list = []
        for j, img in enumerate(imgs):
            idx = i + j
            tform = precomputed_tforms[idx]
            crop = runner.crop_224_with_tform(img, tform)
            # See comment in `_run_online`: legacy `bbox_size` is the
            # actual crop side length (`old_size * crop_scale`), whereas
            # SMIRK's `extract_bbox_center_size` returns just the face
            # extent. Multiply here so `_build_t` / `_draw_bbox` see a
            # consistent unit across all bbox modes.
            bbox_size_full = float(smooth_sizes[idx]) * float(cfg.crop_scale)
            prepared.append(SimpleNamespace(
                crop=crop,
                tform_matrix=tform.params.astype(np.float64),
                bbox_center=smooth_centers[idx],
                bbox_size=bbox_size_full,
                landmarks=lm_cache[idx],
                blendshapes=bs_cache[idx],
                detected=detected_cache[idx],
            ))
        results = runner.encode_prepared(prepared)
        for j, r in enumerate(results):
            payloads.append(FramePayload(
                idx=i + j, src_path=batch[j], result=r,
            ))
    assert img_size is not None
    return img_size, payloads


# --- Path helpers used by the CLI wrappers --------------------------------

def checkpoint_raw_dir(repo_root: Path, idname: str) -> Path:
    return (Path(repo_root) / "metrical-tracker" / "output" / idname
            / "checkpoint_raw")


def raw_imgs_dir(repo_root: Path, idname: str) -> Path:
    return Path(repo_root) / "dataset" / idname / "raw" / "imgs"
