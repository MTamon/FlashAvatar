"""Wire SMIRK's bbox stabilization helpers (release/cuda128) into FlashAvatar.

SMIRK's `utils/bbox_tracker.py` implements a stable-landmark subset + two
smoothing paths for the per-frame face bbox that drives the 224 crop:

* `extract_bbox_center_size(landmarks, use_stable_subset=True,
  size_calibration=None)` — center/size from eye corners / nose bridge /
  temples only, so mouth and blinks don't leak into bbox size. The
  `size_calibration` scalar (PR #8, default `STABLE_LANDMARK_SIZE_CALIBRATION`
  = 1.55) rescales the subset-derived size so the final crop matches the
  extent of the legacy all-landmarks crop — without it the 224 crop covers
  only ~60% of the face, which makes the FLAME mesh and every projected
  vertex appear shrunk in the overlay.
* `OnlineBBoxTracker` — per-frame One-Euro filter wrapper. O(1) state,
  safe inside a real-time / webcam loop.
* `fir_lowpass_offline` — zero-phase symmetric FIR over a 1-D size series.
  Requires the whole sequence in memory; used for offline preprocessing.
* `build_similarity_tform` — rebuilds the skimage similarity transform from
  a (center, size) pair so callers can drop in precomputed smoothed series.

SMIRK ships `utils/` without `__init__.py`, so a plain `from smirk.utils...`
import would fail and also collide with FlashAvatar's own top-level
`utils/` package. We load the file by path via importlib; the resulting
module is cached on a custom name (`smirk_bbox_tracker`) that has no other
consumer.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

_MODULE_NAME = "smirk_bbox_tracker"


def load_bbox_tracker(smirk_root: Path) -> ModuleType:
    """Load and return SMIRK's `utils/bbox_tracker.py` as a module.

    Raises a descriptive `FileNotFoundError` when the SMIRK checkout pre-dates
    the bbox-stabilization commit (MTamon/smirk@release/cuda128, PR #7).
    Raises `RuntimeError` when the checkout is post-PR#7 but pre-PR#8 (i.e.
    missing the `size_calibration` kwarg that keeps the stable-subset crop
    extent matched to the legacy crop). Both error messages include the
    exact `git pull` incantation to update.
    """
    smirk_root = Path(smirk_root).resolve()
    bbox_path = smirk_root / "utils" / "bbox_tracker.py"
    if not bbox_path.is_file():
        raise FileNotFoundError(
            f"SMIRK bbox_tracker.py not found at {bbox_path}. This feature "
            f"ships in MTamon/smirk@release/cuda128 (PR #7); update the "
            f"checkout with:\n"
            f"    (cd {smirk_root} && git pull --ff-only origin release/cuda128)"
        )

    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None and getattr(cached, "__file__", None) == str(bbox_path):
        return cached

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(bbox_path))
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"failed to build import spec for {bbox_path}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so any internal self-imports resolve.
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)

    # Post-PR#8 the stable-subset bbox is rescaled by `size_calibration` so
    # its crop extent matches the legacy all-landmarks crop. Without this
    # kwarg, `--bbox-mode online/offline` silently produces a crop ~60% the
    # size of legacy, which propagates through `_build_t` as a shrunk FLAME
    # mesh in both the `.frame` output and the demo overlay. Fail loud on
    # older checkouts rather than silently render a broken sequence.
    if not supports_size_calibration(mod):
        raise RuntimeError(
            f"SMIRK checkout at {smirk_root} predates PR #8 (stable-subset "
            f"size calibration). Its `extract_bbox_center_size` lacks the "
            f"`size_calibration` kwarg, so --bbox-mode online/offline will "
            f"produce a ~60%-sized crop and the reprojected FLAME mesh "
            f"appears shrunk. Update with:\n"
            f"    (cd {smirk_root} && git pull --ff-only origin release/cuda128)"
        )

    return mod


def supports_size_calibration(bbox_mod: ModuleType) -> bool:
    """Return True iff SMIRK's bbox_tracker exposes PR #8's size-calibration.

    We check both `extract_bbox_center_size` and `OnlineBBoxTracker.__init__`
    because PR #8 added the kwarg to both. The two must move together so a
    partial / hand-patched checkout is detected.
    """
    fn = getattr(bbox_mod, "extract_bbox_center_size", None)
    tracker_cls = getattr(bbox_mod, "OnlineBBoxTracker", None)
    if fn is None or tracker_cls is None:
        return False
    try:
        fn_params = inspect.signature(fn).parameters
        tracker_params = inspect.signature(tracker_cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return "size_calibration" in fn_params and "size_calibration" in tracker_params
