"""Wire SMIRK's bbox stabilization helpers (release/cuda128) into FlashAvatar.

SMIRK's `utils/bbox_tracker.py` implements a stable-landmark subset + two
smoothing paths for the per-frame face bbox that drives the 224 crop:

* `extract_bbox_center_size(landmarks, use_stable_subset=True)` — center/size
  from eye corners / nose bridge / temples only, so mouth and blinks don't
  leak into bbox size.
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
import sys
from pathlib import Path
from types import ModuleType

_MODULE_NAME = "smirk_bbox_tracker"


def load_bbox_tracker(smirk_root: Path) -> ModuleType:
    """Load and return SMIRK's `utils/bbox_tracker.py` as a module.

    Raises a descriptive `FileNotFoundError` when the SMIRK checkout pre-dates
    the bbox-stabilization commit (MTamon/smirk@release/cuda128, PR #7). The
    error message includes the exact `git pull` incantation to update.
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
    return mod
