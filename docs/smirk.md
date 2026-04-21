# SMIRK as an optional FLAME feature extractor

FlashAvatar's default tracker is [metrical-tracker](https://github.com/Zielon/metrical-tracker)
(via MTamon's `claude0420` fork), which performs an optimization-based
FLAME fit per frame. It is accurate on "studio-quality" sequences like
the Obama example but **fails on videos with large head rotations or
motion blur** — the per-frame optimization is sensitive to the initial
guess and accumulates errors when the head moves quickly.

[SMIRK](https://github.com/MTamon/smirk) is a feed-forward FLAME
encoder: it regresses FLAME parameters in a single network pass per
frame, making it robust to fast motion. The `release/cuda128` branch of
MTamon's fork is pin-aligned with FlashAvatar (Python 3.11 / PyTorch
2.9.1 / CUDA 12.8), so it runs in the same environment as FlashAvatar —
no extra env needed.

**SMIRK is opt-in.** It is not installed by `install_128.sh`. Install
it separately with `scripts/setup_smirk.sh` only when you need it.

## Quick start

```bash
source .venv/bin/activate
bash scripts/setup_smirk.sh                      # one-time

# Instead of:
#   bash scripts/run_tracker.sh <idname>
# run:
bash scripts/run_tracker.sh <idname> --smirk     # dispatcher
# or directly:
bash scripts/run_smirk_tracker.sh <idname>
# or via the CLI:
python scripts/preprocess.py smirk --idname <idname>

# Finalize is identical regardless of tracker:
python scripts/preprocess.py finalize --idname <idname>

# Then train as usual:
python train.py --idname <idname>
```

## Feature compatibility matrix

FlashAvatar's `.frame` file format (see
[preprocessing.md](preprocessing.md) for the full spec) expects the
following keys. This table shows how each is derived from SMIRK's
encoder output:

| `.frame` key        | Shape     | SMIRK source                | Conversion |
|---------------------|-----------|-----------------------------|-----------|
| `flame.shape`       | (1, 300)  | `shape_params` per frame    | **median** over the first `--shape-frames` detected frames, shared across every `.frame` file (matches metrical-tracker's "single identity" convention) |
| `flame.exp`         | (1, 100)  | `expression_params` (1, 50) | **zero-pad** the last 50 dims. FlashAvatar's FLAME basis is 100-dim but the extra 50 are simply never excited by SMIRK. |
| `flame.jaw`         | (1, 6)    | `jaw_params` axis-angle (3) | `matrix_to_rotation_6d(axis_angle_to_matrix(aa))` |
| `flame.eyes`        | (1, 12)   | **none** (SMIRK doesn't regress eyes) | identity 6D × 2. The avatar renders with static eyes; drive them externally if you need eye motion. |
| `flame.eyelids`     | (1, 2)    | `eyelid_params`             | clamp `[0, 1]` |
| `opencv.R`          | (1, 3, 3) | `pose_params` axis-angle    | `Rodrigues(aa)` then `diag(1,-1,-1) @ R` to convert OpenGL (y-up) → OpenCV (y-down, +z forward). |
| `opencv.t`          | (1, 3)    | `cam=[s, tx, ty]` + crop `tform` | see "Camera synthesis" below |
| `opencv.K`          | (1, 3, 3) | synthesized                 | `f_px = --focal-px` (default 5000), principal point = full-frame centre. |
| `img_size`          | (W, H)    | full raw frame resolution   | `preprocess finalize` later rewrites K and `img_size` to the final 512×512 head crop — same contract as metrical-tracker. |

### Camera synthesis (the subtle one)

SMIRK uses a **weak-perspective / orthographic** camera on its 224×224
internal crop. FlashAvatar uses a full **perspective** camera (OpenCV
`K/R/t`). The conversion has to satisfy two constraints:

1. The projected FLAME mesh must land at the correct pixel in the full
   raw frame (not SMIRK's 224 crop) — so the final 512 head crop that
   `preprocess finalize` will carve out has the right head position.
2. The projected mesh must have the correct **size** on screen.

We approximate orthographic with a perspective camera at large focal
length. The math lives in `preprocess/smirk_convert.py::_build_t`, but
the summary is:

- SMIRK's internal crop is a **similarity transform** `tform` from full
  frame pixel → 224 crop pixel. The transform is isotropic; its scale
  factor `s_ff = |tform[0,0]|` converts full-frame px to crop px.
- SMIRK's `cam = [s, tx, ty]` places the FLAME origin at crop pixel
  `(112 + 112*s*tx, 112 − 112*s*ty)` (y flipped inside SMIRK's
  renderer).
- Inverting `tform` gives the full-frame pixel of the FLAME origin.
- Matching the apparent size: 1 FLAME unit projects to `s * 112` crop
  pixels ⇒ `s * 112 / s_ff` full-frame pixels. For a perspective
  camera with focal `f_px` at depth `Z`, 1 unit projects to `f_px / Z`
  full-frame pixels. Equating, `Z = f_px * s_ff / (s * 112)`.
- Finally `t = [(u_full − cx) * Z / f_px, (v_full − cy) * Z / f_px, Z]`
  where `(cx, cy) = (W/2, H/2)`.

The `--focal-px` flag controls the perspective approximation: larger
`f_px` → farther depth → closer to orthographic. The default `5000` is
safe for head-only sequences in ~1k–4k resolution; bump to `10000` if
you see noticeable "near-camera" distortion on the edge of the frame.

### Why this decouples from FlashAvatar's 512-crop

`preprocess finalize --crop` computes an **entirely different** square
crop (the union of `*_neckhead.png` masks across the sequence, padded
by `--crop-pad`) from the full raw frame. It then rewrites `opencv.K`
and `img_size` in every `.frame` file to the final 512×512 view. We
write SMIRK's `.frame` files at **full raw resolution** precisely so
this downstream step works unchanged — the SMIRK internal 224 crop is
invisible to FlashAvatar.

## Caveats

1. **SMIRK does not regress eye-ball rotation.** We write an identity
   6D rotation for each eye; the trained avatar will render with static
   eyes. If you need eye motion, drive `flame.eyes` from an external
   signal (e.g. MediaPipe Face Landmarker blendshapes, or a separate
   eye-gaze regressor) as a post-pass after `preprocess smirk`.
2. **SMIRK's weak-perspective camera is an approximation.** The
   ortho→persp conversion produces negligible error when the head is
   small relative to the image and the focal is large, but can
   introduce a subtle scale bias vs metrical-tracker on short focal
   lens / very close-up videos. Use `--verify-dir` to dump overlays
   and inspect.
3. **Per-frame shape jitter.** SMIRK regresses 300-dim shape
   independently per frame; we collapse to the per-sequence median to
   match metrical-tracker. If your sequence has a disguise / expression
   change that meaningfully alters identity, increase `--shape-frames`
   or pre-filter the first frames.
4. **Expression dimension mismatch.** FlashAvatar was trained on the
   upstream tracker's 100-dim expressions. SMIRK only regresses the
   first 50 components; the remaining 50 are exactly zero. In practice
   the first 50 components dominate, but convergence may be slightly
   slower than with metrical-tracker output — consider training a few
   thousand more iterations.
5. **Requires a MediaPipe face detection in the very first frame.**
   SMIRK has no bbox detector; we detect with MediaPipe and propagate
   the last-good landmarks through frames where detection fails. If the
   first frame has no face, the runner errors out — trim the video.

## Running SMIRK's own demos (optional smoke test)

`scripts/setup_smirk.sh` clones SMIRK into `external/smirk/`. Once it
finishes, you can run SMIRK's upstream demos directly from that checkout
to confirm the install is healthy, independently of FlashAvatar's
pipeline. These demos render an overlay / video so you can eyeball the
tracking quality before committing to a full FlashAvatar run.

All demos run in the *same* active FlashAvatar venv — no separate env
switch needed (see "Environment compatibility" below).

```bash
source .venv/bin/activate
cd external/smirk

# (first time only) fetch a small bundle of sample videos for the demos
bash prepare_demos.sh

# Single image -> FLAME overlay + a rendered image
bash demos/run_demo.sh --input_path samples/test_image2.png --crop
#   writes external/smirk/output/ ... (see SMIRK's README for flags)

# Video -> overlay video
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop
#   writes external/smirk/output/dafoe/dafoe.mp4

# Video -> raw FLAME parameters (.pt dict, no rendering)
bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 --crop
#   writes external/smirk/output/dafoe/dafoe.pt
```

Common useful flags (forwarded by the `.sh` wrappers to the underlying
`demos/demo*.py`):

| Flag | Purpose |
|---|---|
| `--crop` | Use MediaPipe to auto-crop faces (match what FlashAvatar's SMIRK path does). Omit only if your inputs are already face-cropped. |
| `--mp_delegate {cpu,gpu}` | Where to run MediaPipe face detection. `gpu` is faster on CUDA boxes but needs the MediaPipe GPU delegate (default fallback is CPU). |
| `--with_eye_pose` | (save_flame only) Also derive eye rot6d + eyelids from MediaPipe blendshapes and pack them into the output .pt. Independent of FlashAvatar. |
| `--batch_size N` | Batch size for encoder inference. |
| `--benchmark` | Print per-stage timing. |

**Note**: these demos are SMIRK's own tooling; their output format
(`.pt` dicts with `shape/exp/pose/cam/...` per frame) is **different
from** FlashAvatar's `.frame` format. To feed SMIRK into FlashAvatar,
use `scripts/run_smirk_tracker.sh` (or `preprocess smirk`) — which
invokes SMIRK's encoder programmatically and performs the .frame
conversion described above. Running SMIRK's demos is strictly a sanity
check, not a replacement for the integration script.

Trouble-shooting:

- `ModuleNotFoundError: src.smirk_encoder` when running a demo → you
  left the `external/smirk/` directory before invoking; `cd` back.
- `FileNotFoundError: .../SMIRK_em1.pt` → `quick_install.sh` didn't
  complete; re-run `bash external/smirk/quick_install.sh`.
- Poor tracking on a specific face → try `--scale 1.6` in
  `demo_video.py` for a looser crop, or provide your own bbox.

See `external/smirk/README.md` for the full flag reference.

## Environment compatibility with FlashAvatar

The cuda128 branch of SMIRK is deliberately pin-aligned with
FlashAvatar's install_128.sh:

| | FlashAvatar `install_128.sh` | SMIRK `external/smirk/install_128.sh` |
|---|---|---|
| Python | 3.11 | 3.11 |
| CUDA   | 12.8 | 12.8 |
| PyTorch | 2.9.1 | 2.9.1 |
| numpy  | 2.2.6 | 2.2.6 |
| chumpy | `git+mattloper/chumpy@main` (numpy 2 compat) | same |

SMIRK-only extras that `setup_smirk.sh` pulls in:
`timm`, `albumentations`, `mediapipe`, `scikit-image`, and
`pytorch_lightning` (optional, inference code tolerates its absence).
None of these are on FlashAvatar's critical path; installing them does
not alter the torch / pytorch3d / diff_gaussian_rasterization /
simple_knn builds that `install_128.sh` produced.

The integration runs *both* SMIRK and FlashAvatar in a **single shared
venv** — there is no second environment to activate. The SMIRK code
itself is loaded via `sys.path.insert(0, external/smirk)` at import
time (see `preprocess/smirk_tracker._ensure_on_pythonpath`) so the
cloned repo doesn't need to be pip-installed as a package.

`setup_smirk.sh` runs a post-install sanity check that verifies the
key imports are still intact (`torch`, `pytorch3d`,
`diff_gaussian_rasterization`, `simple_knn`, and SMIRK's
`src.smirk_encoder`). If you ever see a failure there, something in
SMIRK's install chain downgraded a shared package — report the diff and
we'll fix the pin.

Known non-conflicts (paranoia list):

- `mediapipe` is installed by both metrical-tracker's
  `setup_metrical_tracker.sh` and SMIRK's installer. Both pin the same
  line (0.10.x); whichever runs last wins, but both work.
- `chumpy` is installed from GitHub main by both FlashAvatar and
  SMIRK. Re-running either is idempotent (pip detects the same
  commit).
- `pytorch_lightning` is only a SMIRK-side dep and does NOT pull a
  different torch; FlashAvatar never imports it.

## Verifying the output

Pass `--verify-dir PATH` to `preprocess smirk` (or via the shell
wrappers with `--verify-dir ...`) to dump:

- `verify/stats.csv` — per-frame detection flag, landmark reprojection
  error (px), crop bbox size.
- `verify/overlay_XXXXX.jpg` — a handful of sample frames with the
  FLAME mesh points projected via the synthesized `K/R/t`. Look for:
  - Projected points centred on the face (a gross offset means the R
    y-flip / t sign is wrong).
  - Projected mesh size matching the face size (wrong `Z` shows up as
    an over-/under-scaled mesh).

A median reprojection error below ~10 px on a 1080p input is "looks
correct"; above ~20 px triggers a warning and usually means the
ortho→persp depth estimate needs a larger `--focal-px`.

## Relation to the existing install / pipeline

- `install_128.sh` → unchanged. FlashAvatar's own env is not touched.
- `scripts/setup_metrical_tracker.sh` → unchanged, still the default.
- `scripts/setup_smirk.sh` → new, optional. Runs SMIRK's own
  `install_128.sh` + `quick_install.sh` (which share FlashAvatar's pin
  set) inside the active env.
- `scripts/run_tracker.sh` → now accepts `--smirk` and dispatches to
  `run_smirk_tracker.sh`. Default (no flag) is unchanged: metrical-tracker.
- `preprocess finalize` / training code → unchanged. Consumes the same
  `.frame` format.

The net external impact on an existing FlashAvatar tree that does NOT
opt into SMIRK is zero: no new deps, no behaviour changes.
