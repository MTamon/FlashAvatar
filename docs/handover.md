# FlashAvatar128 — Handover Memo

This document is a reference for future sessions (FLARE integration,
PyTorch 2.12 / CUDA 13.x migration, etc.).  It summarises what was
changed, why, and how the pieces connect.

---

## 1. What was done (branch: `claude/update-pytorch-cuda-support-EgL6N`)

### Commit history (oldest → newest)

| Commit | Summary |
|--------|---------|
| `60fcacf` | **Phase 1-5 umbrella commit** — all Python API fixes, `install_128.sh`, `requirements_128.txt`, `.gitignore`, `README.md`, simple-knn submodule bump |
| `9ad702f` | `install_128.sh`: add runtime simple-knn self-heal (detached checkout to known-good commit) |
| `d5fdda7` | `install_128.sh`: first attempt at pytorch3d tag checkout fix |
| `db72db3` | `install_128.sh`: switch to `git init` + `fetch --depth 1 origin tag` |
| `3928157` | `install_128.sh`: fix tag case — `V0.7.8` (uppercase) not `v0.7.8` |
| `e73bc12` | `install_128.sh`: fetch simple-knn by branch name (`origin/main`) instead of bare SHA |
| `58174e3` | Add Blackwell `sm_120` (RTX 5090) to TORCH_CUDA_ARCH_LIST; add FLAME asset + data docs to README |
| `126949c` | README: comprehensive custom data preparation pipeline |
| *(this session)* | `docs/`, `utils/flame_converter.py` — rationale doc, setup guide, converter, handover |

### Python-side API changes (Phase 1)

| File | Change | Reason |
|------|--------|--------|
| `flame/lbs.py:49` | `torch.autograd.Variable(torch.zeros(...).cuda())` → `torch.zeros(..., device=a.device)` | `Variable` removed in PyTorch 2.x |
| `utils/loss_utils.py:36` | Removed `Variable` wrapper on SSIM window | Same |
| `utils/general_utils.py:157` | `torch.cross(v1, v2)` → `torch.cross(v1, v2, dim=-1)` | `dim` required since PyTorch 2.1 |
| `train.py:71` | `torch.load(...)` → `torch.load(..., weights_only=False)` | Default changed to `True` in PyTorch 2.6 |
| `test.py:65` | Same | Same |
| `scene/__init__.py:39,62` | Same | Same |

### Install infrastructure (Phase 2)

- **`requirements_128.txt`**: pinned versions cloned from DECA128's
  `install_128.sh` + 3 FlashAvatar-only additions (lpips, plyfile,
  loguru).
- **`install_128.sh`**: pip-only installer that:
  - Forces `CC=gcc-11`, `CXX=g++-11`
  - Auto-detects `CUDA_HOME`
  - Sets `TORCH_CUDA_ARCH_LIST` (includes `12.0` for Blackwell)
  - Installs all deps with `--no-deps` to prevent pin drift
  - Builds chumpy from `mattloper/chumpy` git main
  - Builds pytorch3d `V0.7.8` from source (manual git clone to work
    around pip's filtered-clone tag bug)
  - Builds `diff-gaussian-rasterization` and `simple-knn` from
    submodules

### Submodule changes (Phase 3)

- **`submodules/simple-knn`**: pointer bumped to `60f461f` (latest
  `camenduru/simple-knn` main).  Key fixes: `.data` → `.data_ptr<float>()`
  and `#include <float.h>` for `FLT_MAX`.
- **`submodules/diff-gaussian-rasterization`**: left at INRIA main
  (`59f5f77`).  Builds successfully on torch 2.9.1 / CUDA 12.8 /
  sm_120 without patches.

### .gitignore

Changed `*txt` / `*sh` (overly broad) to `*.txt` / `*.sh` (extension-
specific).  Added explicit un-ignores for `!requirements_128.txt`,
`!install_128.sh`, `!README.md`.  Added `build/`, `*.egg-info/`, `*.so`.

---

## 2. Integration interfaces for FLARE

### Using FlashAvatar's deformation model from external code

```python
from src.deform_model import Deform_Model
from utils.flame_converter import FlameConverter

# One-time setup
deform_model = Deform_Model(device="cuda").to("cuda")
deform_model.load_state_dict(torch.load("checkpoint.pth", weights_only=False))
converter = FlameConverter(tracker="deca", device="cuda")

# Per-frame (real-time)
codedict = converter.convert_flame_params_only(deca_output)
codedict["shape"] = shape_param  # loaded once for the identity
verts, rot_delta, scale_coef = deform_model.decode(codedict)
```

### Key classes and entry points

| Component | File | Class / function | Input | Output |
|-----------|------|------------------|-------|--------|
| FLAME converter | `utils/flame_converter.py` | `FlameConverter.convert()` | tracker dict | `.frame` dict |
| FLAME converter (params only) | `utils/flame_converter.py` | `FlameConverter.convert_flame_params_only()` | tracker dict | codedict for `decode()` |
| Deformation model | `src/deform_model.py` | `Deform_Model.decode(codedict)` | codedict | verts, rot_delta, scale_coef |
| Gaussian renderer | `gaussian_renderer/__init__.py` | `render(viewpoint_cam, ...)` | Camera + gaussians | rendered image |
| Scene loader | `scene/__init__.py` | `Scene_mica(...)` | directory paths | Camera list + scene data |
| Training loop | `train.py` | `main` | CLI args | checkpoints + vis |
| Test / video gen | `test.py` | `main` | CLI args | `test.avi` |

### FlameConverter output format

`convert()` returns a dict matching the `.frame` spec:

```python
{
    "flame": {
        "shape":   Tensor(1, 300),  # float32
        "exp":     Tensor(1, 100),
        "jaw":     Tensor(1, 6),    # 6D rotation
        "eyes":    Tensor(1, 12),   # 6D × 2
        "eyelids": Tensor(1, 2),
    },
    "opencv": {
        "K": Tensor(1, 3, 3),
        "R": Tensor(1, 3, 3),
        "t": Tensor(1, 3),
    },
    "img_size": (width, height),
}
```

`convert_flame_params_only()` returns:

```python
{
    "shape":     Tensor(1, 300),
    "expr":      Tensor(1, 100),
    "jaw_pose":  Tensor(1, 6),
    "eyes_pose": Tensor(1, 12),
    "eyelids":   Tensor(1, 2),
}
```

This maps directly to the keys used in `train.py:100-103`.

---

## 3. Tracker support matrix

| Tracker | Expression | Jaw | Eyes | Eyelids | Camera |
|---------|-----------|-----|------|---------|--------|
| DECA | 50D → pad to 100D | axis-angle → rot6d | identity 12D | zeros 2D | weak → full |
| EMOCA | 50D → pad to 100D | axis-angle → rot6d | identity 12D | zeros 2D | weak → full |
| SMIRK | 50D → pad to 100D | axis-angle → rot6d | identity 12D | zeros 2D | weak → full |
| SPARK | 50D → pad to 100D | axis-angle → rot6d | if available | if available | weak → full |

To add a new tracker, add a `TrackerConfig` entry in
`TRACKER_CONFIGS` inside `utils/flame_converter.py`.

---

## 4. Notes for PyTorch 2.12 / CUDA 13.x migration (Phase 7)

### Expected timeline

PyTorch 2.12 is expected May 2026 with CUDA 13.0 / 13.2 support.

### What will likely need to change

1. **`requirements_128.txt`**: bump `torch`, `torchvision`, `triton`,
   and all `nvidia-*-cu12` packages to their cu13 equivalents
   (`nvidia-*-cu13`).
2. **`install_128.sh`**: update `CUDA_HOME` auto-detection to look for
   `/usr/local/cuda-13.0` or `/usr/local/cuda-13.2`.
3. **`TORCH_CUDA_ARCH_LIST`**: verify that sm_120 (Blackwell) is still
   the correct arch name under CUDA 13.x.  NVIDIA may introduce new
   arches (e.g. sm_130 for next-gen).
4. **pytorch3d**: check if v0.7.8 still builds against torch 2.12.
   If not, try v0.7.9 or the latest release.  The tag naming issue
   (uppercase `V`) may persist.
5. **`submodules/diff-gaussian-rasterization`**: likely fine, but
   verify that INRIA main doesn't use APIs removed in CUDA 13.x.
6. **`submodules/simple-knn`**: likely fine (simple CUDA code).
7. **Python API**: `torch.load(weights_only=False)` may eventually be
   removed.  Watch for FutureWarning in torch 2.12 release notes.

### Migration strategy

1. Create a new branch `claude/pytorch-2.12-cuda-13` from this branch.
2. Update pins in `requirements_128.txt`.
3. Run `install_128.sh` and fix build failures.
4. Run `train.py --idname Obama --iterations 5000` to verify.
5. Keep the diff minimal — same approach as this migration.

### What will NOT change

- `utils/flame_converter.py` — pure tensor math, no CUDA dependency.
- `docs/conversion_rationale.md` — mathematical facts, version-agnostic.
- The overall architecture (FLAME → MLP → Gaussians → render).
- The `.frame` file format.

---

## 5. Known limitations and future work

| Item | Status | Notes |
|------|--------|-------|
| Eye tracking | Not estimated by DECA/EMOCA/SMIRK | MediaPipe integration planned for FLARE Phase 2 |
| Eyelid tracking | Not estimated by DECA/EMOCA/SMIRK | Could use AU45 detection or SPARK |
| Camera from weak perspective | Heuristic conversion | FLARE should use actual camera calibration |
| FLAME texture (albedo) | Not used by FlashAvatar | FlashAvatar uses Gaussian splatting for appearance |
| Detail code (DECA) | Not used | FlashAvatar learns detail via per-vertex MLP |
| PyTorch 2.12 / CUDA 13.x | Pending release | See Section 4 |

---

## 6. File inventory (new files in this branch)

| File | Purpose |
|------|---------|
| `install_128.sh` | pip-only installer for Python 3.11 + torch 2.9.1 + CUDA 12.8 |
| `requirements_128.txt` | Pinned dependency set (DECA128-aligned) |
| `utils/flame_converter.py` | DECA/EMOCA/SMIRK/SPARK → FlashAvatar converter |
| `docs/conversion_rationale.md` | Mathematical justification for parameter conversions |
| `docs/setup_guide.md` | Step-by-step setup checklist |
| `docs/handover.md` | This file |

All git history is preserved on branch
`claude/update-pytorch-cuda-support-EgL6N`.  Use
`git log --stat 60fcacf..HEAD` to see the full diff set.
