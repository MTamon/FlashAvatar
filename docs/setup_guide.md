# FlashAvatar128 Setup Guide

Step-by-step checklist for setting up FlashAvatar on a fresh machine.
Follow each section in order.

---

## 1. System prerequisites

| Requirement | How to install |
|-------------|---------------|
| Ubuntu 22.04 | — |
| gcc-11 / g++-11 | `sudo apt install -y gcc-11 g++-11` |
| CUDA Toolkit 12.8 | [NVIDIA installer](https://developer.nvidia.com/cuda-12-8-0-download-archive) (system-wide, `nvcc` on PATH) |
| Python 3.11 | `sudo apt install python3.11 python3.11-venv` or pyenv |
| git | `sudo apt install git` |
| ffmpeg (optional) | `sudo apt install ffmpeg` (for frame extraction) |

Verify:

```bash
gcc-11 --version
nvcc --version          # should report 12.8
python3.11 --version
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

---

## 2. Clone and initialise

```bash
git clone https://github.com/MTamon/FlashAvatar128-.git
cd FlashAvatar128-
git checkout claude/update-pytorch-cuda-support-EgL6N

git submodule update --init --recursive
```

---

## 3. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## 4. FLAME model assets (manual download — licence required)

Register at **https://flame.is.tue.mpg.de/** and accept the licence.

### 4a. FLAME 2020

Download page → **FLAME 2020** → download the zip.

```bash
# Extract generic_model.pkl and place it:
cp /path/to/FLAME2020/generic_model.pkl flame/generic_model.pkl
```

### 4b. FLAME Vertex Masks

Same download page → **FLAME Vertex Masks** → download `FLAME_masks.zip`.

> **Warning**: the zip extracts its contents *flat* (no subdirectory
> is created).  Create the target directory first.

```bash
mkdir -p flame/FLAME_masks
cd flame/FLAME_masks
unzip /path/to/FLAME_masks.zip
cd ../..
```

### Verification

```bash
ls flame/generic_model.pkl         # must exist
ls flame/FLAME_masks/FLAME_masks.pkl  # must exist
```

The following files should already be in the repository:

```
flame/FlameMesh.obj
flame/landmark_embedding.npy
flame/blendshapes/l_eyelid.npy
flame/blendshapes/r_eyelid.npy
flame/mediapipe/mediapipe_landmark_embedding.npz
```

---

## 5. Install dependencies

```bash
# Set CUDA_HOME if not at the default path
export CUDA_HOME=/usr/local/cuda-12.8

# Optional: narrow the arch list to your GPU for faster builds
# Check your compute capability:
#   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Example for RTX 5090 (Blackwell, sm_120):
export TORCH_CUDA_ARCH_LIST="12.0"

bash install_128.sh
```

The script installs:
- pip packages pinned to DECA128-compatible versions
- chumpy from git (numpy 2.x compatible)
- pytorch3d v0.7.8 (source build, ~10 min)
- diff-gaussian-rasterization (source build)
- simple-knn (source build)

A sanity check runs at the end — all four imports must succeed:

```
torch            : 2.9.1+cu128
pytorch3d        : 0.7.8
diff_gaussian_rasterization : ok
simple_knn       : ok
```

---

## 6. Example data (Obama sequence)

Download the [example data](https://drive.google.com/file/d/1_WLvlmHD73jOAO178N7eX5UQqlrL2ghD/view?usp=drive_link)
and extract so that the following paths exist:

```
dataset/Obama/imgs/          # JPEG video frames (00001.jpg, 00002.jpg, ...)
dataset/Obama/parsing/       # segmentation masks (*_neckhead.png, *_mouth.png)
dataset/Obama/alpha/         # foreground masks (00001.jpg, 00002.jpg, ...)
metrical-tracker/output/Obama/checkpoint/  # .frame files (00000.frame, ...)
```

> **Note**: `--idname` must match a name that exists in **both**
> `dataset/` and `metrical-tracker/output/`.

---

## 7. Train (verification run)

```bash
python train.py --idname Obama --iterations 5000
```

- Progress images: `dataset/Obama/log/train/500.jpg`, `1000.jpg`, ...
- Checkpoint: `dataset/Obama/log/ckpt/chkpnt5000.pth`

Expected output:

```
step: 500, huber: 0.001xx
step: 1000, huber: 0.001xx
...
[ITER 5000] Saving Checkpoint
```

---

## 8. Test (generate visualisation video)

```bash
python test.py --idname Obama \
    --checkpoint dataset/Obama/log/ckpt/chkpnt5000.pth
```

Output: `dataset/Obama/log/test.avi` (GT left, rendered right, 25 FPS).

```bash
ffplay dataset/Obama/log/test.avi   # or vlc, mpv, etc.
```

---

## 9. Full-quality training

```bash
python train.py --idname Obama --iterations 150000
```

Runtime: ~5–15 min on an RTX 5090 / 3090 at 512×512.

---

## Quick reference: directory layout after setup

```
FlashAvatar128-/
├── flame/
│   ├── generic_model.pkl              ← downloaded (step 4a)
│   ├── FLAME_masks/FLAME_masks.pkl    ← downloaded (step 4b)
│   ├── FlameMesh.obj                  (in repo)
│   ├── landmark_embedding.npy         (in repo)
│   ├── blendshapes/{l,r}_eyelid.npy   (in repo)
│   └── mediapipe/mediapipe_landmark_embedding.npz  (in repo)
├── dataset/
│   └── Obama/                         ← downloaded (step 6)
│       ├── imgs/
│       ├── parsing/
│       ├── alpha/
│       └── log/                       (created by train/test)
├── metrical-tracker/
│   └── output/Obama/checkpoint/       ← downloaded (step 6)
├── submodules/
│   ├── diff-gaussian-rasterization/   (git submodule)
│   └── simple-knn/                    (git submodule)
├── install_128.sh
├── requirements_128.txt
├── train.py
├── test.py
└── docs/
    ├── setup_guide.md                 (this file)
    ├── conversion_rationale.md
    └── handover.md
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvcc not found` | CUDA_HOME not set or not on PATH | `export CUDA_HOME=/usr/local/cuda-12.8; export PATH=$CUDA_HOME/bin:$PATH` |
| `no kernel image is available` | GPU arch not in TORCH_CUDA_ARCH_LIST | Check `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and set `export TORCH_CUDA_ARCH_LIST="<your_cap>"`, then rebuild pytorch3d + submodules |
| pytorch3d tag not found | Tag case mismatch (V0.7.8 not v0.7.8) | Already fixed in install_128.sh |
| simple-knn SHA fetch fails | Shallow clone can't fetch bare SHA | Already fixed — install_128.sh fetches by branch name |
| `FLAME_masks.pkl` not found | Asset not downloaded | See step 4b |
| `generic_model.pkl` not found | Asset not downloaded | See step 4a |
| `metrical-tracker/output/<id>/checkpoint` not found | Wrong `--idname` or data not extracted | Ensure both `dataset/<id>/` and `metrical-tracker/output/<id>/checkpoint/` exist |
