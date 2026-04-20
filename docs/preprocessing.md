# FlashAvatar Data Preparation

`preprocess/` implements the data preparation pipeline that turns a raw
monocular video into the four inputs `train.py` expects:

1. `dataset/<idname>/imgs/XXXXX.jpg`
2. `dataset/<idname>/parsing/XXXXX_{neckhead,mouth}.png`
3. `dataset/<idname>/alpha/XXXXX.jpg`
4. `metrical-tracker/output/<idname>/checkpoint/XXXXX.frame`

## Three-stage pipeline (one unified env)

All three stages run in the **same** FlashAvatar env (Python 3.11 /
PyTorch 2.9.1 / CUDA 12.8). We use the
[MTamon/metrical-tracker@cuda128](https://github.com/MTamon/metrical-tracker/tree/cuda128)
fork for the tracker; it shares FlashAvatar's pin set so no second env
is needed.

```
video.mp4
   │ ffmpeg
   ▼
raw/imgs/*.jpg  ──► BiSeNet  ──► raw/parsing/*_{neckhead,mouth}.png       │
                └► RVM       ──► raw/alpha/*.jpg                          │  1. prepare
                                                                          │
raw/imgs/*.jpg  ──► metrical-tracker ──► checkpoint_raw/*.frame           │  2. tracker
                                                                          │
   │ crop + resize to --size + K/img_size adjustment                      │  3. finalize
   ▼                                                                      │
dataset/<id>/{imgs,parsing,alpha}/    +    checkpoint/*.frame
```

**Coordinate-system contract**: stages 1 and 2 both run at the original
frame resolution. Stage 3 is the only place that interprets `--crop` /
`--no-crop`, so those upstream modules need no awareness of the flag.
Switching the crop flag never requires rerunning the tracker.

---

## 1. `preprocess.py prepare` — extract / parsing / matting

Runs inside the FlashAvatar env (same `.venv` / conda env used for
`train.py`).

```bash
python scripts/preprocess.py prepare \
    --idname myface \
    --video /path/to/my_video.mp4
```

Outputs:

```
dataset/myface/raw/imgs/00001.jpg ...   # native resolution
dataset/myface/raw/parsing/*.png        # native resolution
dataset/myface/raw/alpha/*.jpg          # native resolution
```

Stage skips: `--skip-extract`, `--skip-parsing`, `--skip-matting`.

| Stage | Dependency | Notes |
|---|---|---|
| extract | `ffmpeg` on PATH | — |
| parsing | BiSeNet (vendored under `preprocess/models/bisenet.py`, [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) MIT) | checkpoint `79999_iter.pth` auto-downloaded via `gdown` if installed; pass `--bisenet-weights PATH` to use a manual copy. |
| matting | [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | fetched via `torch.hub.load` on first run (needs internet). `--rvm-variant mobilenetv3\|resnet50`. |

---

## 2. metrical-tracker (+ MICA) — runs in the FlashAvatar env

Zielon/metrical-tracker upstream pins pytorch 1.12 / python 3.9 / CUDA
11.x, which does not support modern GPUs (e.g. RTX 5090 / Blackwell).
We use the [MTamon/metrical-tracker@cuda128](https://github.com/MTamon/metrical-tracker/tree/cuda128)
fork, which mirrors FlashAvatar's pin set (torch 2.9.1 / CUDA 12.8 /
numpy 2.2.6 / Python 3.11). The tracker needs a 300-dim FLAME shape
code (`identity.npy`) per actor, produced by
[MICA](https://github.com/Zielon/MICA); we use the pin-aligned
[MTamon/MICA@claude/cuda128-pytorch29-update-ZxnsN](https://github.com/MTamon/MICA/tree/claude/cuda128-pytorch29-update-ZxnsN)
fork. Both forks run in FlashAvatar's own env — no second env needed.

Set it up once (with FlashAvatar's env active):

```bash
source .venv/bin/activate       # or: conda activate <envname>
bash scripts/setup_metrical_tracker.sh
```

This:

1. Clones `MTamon/metrical-tracker@cuda128` into
   `external/metrical-tracker/` and pip-installs its extras (mediapipe,
   tensorboard, trimesh, matplotlib, ...) into the active env.
2. Downloads the FLAME 2020 / TextureSpace / FLAME_masks / head-template
   assets into `external/metrical-tracker/data/` (prompts for your
   https://flame.is.tue.mpg.de/ credentials; skip with
   `SKIP_ASSETS=1` or set `FLAME_USER=... FLAME_PASS=...`).
3. Clones `MTamon/MICA@claude/cuda128-pytorch29-update-ZxnsN` into
   `external/MICA/` and pip-installs MICA's extras (insightface 0.7.3,
   onnx, onnxruntime-gpu, gdown).
4. Symlinks `external/MICA/data/FLAME2020` → the tracker's FLAME2020
   (single source of truth; no second download).
5. Downloads MICA's `mica.tar` checkpoint and insightface's
   `antelopev2` / `buffalo_l` packs (to `~/.insightface/models/`) via
   `gdown`.

Overridable env vars: `TRACKER_REPO`, `TRACKER_BRANCH`, `TRACKER_DIR`,
`MICA_REPO`, `MICA_BRANCH`, `MICA_DIR`, `SKIP_ASSETS`, `FLAME_USER`,
`FLAME_PASS`.

Then run the tracker on the raw frames:

```bash
bash scripts/run_tracker.sh myface
```

This:

1. Stages `dataset/myface/raw/imgs/` as the actor's `source/` inside the
   tracker tree via symlink.
2. If `identity.npy` is missing for the actor, runs MICA on the first
   frame and writes `identity.npy` next to `source/`. (Skipped on
   re-runs, so you can hand-provide a better identity if MICA fails to
   detect a face.)
3. Auto-generates `configs/actors/<idname>.yml` and invokes
   `python tracker.py --cfg <yml>`.
4. Renames the resulting `checkpoint/` directory to `checkpoint_raw/`
   (the convention expected by the finalize step).

Manual equivalent:

```bash
source .venv/bin/activate       # or: conda activate <envname>

# (1) identity.npy: run MICA on the first frame.
cd external/MICA
python demo.py \
    -i /tmp/mica_in  -o /tmp/mica_out  -a /tmp/mica_arc \
    -m data/pretrained/mica.tar
cp /tmp/mica_out/<first-frame-stem>/identity.npy \
   ../metrical-tracker/input/myface/identity.npy

# (2) tracker: write configs/actors/myface.yml, then:
cd ../metrical-tracker
python tracker.py --cfg configs/actors/myface.yml
# then mv .../output/myface/checkpoint .../output/myface/checkpoint_raw
```

---

## 3. `preprocess.py finalize` — crop / resize / K adjustment

Back in the FlashAvatar env:

```bash
python scripts/preprocess.py finalize \
    --idname myface \
    --crop           # or --no-crop
    --size 512
```

Outputs:

```
dataset/myface/imgs/00001.jpg ...           # --size x --size
dataset/myface/parsing/*.png                # --size x --size
dataset/myface/alpha/*.jpg                  # --size x --size

metrical-tracker/output/myface/
├── checkpoint_raw/*.frame                  # tracker output (unchanged)
└── checkpoint/*.frame                      # rewritten K / img_size
```

Then:

```bash
python train.py --idname myface --iterations 5000
```

### Crop on / off

- `--crop` (default): derives a single square bbox that covers the union
  of `*_neckhead.png` masks (padded by `--crop-pad`, default 0.15) and
  uses it for every frame. Stable, non-jittery, face-centred.
- `--no-crop`: no face detection. The largest centred square is cropped
  from the source frame and resized to `--size`. Useful when the source
  video is already cropped or square.

Because parsing, matting and the tracker all run on the full frame,
switching `--crop` / `--no-crop` only requires rerunning `finalize`.

---

## Model checkpoints

| Model | Source | Expected location |
|---|---|---|
| BiSeNet `79999_iter.pth` | [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (Google Drive) | `preprocess_weights/79999_iter.pth` (auto if `gdown` installed; else download manually or pass `--bisenet-weights PATH`). |
| RVM | `torch.hub.load("PeterL1n/RobustVideoMatting", ...)` | torch hub cache. |
| metrical-tracker / FLAME | [MTamon/metrical-tracker@cuda128](https://github.com/MTamon/metrical-tracker/tree/cuda128) | handled by `scripts/setup_metrical_tracker.sh`; FLAME assets are gated at https://flame.is.tue.mpg.de/. |
| MICA (`mica.tar`) | [MTamon/MICA@claude/cuda128-pytorch29-update-ZxnsN](https://github.com/MTamon/MICA/tree/claude/cuda128-pytorch29-update-ZxnsN) | downloaded to `external/MICA/data/pretrained/mica.tar` via `gdown` (same script). |
| insightface `antelopev2`, `buffalo_l` | Google Drive (see MICA fork's `install.sh`) | downloaded to `~/.insightface/models/` via `gdown` (same script). |

## K adjustment details

Given a source frame of size `(W, H)` with intrinsics

```
    [fx  0  cx]
K = [ 0 fy  cy]
    [ 0  0   1]
```

and a square crop of side `b` at top-left `(x0, y0)` resized to `s x s`:

```
s_scale = s / b
fx' = fx * s_scale
fy' = fy * s_scale
cx' = (cx - x0) * s_scale
cy' = (cy - y0) * s_scale
img_size' = (s, s)
```

`preprocess.crop.adjusted_K` / `adjust_frame_files` apply this to every
`.frame` written to `checkpoint/`.
