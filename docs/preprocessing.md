# FlashAvatar Data Preparation

`preprocess/` implements the data preparation pipeline that turns a raw
monocular video into the four inputs `train.py` expects:

1. `dataset/<idname>/imgs/XXXXX.jpg`
2. `dataset/<idname>/parsing/XXXXX_{neckhead,mouth}.png`
3. `dataset/<idname>/alpha/XXXXX.jpg`
4. `metrical-tracker/output/<idname>/checkpoint/XXXXX.frame`

## Two-stage pipeline (split across the tracker boundary)

The heavy-weight FLAME tracker (metrical-tracker) runs in its own conda
env, so the pipeline is split into three pieces. The FlashAvatar env
drives steps 1 and 3; the tracker env drives step 2.

```
video.mp4
   │ ffmpeg
   ▼
raw/imgs/*.jpg  ──► BiSeNet  ──► raw/parsing/*_{neckhead,mouth}.png       │
                └► RVM       ──► raw/alpha/*.jpg                          │  1. prepare (FlashAvatar env)
                                                                          │
          ─────────────────── activate tracker env ────────────────────── ─
                                                                          │
raw/imgs/*.jpg  ──► metrical-tracker ──► checkpoint_raw/*.frame           │  2. tracker (tracker env)
                                                                          │
          ─────────────────── back to FlashAvatar env ─────────────────── ─
                                                                          │
   │ crop + resize to --size + K/img_size adjustment                      │  3. finalize (FlashAvatar env)
   ▼                                                                      │
dataset/<id>/{imgs,parsing,alpha}/    +    checkpoint/*.frame
```

**Coordinate-system contract**: stages 1 and 2 all run at the original
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

## 2. metrical-tracker — runs in its own env

metrical-tracker requires pytorch 1.12 / python 3.9 / CUDA 11.x, which
conflicts with FlashAvatar's environment. Set it up once with:

```bash
bash scripts/setup_metrical_tracker.sh
```

This clones upstream into `external/metrical-tracker/`, creates a conda
env (default name `tracker`), installs PyTorch 1.12 + CUDA 11.3 +
requirements, and runs the upstream `install.sh`. License-gated FLAME /
MICA assets may need to be placed manually — follow the upstream
instructions at https://github.com/Zielon/metrical-tracker if the
install.sh step prompts for them.

Overridable env vars: `TRACKER_DIR`, `ENV_NAME`, `PY_VERSION`, `CUDA`.

Then run the tracker on the raw frames:

```bash
bash scripts/run_tracker.sh myface
```

This activates the `tracker` env, runs `python tracker.py --input_dir
... --output_dir ...`, and renames the resulting `checkpoint/` directory
to `checkpoint_raw/` (the convention expected by the finalize step).

Manual equivalent:

```bash
conda activate tracker
cd external/metrical-tracker
python tracker.py \
    --input_dir /.../dataset/myface/raw/imgs \
    --output_dir /.../metrical-tracker/output/myface
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
| metrical-tracker / MICA / FLAME | see upstream | handled by `scripts/setup_metrical_tracker.sh` + license-gated manual steps. |

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
