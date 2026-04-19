# FlashAvatar Data Preparation

`preprocess/` implements the data preparation pipeline that turns a raw
monocular video into the four inputs `train.py` expects:

1. `dataset/<idname>/imgs/XXXXX.jpg`
2. `dataset/<idname>/parsing/XXXXX_{neckhead,mouth}.png`
3. `dataset/<idname>/alpha/XXXXX.jpg`
4. `metrical-tracker/output/<idname>/checkpoint/XXXXX.frame`

## Overview

```
video.mp4
   │ ffmpeg
   ▼
raw/imgs/*.jpg  ──► BiSeNet  ──► raw/parsing/*_{neckhead,mouth}.png
                └► RVM       ──► raw/alpha/*.jpg
                └► tracker   ──► checkpoint_raw/*.frame
                                         │
                  ┌──────────────────────┘
                  ▼
              crop stage  (square bbox + resize to 512 + K adjustment)
                  │
                  ▼
    dataset/<id>/{imgs,parsing,alpha}/   +   checkpoint/*.frame
```

**Coordinate-system contract**: parsing, matting and the tracker all run at
the original frame resolution. The crop stage is the only place that
interprets `--crop` / `--no-crop`, so those upstream modules need no
awareness of the flag.

## Dependencies

| Stage | Dependency | Notes |
|---|---|---|
| extract | `ffmpeg` on PATH | — |
| parsing | BiSeNet (vendored under `preprocess/models/bisenet.py`) | checkpoint `79999_iter.pth` from [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch). Auto-downloaded via `gdown` if installed. |
| matting | [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | Fetched via `torch.hub.load` on first run (needs internet). |
| tracker | [metrical-tracker](https://github.com/Zielon/metrical-tracker) | External tool. Either run by hand or pass `--tracker-cmd` template. |
| crop | `Pillow`, `numpy`, `torch` | Already in `requirements_128.txt`. |

## Quick start

```bash
python scripts/preprocess.py \
    --idname myface \
    --video /path/to/my_video.mp4 \
    --crop \
    --size 512
```

Resulting tree:

```
dataset/myface/
├── raw/
│   ├── imgs/00001.jpg ...       # native resolution
│   ├── parsing/*.png            # native resolution
│   └── alpha/*.jpg              # native resolution
├── imgs/00001.jpg ...           # 512x512, face-centred crop
├── parsing/*.png                # 512x512, matching crop
└── alpha/*.jpg                  # 512x512, matching crop

metrical-tracker/output/myface/
├── checkpoint_raw/*.frame       # tracker output, native K
└── checkpoint/*.frame           # rewritten K / img_size = (512, 512)
```

Then:

```bash
python train.py --idname myface --iterations 5000
```

## Crop on / off

- `--crop` (default): derives a single square bbox that covers the union of
  `*_neckhead.png` masks (padded by `--crop-pad`, default 0.15) and uses it
  for every frame. This gives a stable, non-jittery crop centred on the
  face. Camera intrinsics `K` are rescaled accordingly.
- `--no-crop`: no face detection. The largest centred square is cropped
  from the source frame and resized to `--size`. Useful when the source
  video is already cropped or square.

Because parsing, matting and the tracker run before the crop stage and
always on the full frame, switching `--crop` / `--no-crop` does **not**
require rerunning them. Just rerun `preprocess` with `--skip-parsing
--skip-matting --skip-tracker` to re-apply a different crop.

## Skipping stages

Individual stages can be skipped when their outputs already exist:

```bash
# Only (re)run the crop stage, e.g. to change --size or toggle --crop.
python scripts/preprocess.py --idname myface \
    --skip-extract --skip-parsing --skip-matting --skip-tracker \
    --no-crop --size 512
```

## Metrical-tracker

The tracker is an external tool with its own environment. Two options:

1. **Run it by hand**, writing output to
   `metrical-tracker/output/<idname>/checkpoint_raw/` and then call
   `preprocess` with `--skip-tracker` so the crop stage picks it up.
2. **Delegate to the script** by providing a shell template via
   `--tracker-cmd`. Available placeholders: `{imgs_dir}`, `{ckpt_dir}`,
   `{idname}`. Example:

   ```bash
   python scripts/preprocess.py --idname myface --video my.mp4 \
       --tracker-cmd "python /opt/metrical-tracker/tracker.py \
                      --input_dir {imgs_dir} \
                      --output_dir metrical-tracker/output/{idname}"
   ```

The tracker output may go either to `checkpoint_raw/` or `checkpoint/`; the
script auto-detects and promotes in that order.

## Model checkpoints

| Model | Source | Expected location |
|---|---|---|
| BiSeNet `79999_iter.pth` | [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (Google Drive) | `preprocess_weights/79999_iter.pth` (auto if `gdown` installed; else download manually and put there, or pass `--bisenet-weights PATH`) |
| RVM | `torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")` | torch hub cache |

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

This is what `preprocess.crop.adjusted_K` and `adjust_frame_files` apply to
every `.frame` file.
