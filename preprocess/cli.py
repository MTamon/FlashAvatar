"""End-to-end preprocessing orchestrator.

Layout produced under FlashAvatar's repo root:

    dataset/<idname>/
        raw/imgs/              # full-resolution frames (ffmpeg output)
        raw/parsing/           # full-resolution parsing masks
        raw/alpha/             # full-resolution alpha masks
        imgs/                  # --size x --size, post crop/no-crop
        parsing/               # --size x --size
        alpha/                 # --size x --size

    metrical-tracker/output/<idname>/
        checkpoint_raw/        # tracker output at native resolution
        checkpoint/            # adjusted .frame files for --size x --size

Upstream stages (parsing / matting / tracker) always work on `raw/`. The
crop stage is the only place that interprets `--crop` / `--no-crop` and
writes the final directory used by `train.py`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

from . import crop as crop_mod
from . import extract as extract_mod
from . import matting as matting_mod
from . import parsing as parsing_mod
from . import tracker as tracker_mod


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _infer_image_size(imgs_dir: Path) -> tuple[int, int]:
    frames = sorted(imgs_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames under {imgs_dir}")
    with Image.open(frames[0]) as im:
        return im.size  # (W, H)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess",
        description="FlashAvatar data preparation pipeline",
    )
    p.add_argument("--idname", required=True,
                   help="Identity name. Used as the dataset subdirectory.")
    p.add_argument("--video", type=Path, default=None,
                   help="Source video. Skipped if raw/imgs already populated.")
    p.add_argument("--repo-root", type=Path, default=_repo_root(),
                   help="FlashAvatar repository root (default: autodetected).")

    p.add_argument("--size", type=int, default=512,
                   help="Output image side length (default: 512).")
    crop_group = p.add_mutually_exclusive_group()
    crop_group.add_argument("--crop", dest="crop", action="store_true",
                            help="Tight face-centred square crop (default).")
    crop_group.add_argument("--no-crop", dest="crop", action="store_false",
                            help="Centre-square crop only, no face detection.")
    p.set_defaults(crop=True)
    p.add_argument("--crop-pad", type=float, default=0.15,
                   help="Fractional padding around the head bbox (crop=on).")

    p.add_argument("--device", default="cuda",
                   help="Torch device for parsing / matting (default: cuda).")

    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-parsing", action="store_true")
    p.add_argument("--skip-matting", action="store_true")
    p.add_argument("--skip-tracker", action="store_true",
                   help="Skip tracker step. Requires tracker output to "
                        "already exist, or the crop step will fail.")
    p.add_argument("--skip-crop", action="store_true",
                   help="Skip the final crop/resize step (outputs remain in raw/).")

    p.add_argument("--bisenet-weights", type=Path, default=None,
                   help="Path to the BiSeNet 79999_iter.pth checkpoint.")
    p.add_argument("--rvm-variant", default="mobilenetv3",
                   choices=["mobilenetv3", "resnet50"])
    p.add_argument("--tracker-cmd", default=None,
                   help="Shell template for invoking metrical-tracker. "
                        "See preprocess.tracker.run_tracker for placeholders.")

    p.add_argument("--overwrite", action="store_true",
                   help="Re-run stages even when their outputs already exist.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    repo = Path(args.repo_root).resolve()
    dataset_dir = repo / "dataset" / args.idname
    raw = dataset_dir / "raw"
    raw_imgs = raw / "imgs"
    raw_parsing = raw / "parsing"
    raw_alpha = raw / "alpha"
    final_imgs = dataset_dir / "imgs"
    final_parsing = dataset_dir / "parsing"
    final_alpha = dataset_dir / "alpha"

    ckpt_raw = repo / "metrical-tracker" / "output" / args.idname / "checkpoint_raw"
    ckpt_final = tracker_mod.checkpoint_dir(repo, args.idname)

    # 1. extract
    if not args.skip_extract:
        if args.video is None and not any(raw_imgs.glob("*.jpg")):
            print("[extract] --video not given and no frames under raw/imgs; "
                  "skipping extraction.")
        elif args.video is not None:
            print(f"[extract] {args.video} -> {raw_imgs}")
            n = extract_mod.extract_frames(
                args.video, raw_imgs, overwrite=args.overwrite,
            )
            print(f"[extract] {n} frames")

    n_frames = len(list(raw_imgs.glob("*.jpg")))
    if n_frames == 0:
        print(f"no frames under {raw_imgs}; aborting", file=sys.stderr)
        return 1

    # 2. parsing
    if not args.skip_parsing:
        print(f"[parsing] {raw_imgs} -> {raw_parsing}")
        parsing_mod.run_parsing(
            raw_imgs, raw_parsing,
            weights_path=args.bisenet_weights, device=args.device,
            overwrite=args.overwrite,
        )

    # 3. matting
    if not args.skip_matting:
        print(f"[matting] {raw_imgs} -> {raw_alpha}")
        matting_mod.run_matting(
            raw_imgs, raw_alpha,
            variant=args.rvm_variant, device=args.device,
            overwrite=args.overwrite,
        )

    # 4. tracker
    if not args.skip_tracker:
        if tracker_mod.count_frames(ckpt_raw) >= n_frames and not args.overwrite:
            print(f"[tracker] reusing {ckpt_raw}")
        elif args.tracker_cmd:
            print(f"[tracker] running: {args.tracker_cmd}")
            ckpt_raw.mkdir(parents=True, exist_ok=True)
            tracker_mod.run_tracker(
                args.tracker_cmd, raw_imgs, ckpt_raw, args.idname,
            )
        else:
            # The user may have run the tracker by hand straight into the final
            # dir. Accept that too.
            if tracker_mod.count_frames(ckpt_final) > 0 and \
                    tracker_mod.count_frames(ckpt_raw) == 0:
                print(f"[tracker] copying {ckpt_final} -> {ckpt_raw} "
                      f"(treating existing final output as raw).")
                ckpt_raw.mkdir(parents=True, exist_ok=True)
                for f in ckpt_final.glob("*.frame"):
                    shutil.copy2(f, ckpt_raw / f.name)
            else:
                tracker_mod.verify_or_hint(ckpt_raw, n_frames)

    # 5. crop / resize / K adjustment
    if args.skip_crop:
        print("[crop] skipped (raw outputs only)")
        return 0

    img_wh = _infer_image_size(raw_imgs)
    bbox = crop_mod.compute_bbox(
        raw_parsing if args.crop else None, img_wh, crop=args.crop,
    )
    print(f"[crop] bbox x0={bbox.x0} y0={bbox.y0} size={bbox.size} "
          f"-> {args.size}x{args.size} (crop={args.crop})")

    crop_mod.apply_to_images(raw_imgs, final_imgs, bbox, args.size,
                             overwrite=args.overwrite)
    crop_mod.apply_to_parsing(raw_parsing, final_parsing, bbox, args.size,
                              overwrite=args.overwrite)
    crop_mod.apply_to_alpha(raw_alpha, final_alpha, bbox, args.size,
                            overwrite=args.overwrite)
    crop_mod.adjust_frame_files(ckpt_raw, ckpt_final, bbox, args.size)

    print(f"[done] dataset/{args.idname}/ ready for train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
