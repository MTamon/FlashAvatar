"""Two-stage preprocessing orchestrator.

The pipeline is split along the metrical-tracker boundary so that the
heavy-weight tracker can run in its own conda environment:

  1. `prepare`  — extract + parsing + matting.
                  Runs inside the FlashAvatar python environment.
                  Writes dataset/<id>/raw/{imgs,parsing,alpha}/.

  2. (external) — activate the tracker env and run metrical-tracker on
                  dataset/<id>/raw/imgs/, placing its output under
                  metrical-tracker/output/<id>/checkpoint_raw/.
                  See scripts/setup_metrical_tracker.sh for env setup
                  and scripts/run_tracker.sh for a convenience wrapper.

  3. `finalize` — crop/resize + K/img_size adjustment.
                  Runs inside the FlashAvatar python environment.
                  Writes dataset/<id>/{imgs,parsing,alpha}/ and
                  metrical-tracker/output/<id>/checkpoint/.

`finalize` is the only stage that interprets `--crop` / `--no-crop`, so
switching the crop flag never requires rerunning any stage above it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from . import crop as crop_mod
from . import deblur_filter as deblur_mod
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


def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--idname", required=True,
                     help="Identity name. Used as the dataset subdirectory.")
    sub.add_argument("--repo-root", type=Path, default=_repo_root(),
                     help="FlashAvatar repository root (default: autodetected).")
    sub.add_argument("--overwrite", action="store_true",
                     help="Re-run stages even when their outputs already exist.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess",
        description="FlashAvatar data preparation pipeline (split across "
                    "the metrical-tracker boundary).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- prepare ----
    pp = sub.add_parser(
        "prepare",
        help="Run stages that do NOT require metrical-tracker "
             "(extract + parsing + matting).",
    )
    _add_common(pp)
    pp.add_argument("--video", type=Path, default=None,
                    help="Source video. Skipped if raw/imgs already populated.")
    pp.add_argument("--device", default="cuda",
                    help="Torch device for parsing / matting (default: cuda).")
    pp.add_argument("--bisenet-weights", type=Path, default=None,
                    help="Path to the BiSeNet 79999_iter.pth checkpoint.")
    pp.add_argument("--rvm-variant", default="mobilenetv3",
                    choices=["mobilenetv3", "resnet50"])
    pp.add_argument("--skip-extract", action="store_true")
    pp.add_argument("--skip-parsing", action="store_true")
    pp.add_argument("--skip-matting", action="store_true")

    # ---- finalize ----
    fp = sub.add_parser(
        "finalize",
        help="Apply crop/resize + adjust camera K. Requires metrical-tracker "
             "output under metrical-tracker/output/<idname>/checkpoint_raw/.",
    )
    _add_common(fp)
    fp.add_argument("--size", type=int, default=512,
                    help="Output image side length (default: 512).")
    crop_group = fp.add_mutually_exclusive_group()
    crop_group.add_argument("--crop", dest="crop", action="store_true",
                            help="Tight face-centred square crop (default).")
    crop_group.add_argument("--no-crop", dest="crop", action="store_false",
                            help="Centre-square crop only, no face detection.")
    fp.set_defaults(crop=True)
    fp.add_argument("--crop-pad", type=float, default=0.15,
                    help="Fractional padding around the head bbox (crop=on).")

    # ---- filter-blur ----
    fb = sub.add_parser(
        "filter-blur",
        help="Score raw frames by face-region Laplacian variance and write "
             "raw/keep_list.txt (non-destructive). Skip this stage entirely "
             "to keep every frame.",
    )
    _add_common(fb)
    fb.add_argument("--percentile", type=float, default=15.0,
                    help="Drop frames whose variance is below this percentile "
                         "of the sequence (default: 15).")
    fb.add_argument("--absolute-threshold", type=float, default=None,
                    help="Override --percentile with an absolute variance "
                         "cutoff (frames with var >= cutoff are kept).")
    face_group = fb.add_mutually_exclusive_group()
    face_group.add_argument(
        "--face-mask", dest="face_mask", action="store_true",
        help="Measure variance only over the neck/head parsing mask "
             "(default). Falls back to whole-frame if parsing is missing.",
    )
    face_group.add_argument(
        "--no-face-mask", dest="face_mask", action="store_false",
        help="Measure variance over the whole frame.",
    )
    fb.set_defaults(face_mask=True)
    fb.add_argument("--preview-count", type=int, default=12,
                    help="Number of blurriest dropped frames to include in "
                         "raw/blur_preview.jpg (0 to skip).")
    fb.add_argument("--dry-run", action="store_true",
                    help="Report counts only; do not write keep_list / CSV / "
                         "preview.")

    return p


def cmd_prepare(args: argparse.Namespace) -> int:
    repo = Path(args.repo_root).resolve()
    dataset_dir = repo / "dataset" / args.idname
    raw = dataset_dir / "raw"
    raw_imgs = raw / "imgs"
    raw_parsing = raw / "parsing"
    raw_alpha = raw / "alpha"

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

    print(
        f"[prepare] done. Next: run metrical-tracker on\n"
        f"    {raw_imgs}\n"
        f"and place its output at\n"
        f"    {repo / 'metrical-tracker' / 'output' / args.idname / 'checkpoint_raw'}\n"
        f"then run `python scripts/preprocess.py finalize --idname {args.idname}`."
    )
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
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

    n_frames = len(list(raw_imgs.glob("*.jpg")))
    if n_frames == 0:
        print(f"no frames under {raw_imgs}; run `preprocess prepare` first.",
              file=sys.stderr)
        return 1

    # Accept tracker output placed at either checkpoint_raw/ or checkpoint/.
    if tracker_mod.count_frames(ckpt_raw) == 0 and \
            tracker_mod.count_frames(ckpt_final) > 0:
        print(f"[finalize] no {ckpt_raw.name}/ but {ckpt_final.name}/ exists; "
              f"treating the latter as the raw tracker output.")
        ckpt_raw = ckpt_final
    tracker_mod.verify_or_hint(ckpt_raw, n_frames)

    img_wh = _infer_image_size(raw_imgs)
    bbox = crop_mod.compute_bbox(
        raw_parsing if args.crop else None, img_wh, crop=args.crop,
    )
    print(f"[finalize] bbox x0={bbox.x0} y0={bbox.y0} size={bbox.size} "
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


def cmd_filter_blur(args: argparse.Namespace) -> int:
    repo = Path(args.repo_root).resolve()
    dataset_dir = repo / "dataset" / args.idname
    raw = dataset_dir / "raw"
    raw_imgs = raw / "imgs"
    raw_parsing = raw / "parsing"

    if not any(raw_imgs.glob("*.jpg")):
        print(f"no frames under {raw_imgs}; run `preprocess prepare` first.",
              file=sys.stderr)
        return 1

    parsing_dir: Path | None = None
    if args.face_mask:
        if raw_parsing.is_dir() and any(raw_parsing.glob("*_neckhead.png")):
            parsing_dir = raw_parsing
        else:
            print(f"[filter-blur] --face-mask set but no parsing output under "
                  f"{raw_parsing}; falling back to whole-frame variance.")

    report = deblur_mod.run_blur_filter(
        imgs_dir=raw_imgs,
        out_dir=raw,
        parsing_dir=parsing_dir,
        percentile=args.percentile,
        absolute=args.absolute_threshold,
        preview_count=args.preview_count,
        dry_run=args.dry_run,
    )

    mode = "dry-run" if args.dry_run else "write"
    thr = report["threshold"]
    thr_str = f"{thr:.3f}" if thr == thr else "n/a"  # NaN check
    print(f"[filter-blur] {mode}: kept {report['kept']}/{report['total']} "
          f"frames (dropped {report['dropped']}), threshold={thr_str}")
    if not args.dry_run:
        print(f"[filter-blur] wrote {report['keep_list']}")
        print(f"[filter-blur] wrote {report['csv']}")
        if report["preview"] is not None:
            print(f"[filter-blur] wrote {report['preview']}")
        print(
            "[filter-blur] `train.py` / `test.py` will auto-consume this "
            "keep_list; pass --ignore-keep-list to override. The "
            "metrical-tracker itself still sees every frame; re-run "
            "`filter-blur` after changing --percentile to regenerate."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "finalize":
        return cmd_finalize(args)
    if args.command == "filter-blur":
        return cmd_filter_blur(args)
    print(f"unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
