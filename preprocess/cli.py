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
from . import smirk_tracker as smirk_mod


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

    # ---- smirk (alternative FLAME tracker) ----
    sm = sub.add_parser(
        "smirk",
        help="Alternative FLAME tracker using SMIRK (MTamon/smirk@release/"
             "cuda128). More robust to large head rotation / motion blur "
             "than metrical-tracker. Requires a prior "
             "`bash scripts/setup_smirk.sh`.",
    )
    _add_common(sm)
    sm.add_argument("--smirk-root", type=Path,
                    default=Path("external/smirk"),
                    help="SMIRK checkout root (default: external/smirk).")
    sm.add_argument("--checkpoint", type=Path, default=None,
                    help="SMIRK_em1.pt path. "
                         "Default: <smirk-root>/pretrained_models/SMIRK_em1.pt")
    sm.add_argument("--device", default="cuda")
    sm.add_argument("--batch-size", type=int, default=8)
    sm.add_argument("--crop-scale", type=float, default=1.4,
                    help="SMIRK face-crop scale factor (default 1.4, matches "
                         "SMIRK demo).")
    sm.add_argument("--focal-px", type=float, default=5000.0,
                    help="Synthesized perspective focal in pixels (larger = "
                         "closer to weak-perspective; default 5000).")
    sm.add_argument("--shape-frames", type=int, default=150,
                    help="Canonicalize FLAME identity across the first N "
                         "detected frames (median).")
    sm.add_argument("--eye-mode", choices=["blendshapes", "zero"],
                    default="zero",
                    help="Source of FLAME eye_pose. 'zero' (default) writes "
                         "identity — safe default, matches the pre-eye-pose "
                         "behaviour. 'blendshapes' enables eye tracking: "
                         "per-eye yaw/pitch are derived from MediaPipe Face "
                         "Landmarker ARKit blendshapes "
                         "(eyeLookIn/Out/Up/Down*), which requires "
                         "face_landmarker.task from SMIRK's quick_install.sh. "
                         "Pick 'blendshapes' if you need the trained avatar "
                         "to track gaze.")
    sm.add_argument("--verify-dir", type=Path, default=None,
                    help="If set, dump landmark reprojection stats + overlay "
                         "JPEGs to this directory for sanity check.")
    sm.add_argument("--demo-video", type=Path, default=None,
                    help="If set, render an overlay mp4 at this path with "
                         "bbox + MediaPipe landmarks + FLAME mesh reprojection "
                         "drawn on the source frames. Writes a sibling "
                         "`<stem>_stats.csv` with per-frame bbox/translation "
                         "first-differences so jitter is visible as a time "
                         "series. Useful to diagnose whether jitter originates "
                         "at the landmark / crop / encoder / camera layer.")
    sm.add_argument("--demo-fps", type=float, default=25.0,
                    help="Playback fps for --demo-video (default 25). The "
                         "SMIRK pipeline is frame-indexed, so this is a hint "
                         "for the mp4 encoder only.")

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


def cmd_smirk(args: argparse.Namespace) -> int:
    repo = Path(args.repo_root).resolve()
    raw_imgs = smirk_mod.raw_imgs_dir(repo, args.idname)
    ckpt_raw = smirk_mod.checkpoint_raw_dir(repo, args.idname)

    if not any(raw_imgs.glob("*.jpg")):
        print(f"no frames under {raw_imgs}; run `preprocess prepare` first.",
              file=sys.stderr)
        return 1

    smirk_root = Path(args.smirk_root)
    if not smirk_root.is_absolute():
        smirk_root = repo / smirk_root
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = smirk_root / "pretrained_models" / "SMIRK_em1.pt"
    if not ckpt_path.is_file():
        print(f"error: SMIRK checkpoint not found at {ckpt_path}.\n"
              f"  Run `bash scripts/setup_smirk.sh` to clone + install + "
              f"download weights.", file=sys.stderr)
        return 1

    cfg = smirk_mod.SmirkConfig(
        smirk_root=smirk_root,
        checkpoint=ckpt_path,
        device=args.device,
        crop_scale=args.crop_scale,
        focal_px=args.focal_px,
        shape_frames=args.shape_frames,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        eye_mode=args.eye_mode,
    )
    print(f"[smirk] {raw_imgs} -> {ckpt_raw}")
    n = smirk_mod.run(cfg, raw_imgs, ckpt_raw,
                      verify_dir=args.verify_dir,
                      demo_path=args.demo_video,
                      demo_fps=args.demo_fps)
    print(f"[smirk] wrote {n} .frame files")
    print(f"[smirk] next: python scripts/preprocess.py finalize "
          f"--idname {args.idname}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "finalize":
        return cmd_finalize(args)
    if args.command == "filter-blur":
        return cmd_filter_blur(args)
    if args.command == "smirk":
        return cmd_smirk(args)
    print(f"unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
