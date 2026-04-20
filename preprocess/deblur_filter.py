"""Motion-blur frame filtering via face-region Laplacian variance.

For each raw frame we compute the variance of the 4-neighbour discrete
Laplacian over the neck/head parsing mask (Pech-Pacheco 2000). Frames whose
variance falls below a percentile of the sequence (default: 15th) are marked
as blurry and excluded from the resulting ``raw/keep_list.txt``.

The filter is non-destructive: ``raw/imgs/`` is never modified. Downstream
stages opt in by consuming ``keep_list.txt`` explicitly.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class FrameScore:
    name: str
    variance: float
    face_pixels: int


def _laplacian_variance(gray: np.ndarray,
                        mask: np.ndarray | None) -> tuple[float, int]:
    g = gray.astype(np.float32)
    lap = (g[2:, 1:-1] + g[:-2, 1:-1] + g[1:-1, 2:] + g[1:-1, :-2]
           - 4.0 * g[1:-1, 1:-1])
    if mask is None:
        return float(lap.var()), int(lap.size)
    m = mask[1:-1, 1:-1] > 0
    if not m.any():
        return float("nan"), 0
    sel = lap[m]
    return float(sel.var()), int(sel.size)


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def _load_face_mask(parsing_dir: Path, stem: str,
                    shape: tuple[int, int]) -> np.ndarray | None:
    p = parsing_dir / f"{stem}_neckhead.png"
    if not p.is_file():
        return None
    with Image.open(p) as im:
        im = im.convert("L")
        if im.size != (shape[1], shape[0]):
            im = im.resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(im)


def compute_blur_scores(imgs_dir: Path,
                        parsing_dir: Path | None) -> list[FrameScore]:
    frames = sorted(Path(imgs_dir).glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no *.jpg frames under {imgs_dir}")

    scores: list[FrameScore] = []
    for f in tqdm(frames, desc="blur"):
        gray = _load_gray(f)
        mask = (_load_face_mask(parsing_dir, f.stem, gray.shape)
                if parsing_dir is not None else None)
        var, npx = _laplacian_variance(gray, mask)
        scores.append(FrameScore(name=f.stem, variance=var, face_pixels=npx))
    return scores


def select_keep(scores: Sequence[FrameScore], percentile: float,
                absolute: float | None) -> tuple[set[str], float]:
    """Return (keep_names, threshold).

    Frames with NaN variance (no face mask coverage) are always dropped.
    """
    valid = [s for s in scores
             if np.isfinite(s.variance) and s.face_pixels > 0]
    if not valid:
        return set(), float("nan")

    if absolute is not None:
        threshold = float(absolute)
    else:
        vals = np.asarray([s.variance for s in valid], dtype=np.float64)
        threshold = float(np.percentile(vals, percentile))

    keep = {s.name for s in valid if s.variance >= threshold}
    return keep, threshold


def _montage(paths: list[Path], cell: int, cols: int) -> Image.Image:
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cell * cols, cell * rows), (0, 0, 0))
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB").copy()
        im.thumbnail((cell, cell), Image.BICUBIC)
        r, c = divmod(i, cols)
        x = c * cell + (cell - im.width) // 2
        y = r * cell + (cell - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas


def run_blur_filter(imgs_dir: Path, out_dir: Path,
                    parsing_dir: Path | None = None,
                    percentile: float = 15.0,
                    absolute: float | None = None,
                    preview_count: int = 12,
                    dry_run: bool = False) -> dict:
    imgs_dir = Path(imgs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = compute_blur_scores(imgs_dir, parsing_dir)
    keep, threshold = select_keep(scores, percentile, absolute)

    report: dict = {
        "total": len(scores),
        "kept": sum(1 for s in scores if s.name in keep),
        "dropped": sum(1 for s in scores if s.name not in keep),
        "threshold": threshold,
        "csv": None,
        "keep_list": None,
        "preview": None,
    }
    if dry_run:
        return report

    csv_path = out_dir / "blur_scores.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "laplacian_variance", "face_pixels", "kept"])
        for s in scores:
            v = "" if not np.isfinite(s.variance) else f"{s.variance:.6f}"
            w.writerow([s.name, v, s.face_pixels, int(s.name in keep)])

    keep_path = out_dir / "keep_list.txt"
    with keep_path.open("w") as fh:
        for s in scores:
            if s.name in keep:
                fh.write(f"{s.name}\n")

    preview_path: Path | None = None
    if preview_count > 0:
        dropped = [s for s in scores if s.name not in keep]
        dropped.sort(
            key=lambda s: (0, s.variance) if np.isfinite(s.variance)
            else (1, 0.0),
        )
        worst = dropped[:preview_count]
        if worst:
            cols = min(4, len(worst))
            paths = [imgs_dir / f"{s.name}.jpg" for s in worst]
            montage = _montage(paths, cell=256, cols=cols)
            preview_path = out_dir / "blur_preview.jpg"
            montage.save(preview_path, quality=90)

    report["csv"] = csv_path
    report["keep_list"] = keep_path
    report["preview"] = preview_path
    return report
