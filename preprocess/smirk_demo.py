"""Demo renderer: overlay SMIRK outputs on the source frames and encode an
mp4 so an operator can eyeball how stable the crop bbox, the detected
MediaPipe landmarks, and the reprojected FLAME mesh are frame-to-frame.

Primary use-case: diagnosing jitter. The `.frame` files carry per-frame
R/t/bbox_size; without a side-by-side overlay it's very hard to tell
whether the jitter originates at the landmark layer, the crop/tform
layer, the encoder, or the weak-perspective -> perspective bridge. This
module draws all three layers together so they can be compared visually.

Also writes a `demo_stats.csv` with per-frame bbox / camera-translation
first-differences so the jitter is visible as a time series and not just
on playback.

The implementation deliberately mirrors `smirk_verify.py` rather than
refactoring common helpers: keeping the demo self-contained means the
verify script (which is a sanity gate) doesn't pick up a dependency on
OpenCV and the demo can be removed without touching the tracker.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def dump_demo(payloads, shape, img_size, cfg, demo_path: Path,
              fps: float = 25.0, draw_bbox: bool = True,
              draw_landmarks: bool = True, draw_mesh: bool = True,
              mesh_stride: int = 8, lock_bbox: bool = False,
              smooth_bbox: int = 0) -> None:
    """Write an overlay mp4 + stats CSV alongside it.

    Args:
        payloads: list of `FramePayload` as built by `smirk_tracker.run`.
        shape: (300,) canonicalized FLAME identity (np.float32).
        img_size: (W, H) of the source frames.
        cfg: `SmirkConfig` (reads cfg.device, cfg.focal_px, cfg.eye_mode).
        demo_path: output mp4 path. The CSV is written next to it as
                   `<stem>_stats.csv`.
        fps: encoder fps. The SMIRK pipeline is frame-indexed (not time-
             indexed) so this is purely a playback hint.
        mesh_stride: project every Nth FLAME vertex. V=5023 so stride=8
                     gives ~630 points, dense enough to see shape drift
                     but sparse enough not to saturate the frame.
        lock_bbox: if True, reuse the first frame's bbox (center + size) for
                   every subsequent frame. Diagnostic: if jitter disappears,
                   bbox instability is the dominant source. The MediaPipe
                   overlay still shows the per-frame detected landmarks so
                   the viewer can see the face drift away from the locked
                   bbox.
        smooth_bbox: if > 0, replace each frame's bbox_{center,size} with a
                   centered moving average over a (2N+1)-frame window. Use
                   this as a middle ground between lock-bbox and no-op.
                   Causal/non-causal doesn't matter for this diagnostic — we
                   have the whole sequence in memory.
    """
    demo_path = Path(demo_path)
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = demo_path.with_name(demo_path.stem + "_stats.csv")

    flame = _load_flame(cfg.device)
    w, h = img_size

    # Pre-compute the bbox series once so lock / smooth apply uniformly.
    bbox_centers, bbox_sizes = _prepare_bbox_series(
        payloads, lock_bbox=lock_bbox, smooth_bbox=smooth_bbox,
    )
    if lock_bbox:
        print(f"[smirk/demo] --lock-bbox: using frame 0 bbox for all "
              f"{len(payloads)} frames "
              f"(center={bbox_centers[0]}, size={bbox_sizes[0]:.1f})")
    elif smooth_bbox > 0:
        print(f"[smirk/demo] --smooth-bbox {smooth_bbox}: centred moving "
              f"average over {2 * smooth_bbox + 1}-frame window")

    # mp4v is widely available in OpenCV wheels and plays back in browsers /
    # VS Code previews without a separate codec install. Framerate is a
    # hint only — the pipeline itself is frame-indexed.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(demo_path), fourcc, float(fps), (w, h), True)
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {demo_path} — check that "
            f"OpenCV was built with mp4v support, or change the extension.",
        )

    prev_bbox_center: np.ndarray | None = None
    prev_bbox_size: float | None = None
    prev_t: np.ndarray | None = None
    stats_rows: list[dict] = []

    try:
        for i, p in enumerate(tqdm(payloads, desc="smirk/demo")):
            r = p.result
            bc_used = bbox_centers[i]
            bs_used = bbox_sizes[i]
            K, R, t = _rebuild_camera(r, w, h, cfg.focal_px,
                                      bbox_center=bc_used, bbox_size=bs_used)

            # FLAME mesh in canonical frame.
            X_pix = _project_flame_mesh(flame, shape, r, K, R, t, cfg)

            # Compose the overlay. cv2 works in BGR; we read RGB from disk.
            frame_bgr = cv2.imread(str(p.src_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                # Fall back to PIL path (jpg odd-quality on some installs).
                from PIL import Image as _Image
                frame_bgr = cv2.cvtColor(
                    np.array(_Image.open(p.src_path).convert("RGB")),
                    cv2.COLOR_RGB2BGR,
                )
            _resize_if_needed(frame_bgr, (w, h))

            if draw_mesh:
                _draw_mesh(frame_bgr, X_pix, stride=mesh_stride)
            if draw_landmarks and r.landmarks is not None:
                _draw_landmarks(frame_bgr, r.landmarks)
            if draw_bbox:
                # With lock/smooth, show BOTH: thin raw bbox (for reference)
                # and thick used-for-camera bbox (what actually drives t).
                if lock_bbox or smooth_bbox > 0:
                    _draw_bbox(frame_bgr, r.bbox_center, r.bbox_size,
                               detected=r.detected, thickness=1, dim=True)
                _draw_bbox(frame_bgr, bc_used, bs_used,
                           detected=r.detected, thickness=2)

            # Top-left HUD: frame index + detection flag + used bbox size +
            # derived Z so the reader can see the Z response directly.
            hud = (f"f={p.idx:05d}  det={int(r.detected)}  "
                   f"bbox={bs_used:6.1f}px  Z={t[2]:7.3f}")
            _draw_hud(frame_bgr, hud)

            writer.write(frame_bgr)

            # Per-frame jitter stats. Columns report BOTH the raw MediaPipe-
            # derived bbox (bbox_*) and the bbox actually fed into `_build_t`
            # (used_bbox_*), so --lock-bbox / --smooth-bbox runs remain
            # comparable to the baseline CSV.
            d_c = (np.zeros(2) if prev_bbox_center is None
                   else bc_used - prev_bbox_center)
            d_s = (0.0 if prev_bbox_size is None
                   else bs_used - prev_bbox_size)
            d_t = np.zeros(3) if prev_t is None else t - prev_t
            stats_rows.append({
                "idx": p.idx,
                "detected": int(r.detected),
                "bbox_cx": f"{r.bbox_center[0]:.3f}",
                "bbox_cy": f"{r.bbox_center[1]:.3f}",
                "bbox_size": f"{r.bbox_size:.3f}",
                "used_bbox_cx": f"{bc_used[0]:.3f}",
                "used_bbox_cy": f"{bc_used[1]:.3f}",
                "used_bbox_size": f"{bs_used:.3f}",
                "t_x": f"{t[0]:.5f}",
                "t_y": f"{t[1]:.5f}",
                "t_z": f"{t[2]:.5f}",
                "d_used_bbox_cx": f"{d_c[0]:.3f}",
                "d_used_bbox_cy": f"{d_c[1]:.3f}",
                "d_used_bbox_size": f"{d_s:.3f}",
                "d_t_x": f"{d_t[0]:.5f}",
                "d_t_y": f"{d_t[1]:.5f}",
                "d_t_z": f"{d_t[2]:.5f}",
                "speed_bbox_center": f"{float(np.linalg.norm(d_c)):.3f}",
                "speed_t": f"{float(np.linalg.norm(d_t)):.5f}",
            })
            prev_bbox_center = bc_used.copy()
            prev_bbox_size = float(bs_used)
            prev_t = t.copy()
    finally:
        writer.release()

    with open(stats_path, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(stats_rows[0].keys()))
        w_.writeheader()
        w_.writerows(stats_rows)

    # Quick textual summary — surfaces jitter without opening the video.
    _print_summary(stats_rows)
    print(f"[smirk/demo] wrote {demo_path}")
    print(f"[smirk/demo] wrote {stats_path}")


# --- bbox diagnostics -----------------------------------------------------

def _prepare_bbox_series(payloads, *, lock_bbox: bool, smooth_bbox: int
                         ) -> tuple[list[np.ndarray], list[float]]:
    """Return the (bbox_center, bbox_size) actually fed into `_build_t` per
    frame, after applying --lock-bbox / --smooth-bbox.

    This is a pure function over the payload list; it doesn't touch the
    stored FrameResult.bbox_* fields (the raw detection is still drawn in
    the overlay for visual reference).
    """
    raw_centers = [p.result.bbox_center.astype(np.float64) for p in payloads]
    raw_sizes = [float(p.result.bbox_size) for p in payloads]
    n = len(payloads)

    if lock_bbox:
        c0, s0 = raw_centers[0], raw_sizes[0]
        return [c0.copy() for _ in range(n)], [s0 for _ in range(n)]

    if smooth_bbox > 0:
        w_ = int(smooth_bbox)
        # Centred moving average. Offline only — diagnostic, not the
        # real-time path.
        centers = []
        sizes = []
        cx = np.array([c[0] for c in raw_centers])
        cy = np.array([c[1] for c in raw_centers])
        sz = np.array(raw_sizes)
        for i in range(n):
            lo = max(0, i - w_)
            hi = min(n, i + w_ + 1)
            centers.append(np.array([cx[lo:hi].mean(), cy[lo:hi].mean()]))
            sizes.append(float(sz[lo:hi].mean()))
        return centers, sizes

    return raw_centers, raw_sizes


# --- drawing helpers ------------------------------------------------------

# BGR colors (we operate on cv2's BGR frames).
_COL_BBOX_OK = (255, 255, 0)     # cyan-ish: MediaPipe detected on this frame
_COL_BBOX_REUSE = (0, 140, 255)  # amber: re-using previous landmarks
_COL_LANDMARKS = (60, 60, 255)   # red: MediaPipe landmarks
_COL_MESH = (0, 220, 0)          # green: projected FLAME mesh
_COL_HUD_BG = (0, 0, 0)
_COL_HUD_FG = (255, 255, 255)


def _draw_bbox(frame_bgr: np.ndarray, center: np.ndarray,
               size: float, detected: bool,
               thickness: int = 2, dim: bool = False) -> None:
    cx, cy = float(center[0]), float(center[1])
    half = float(size) / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))
    col = _COL_BBOX_OK if detected else _COL_BBOX_REUSE
    if dim:
        col = tuple(int(c * 0.45) for c in col)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), col, thickness)
    cv2.drawMarker(frame_bgr, (int(round(cx)), int(round(cy))), col,
                   markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)


def _draw_landmarks(frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
    for x, y in landmarks:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(frame_bgr, (int(round(float(x))), int(round(float(y)))),
                   1, _COL_LANDMARKS, -1, lineType=cv2.LINE_AA)


def _draw_mesh(frame_bgr: np.ndarray, X_pix: np.ndarray,
               stride: int) -> None:
    h, w = frame_bgr.shape[:2]
    pts = X_pix[::max(1, int(stride))]
    for x, y in pts:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(frame_bgr, (xi, yi), 1, _COL_MESH, -1,
                       lineType=cv2.LINE_AA)


def _draw_hud(frame_bgr: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = 4
    cv2.rectangle(frame_bgr, (8, 8), (8 + tw + 2 * pad,
                                      8 + th + 2 * pad + base), _COL_HUD_BG, -1)
    cv2.putText(frame_bgr, text, (8 + pad, 8 + pad + th), font, scale,
                _COL_HUD_FG, thick, cv2.LINE_AA)


def _resize_if_needed(frame_bgr: np.ndarray,
                      target_wh: tuple[int, int]) -> None:
    # Sanity: the pipeline assumes all source frames share the same size.
    # Rather than silently resize (which would misalign landmarks), raise.
    h, w = frame_bgr.shape[:2]
    tw, th = target_wh
    if (w, h) != (tw, th):
        raise RuntimeError(
            f"frame size {(w, h)} != expected {(tw, th)} — demo overlay "
            f"assumes a homogeneous sequence. Re-run `preprocess prepare` "
            f"or check for mixed-resolution sources.",
        )


# --- FLAME / camera helpers ----------------------------------------------

def _load_flame(device: str):
    from flame import FLAME_mica, parse_args  # type: ignore
    cfg = parse_args()
    return FLAME_mica(cfg).to(device).eval()


def _rebuild_camera(r, w: int, h: int, focal_px: float,
                    bbox_center: np.ndarray | None = None,
                    bbox_size: float | None = None):
    """Rebuild (K, R, t) from a FrameResult. `bbox_center` / `bbox_size`
    can be overridden by the caller (used by --lock-bbox / --smooth-bbox
    to feed a stabilized bbox into `_build_t` without mutating the
    FrameResult itself).
    """
    from preprocess.smirk_convert import _build_K, _build_R, _build_t
    bc = r.bbox_center if bbox_center is None else bbox_center
    bs = r.bbox_size if bbox_size is None else bbox_size
    K = _build_K(w, h, focal_px)
    R = _build_R(r.pose_params)
    t = _build_t(r.cam, bc, bs, w, h, focal_px)
    return K, R, t


def _project_flame_mesh(flame, shape_np, r, K, R, t, cfg) -> np.ndarray:
    from preprocess.smirk_convert import (
        _pad_expression, _axis_angle_to_rot6d, _default_eye_pose_6d,
        eye_pose_6d_from_blendshapes,
    )
    device = cfg.device
    with torch.no_grad():
        shape_t = torch.from_numpy(shape_np).float().unsqueeze(0).to(device)
        exp_t = torch.from_numpy(
            _pad_expression(r.expression_params, 100),
        ).float().unsqueeze(0).to(device)
        jaw_t = torch.from_numpy(
            _axis_angle_to_rot6d(r.jaw_params),
        ).float().unsqueeze(0).to(device)
        eyelid_t = torch.from_numpy(
            np.clip(r.eyelid_params, 0.0, 1.0).astype(np.float32),
        ).unsqueeze(0).to(device)
        if getattr(cfg, "eye_mode", "zero") == "blendshapes":
            eyes_np = eye_pose_6d_from_blendshapes(r.blendshapes)
        else:
            eyes_np = _default_eye_pose_6d()
        eyes_t = torch.from_numpy(eyes_np).float().unsqueeze(0).to(device)
        verts = flame.forward_geo(
            shape_t, expression_params=exp_t,
            jaw_pose_params=jaw_t, eye_pose_params=eyes_t,
            eyelid_params=eyelid_t,
        )[0].cpu().numpy()  # (V, 3) canonical frame

    X_cam = (R @ verts.T).T + t[None, :]
    X_pix = (K @ X_cam.T).T
    z = np.clip(X_pix[:, 2:3], 1e-6, None)
    return X_pix[:, :2] / z


# --- summary --------------------------------------------------------------

def _print_summary(rows: list[dict]) -> None:
    if len(rows) < 2:
        return
    speeds_c = np.array([float(r["speed_bbox_center"]) for r in rows[1:]])
    speeds_s = np.array([abs(float(r["d_bbox_size"])) for r in rows[1:]])
    speeds_t = np.array([float(r["speed_t"]) for r in rows[1:]])
    det_rate = np.mean([int(r["detected"]) for r in rows])

    def q(a):
        return (float(np.median(a)), float(np.percentile(a, 95)),
                float(a.max()))

    m_c, p95_c, mx_c = q(speeds_c)
    m_s, p95_s, mx_s = q(speeds_s)
    m_t, p95_t, mx_t = q(speeds_t)
    print(
        f"[smirk/demo] frames={len(rows)} det_rate={det_rate:.3f}\n"
        f"            bbox_center |diff|  median={m_c:.2f}px  "
        f"p95={p95_c:.2f}px  max={mx_c:.2f}px\n"
        f"            bbox_size   |diff|  median={m_s:.2f}px  "
        f"p95={p95_s:.2f}px  max={mx_s:.2f}px\n"
        f"            t           |diff|  median={m_t:.4f}     "
        f"p95={p95_t:.4f}     max={mx_t:.4f}\n"
        f"            (high p95/max vs median = jitter)",
    )
