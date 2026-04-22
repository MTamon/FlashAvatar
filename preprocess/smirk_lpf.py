"""Temporal low-pass filter for SMIRK outputs across the frame sequence.

Motivating use-case: the downstream Listening Head Generation model trains
on 1st and 2nd temporal differences (velocity / acceleration) of FLAME
features. Even sub-pixel-level per-frame jitter in the SMIRK encoder output
amplifies catastrophically under differentiation — past runs produced
velocity/acceleration features that were dominated by noise rather than
signal. Applying a zero-phase FIR LPF at the teacher-data preparation stage
(this pipeline) is the standard fix.

Design choices, briefly:

* Linear-phase FIR via `scipy.signal.firwin` with Hamming window (default).
  Matches the filter the operator used successfully in past projects, so
  the tuning intuition (cutoff_freq / lpf_bin) carries over directly.
* Zero-phase application via `scipy.signal.filtfilt`. Because we are
  preparing offline training data, we can afford the forward-backward
  pass; it eliminates the group delay a pure causal FIR would introduce,
  so the filtered signal still aligns temporally with the source video
  and with any downstream derivative features.
* FPS-aware: cutoff is specified in Hz and normalised by the caller-
  supplied `fps` (the SMIRK/FlashAvatar pipeline is frame-indexed and
  doesn't otherwise need `fps`, so it must be provided explicitly when
  LPF is requested).
* Operates in axis-angle space for `pose_params` / `jaw_params`. This is
  safe only while rotations stay well inside the (-π, π) branch (typical
  for head rotation and jaw motion). Filtering matrices / quaternions
  would be more principled but adds dependencies; revisit if any sample
  trips the branch cut.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Channel name -> FrameResult attribute. Exposed as the `--lpf-channels`
# vocabulary so the CLI help can enumerate it. `shape` is intentionally
# omitted: `canonicalize_shape` collapses it to a single per-identity
# value before `.frame` is written, so there's nothing to smooth.
CHANNEL_ATTR = {
    "cam": "cam",                  # (3,) weak-perspective [s, tx, ty]
    "pose": "pose_params",         # (3,) axis-angle head rotation
    "exp": "expression_params",    # (50,)
    "jaw": "jaw_params",           # (3,) axis-angle
    "eyelids": "eyelid_params",    # (2,) in [0, 1]
}


@dataclass
class LpfConfig:
    cutoff_hz: float
    fps: float
    window_sec: float = 1.0
    channels: tuple[str, ...] = ("cam", "pose")

    def validate(self) -> None:
        nyq = self.fps / 2.0
        if self.cutoff_hz <= 0:
            raise ValueError(f"--lpf-cutoff must be > 0 (got {self.cutoff_hz})")
        if self.cutoff_hz >= nyq:
            raise ValueError(
                f"--lpf-cutoff {self.cutoff_hz} Hz >= Nyquist {nyq} Hz "
                f"(fps={self.fps}); widen fps or lower cutoff.")
        if self.window_sec <= 0:
            raise ValueError(
                f"--lpf-window-sec must be > 0 (got {self.window_sec})")
        bad = [c for c in self.channels if c not in CHANNEL_ATTR]
        if bad:
            raise ValueError(
                f"unknown LPF channel(s): {bad}. "
                f"available: {sorted(CHANNEL_ATTR)}")


def design_firwin(cfg: LpfConfig) -> np.ndarray:
    """Return the FIR coefficients for the configured LPF.

    Filter length is rounded to the nearest odd sample count >= 3 so the
    filter is symmetric (linear phase).
    """
    from scipy.signal import firwin

    n = int(round(cfg.window_sec * cfg.fps))
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1
    return np.asarray(firwin(n, cfg.cutoff_hz / (cfg.fps / 2.0)),
                      dtype=np.float64)


def smooth_payloads(payloads, cfg: LpfConfig) -> None:
    """Apply the configured LPF to each channel across all payloads.

    Mutates `payloads[i].result` in place so downstream `.frame` writers
    and demo renderers pick up the smoothed values automatically.

    Raises if the sequence is too short for `filtfilt`'s default padding
    length (3 * filter_length); callers should trap and either shorten
    `--lpf-window-sec` or disable LPF on short clips.
    """
    from scipy.signal import filtfilt

    cfg.validate()
    coef = design_firwin(cfg)
    n = len(payloads)
    # filtfilt uses `padlen = 3 * max(len(a), len(b)) - 1` by default.
    # Signal must be strictly longer than padlen.
    if n <= 3 * coef.shape[0]:
        raise ValueError(
            f"sequence of {n} frames is too short for LPF with filter "
            f"length {coef.shape[0]} (needs > {3 * coef.shape[0]}). "
            f"Shorten --lpf-window-sec or drop --lpf-cutoff for this clip.",
        )

    print(f"[smirk/lpf] cutoff={cfg.cutoff_hz}Hz fps={cfg.fps} "
          f"length={coef.shape[0]} channels={list(cfg.channels)}")

    for ch in cfg.channels:
        attr = CHANNEL_ATTR[ch]
        stacked = np.stack(
            [np.asarray(getattr(p.result, attr), dtype=np.float64)
             for p in payloads],
            axis=0,
        )  # (N, D)
        smoothed = filtfilt(coef, [1.0], stacked, axis=0)
        # Preserve original dtype per-frame (typical: float32).
        orig_dtype = np.asarray(
            getattr(payloads[0].result, attr)).dtype
        for i, p in enumerate(payloads):
            setattr(p.result, attr, smoothed[i].astype(orig_dtype))
