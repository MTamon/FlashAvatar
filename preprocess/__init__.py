"""Data preparation / preprocessing pipeline for FlashAvatar training.

Stages:
  1. extract  : video (mp4 etc.) -> imgs/XXXXX.jpg (native resolution)
  2. parsing  : imgs -> parsing/XXXXX_{neckhead,mouth}.png (BiSeNet)
  3. matting  : imgs -> alpha/XXXXX.jpg (RobustVideoMatting)
  4. tracker  : imgs -> metrical-tracker/output/<id>/checkpoint/XXXXX.frame
  5. crop     : down-stream formatting. Takes raw/ outputs + .frame files and
                produces <id>/imgs, <id>/parsing, <id>/alpha at --size x --size,
                and rewrites .frame files so opencv.K / img_size match.

Design: stages 2-4 always run in the *original* camera coordinate system
(full-resolution frames). Only stage 5 decides whether a tight face-centred
crop is applied (`--crop`) or a plain centre-square is used (`--no-crop`).
This keeps the per-stage code independent of the crop flag.
"""
