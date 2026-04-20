#!/usr/bin/env bash
# Run metrical-tracker (MTamon/metrical-tracker@cuda128) against
# `dataset/<idname>/raw/imgs/`, and stage the result at
# `metrical-tracker/output/<idname>/checkpoint_raw/` so
# `preprocess.py finalize` can consume it.
#
# The cuda128 fork shares FlashAvatar's env (torch 2.9.1 / cu128 / py3.11),
# so this script does NOT activate a separate env — it expects to be run
# inside the active FlashAvatar env (the one from install_128.sh).
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/run_tracker.sh <idname> [extra tracker args...]
#
# Overridable environment variables:
#   TRACKER_DIR   tracker install dir (default: external/metrical-tracker)

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <idname> [extra tracker args...]" >&2
  exit 1
fi

IDNAME=$1
shift || true

TRACKER_DIR=${TRACKER_DIR:-external/metrical-tracker}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
abs_tracker_dir="$repo_root/$TRACKER_DIR"

if [ ! -d "$abs_tracker_dir" ]; then
  echo "error: $abs_tracker_dir not found. Run scripts/setup_metrical_tracker.sh first." >&2
  exit 1
fi

imgs_dir="$repo_root/dataset/$IDNAME/raw/imgs"
out_dir="$repo_root/metrical-tracker/output/$IDNAME"

if [ ! -d "$imgs_dir" ] || [ -z "$(ls -A "$imgs_dir" 2>/dev/null)" ]; then
  echo "error: no frames at $imgs_dir." >&2
  echo "Run \`python scripts/preprocess.py prepare --idname $IDNAME --video ...\` first." >&2
  exit 1
fi

# Sanity-check the env has the packages tracker.py imports.
missing=()
for mod in cv2 mediapipe face_alignment torch numpy pytorch3d chumpy tensorboard; do
  if ! python -c "import $mod" >/dev/null 2>&1; then
    missing+=("$mod")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "error: the active python env is missing: ${missing[*]}" >&2
  echo "Repair with:" >&2
  echo "    bash scripts/setup_metrical_tracker.sh" >&2
  exit 1
fi

mkdir -p "$out_dir"
cd "$abs_tracker_dir"

# Belt-and-suspenders: make sure datasets/ is a regular package so that
# HuggingFace's `datasets` (if it happens to be installed in the active
# env) doesn't shadow the tracker's local datasets module. The setup
# script already does this, but run after a manual re-clone or partial
# setup without it, the file may be missing.
if [ -d datasets ] && [ ! -f datasets/__init__.py ]; then
  touch datasets/__init__.py
fi

echo "[tracker] $imgs_dir -> $out_dir"
python tracker.py \
    --input_dir "$imgs_dir" \
    --output_dir "$out_dir" \
    "$@"

# tracker writes to checkpoint/; rename to checkpoint_raw/ so
# `preprocess finalize` treats it as the pre-crop source of truth.
if [ -d "$out_dir/checkpoint" ] && [ ! -d "$out_dir/checkpoint_raw" ]; then
  mv "$out_dir/checkpoint" "$out_dir/checkpoint_raw"
  echo "[tracker] moved checkpoint/ -> checkpoint_raw/"
fi

cat <<EOM

[done] tracker output at: $out_dir/checkpoint_raw/

Next:

  python scripts/preprocess.py finalize --idname $IDNAME

EOM
