#!/usr/bin/env bash
# Run metrical-tracker (MTamon/metrical-tracker@claude0420) against
# `dataset/<idname>/raw/imgs/`, and stage the result at
# `metrical-tracker/output/<idname>/checkpoint_raw/` so
# `preprocess.py finalize` can consume it.
#
# The claude0420 fork shares FlashAvatar's env (torch 2.9.1 / cu128 /
# py3.11), so this script does NOT activate a separate env — it expects
# to be run inside the active FlashAvatar env (from install_128.sh).
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/run_tracker.sh <idname> [extra tracker args...]
#
# If <actor>/identity.npy is missing, MICA is invoked on the first frame
# to generate it (a 300-dim FLAME shape code). Requires that MICA has
# been set up via `scripts/setup_metrical_tracker.sh`.
#
# Overridable environment variables:
#   TRACKER_DIR   tracker install dir (default: external/metrical-tracker)
#   MICA_DIR      MICA install dir    (default: external/MICA)

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <idname> [extra tracker args...]" >&2
  exit 1
fi

IDNAME=$1
shift || true

TRACKER_DIR=${TRACKER_DIR:-external/metrical-tracker}
MICA_DIR=${MICA_DIR:-external/MICA}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
abs_tracker_dir="$repo_root/$TRACKER_DIR"
abs_mica_dir="$repo_root/$MICA_DIR"

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

# MTamon fork's CLI is `--cfg <yaml>`, not --input_dir/--output_dir.
# Expected layout (see datasets/generate_dataset.py):
#   <actor>/source/*.jpg|png   frames
#   <actor>/kpt/ , bbox.pt     written by the dataset generator
# Output goes to <save_folder>/<cfg-stem>/{checkpoint,input,pyramid,...}.
# Map our pipeline to this by:
#   1. staging  external/metrical-tracker/input/<IDNAME>/source -> dataset/<IDNAME>/raw/imgs  (symlink)
#   2. writing  configs/actors/<IDNAME>.yml with actor=<stage>, save_folder=<out_parent>/
#   3. invoking python tracker.py --cfg configs/actors/<IDNAME>.yml
# cfg-stem = <IDNAME>, so tracker writes <out_parent>/<IDNAME>/checkpoint/.
stage_dir="$abs_tracker_dir/input/$IDNAME"
mkdir -p "$stage_dir"
ln -sfn "$imgs_dir" "$stage_dir/source"

# tracker.py needs <actor>/identity.npy (a 300-dim FLAME shape code from
# MICA). Generate it from the first frame on demand. Skip if already
# present so re-runs are fast and allow hand-provided identities.
if [ ! -f "$stage_dir/identity.npy" ]; then
  if [ ! -d "$abs_mica_dir" ]; then
    echo "error: $abs_mica_dir not found. Run scripts/setup_metrical_tracker.sh first." >&2
    exit 1
  fi
  if [ ! -f "$abs_mica_dir/data/pretrained/mica.tar" ]; then
    echo "error: MICA checkpoint missing at $abs_mica_dir/data/pretrained/mica.tar." >&2
    echo "Re-run scripts/setup_metrical_tracker.sh (or unset SKIP_ASSETS)." >&2
    exit 1
  fi

  # Pick the lexically first frame. A plain `ls | sort | head` pipeline
  # would trigger SIGPIPE on `ls` / `sort` once `head` closes after one
  # line — combined with `set -o pipefail`, the whole script would die
  # with exit 141 on large image dirs. Use a bash array of the glob
  # expansion instead (already lexically sorted).
  frames=("$imgs_dir"/*)
  if [ ${#frames[@]} -eq 0 ] || [ ! -e "${frames[0]}" ]; then
    echo "error: no frames in $imgs_dir." >&2
    exit 1
  fi
  first_frame=$(basename "${frames[0]}")
  mica_work="$stage_dir/_mica"
  rm -rf "$mica_work"
  mkdir -p "$mica_work/input"
  cp "$imgs_dir/$first_frame" "$mica_work/input/"

  echo "[mica] running MICA on $first_frame to generate identity.npy ..."
  (
    cd "$abs_mica_dir"
    python demo.py \
        -i "$mica_work/input" \
        -o "$mica_work/output" \
        -a "$mica_work/arcface" \
        -m data/pretrained/mica.tar
  )

  frame_stem="${first_frame%.*}"
  identity_src="$mica_work/output/$frame_stem/identity.npy"
  if [ ! -f "$identity_src" ]; then
    echo "error: MICA did not produce $identity_src (face not detected?)." >&2
    echo "Try a different first frame, or place identity.npy manually at:" >&2
    echo "    $stage_dir/identity.npy" >&2
    exit 1
  fi
  cp "$identity_src" "$stage_dir/identity.npy"
  rm -rf "$mica_work"
  echo "[mica] wrote $stage_dir/identity.npy"
fi

cfg_file="$abs_tracker_dir/configs/actors/${IDNAME}.yml"
mkdir -p "$(dirname "$cfg_file")"
out_parent=$(dirname "$out_dir")
cat >"$cfg_file" <<YAML
# Auto-generated by scripts/run_tracker.sh for idname=${IDNAME}.
# Re-run the script to regenerate; hand-edit only if you know what you want.
actor: '$stage_dir'
save_folder: '$out_parent/'
optimize_shape: true
optimize_jaw: true
begin_frames: 1
keyframes: [ 0, 1 ]
# Speed flags added in MTamon/metrical-tracker@claude0420. These match
# the config.py defaults; kept explicit so the profile is visible here
# and easy to tune per run.
opt_cache_albedos: true
opt_rot_transpose: true
opt_log_every: 10
YAML

echo "[tracker] $imgs_dir -> $out_dir"
echo "[tracker] cfg: $cfg_file"
python tracker.py --cfg "$cfg_file" "$@"

# tracker writes to <out_dir>/checkpoint/; rename to checkpoint_raw/ so
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
