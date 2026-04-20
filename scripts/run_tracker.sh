#!/usr/bin/env bash
# Convenience wrapper: activate the metrical-tracker conda env, run the
# tracker against `dataset/<idname>/raw/imgs/`, and stage the result at
# `metrical-tracker/output/<idname>/checkpoint_raw/` so
# `preprocess.py finalize` can consume it.
#
# Usage:
#   bash scripts/run_tracker.sh <idname> [extra tracker args...]
#
# Overridable environment variables:
#   TRACKER_DIR   tracker install dir (default: external/metrical-tracker)
#   ENV_NAME      conda env name      (default: tracker)

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <idname> [extra tracker args...]" >&2
  exit 1
fi

IDNAME=$1
shift || true

TRACKER_DIR=${TRACKER_DIR:-external/metrical-tracker}
ENV_NAME=${ENV_NAME:-tracker}

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

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not on PATH." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Sanity-check the env has the packages tracker.py imports.
missing=()
for mod in cv2 mediapipe face_alignment torch numpy pytorch3d chumpy; do
  if ! python -c "import $mod" >/dev/null 2>&1; then
    missing+=("$mod")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "error: the '$ENV_NAME' env is missing: ${missing[*]}" >&2
  echo "Repair with:" >&2
  echo "    bash scripts/setup_metrical_tracker.sh" >&2
  echo "or manually (the install recipe differs per package):" >&2
  for mod in "${missing[@]}"; do
    case "$mod" in
      chumpy)
        echo "    conda activate $ENV_NAME && pip install --no-build-isolation chumpy" >&2
        ;;
      pytorch3d)
        echo "    conda activate $ENV_NAME && pip install --no-index --no-cache-dir pytorch3d \\" >&2
        echo "        -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html" >&2
        ;;
      *)
        echo "    conda activate $ENV_NAME && pip install $mod" >&2
        ;;
    esac
  done
  exit 1
fi

mkdir -p "$out_dir"
cd "$abs_tracker_dir"

echo "[tracker] $imgs_dir -> $out_dir (env: $ENV_NAME)"
python tracker.py \
    --input_dir "$imgs_dir" \
    --output_dir "$out_dir" \
    "$@"

# The upstream tracker writes to checkpoint/; rename to checkpoint_raw/ so
# `preprocess finalize` treats it as the pre-crop source of truth.
if [ -d "$out_dir/checkpoint" ] && [ ! -d "$out_dir/checkpoint_raw" ]; then
  mv "$out_dir/checkpoint" "$out_dir/checkpoint_raw"
  echo "[tracker] moved checkpoint/ -> checkpoint_raw/"
fi

cat <<EOM

[done] tracker output at: $out_dir/checkpoint_raw/

Next: back in the FlashAvatar env, run:

  python scripts/preprocess.py finalize --idname $IDNAME

EOM
