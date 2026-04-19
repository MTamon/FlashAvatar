#!/usr/bin/env bash
# Set up a conda environment for metrical-tracker (the FLAME tracker used
# by FlashAvatar). This follows the upstream instructions at
# https://github.com/Zielon/metrical-tracker and keeps the tracker env
# fully separate from FlashAvatar's env.
#
# Usage:
#   bash scripts/setup_metrical_tracker.sh
#
# Overridable environment variables:
#   TRACKER_DIR   target dir for the clone (default: external/metrical-tracker)
#   ENV_NAME      conda env name          (default: tracker)
#   PY_VERSION    python version          (default: 3.9)
#   CUDA          cudatoolkit spec         (default: 11.3)
#
# Notes on license-gated assets (FLAME 2020 etc.):
# metrical-tracker / MICA need assets from flame.is.tue.mpg.de that require
# registration. The upstream install.sh will prompt / document how to supply
# them. Expect to re-run that step manually after registering.

set -euo pipefail

TRACKER_DIR=${TRACKER_DIR:-external/metrical-tracker}
ENV_NAME=${ENV_NAME:-tracker}
PY_VERSION=${PY_VERSION:-3.9}
CUDA=${CUDA:-11.3}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

# ---------- 1. conda availability ----------
if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda is not on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------- 2. clone ----------
abs_tracker_dir="$repo_root/$TRACKER_DIR"
if [ ! -d "$abs_tracker_dir" ]; then
  echo "[1/4] Cloning metrical-tracker into $TRACKER_DIR ..."
  mkdir -p "$(dirname "$abs_tracker_dir")"
  git clone --recurse-submodules \
      https://github.com/Zielon/metrical-tracker.git \
      "$abs_tracker_dir"
else
  echo "[1/4] $TRACKER_DIR already exists, skipping clone."
  (cd "$abs_tracker_dir" && git submodule update --init --recursive || true)
fi

# ---------- 3. conda env ----------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[2/4] Creating conda env '$ENV_NAME' (python $PY_VERSION) ..."
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
else
  echo "[2/4] Conda env '$ENV_NAME' already exists, reusing."
fi

conda activate "$ENV_NAME"

# ---------- 4. dependencies ----------
echo "[3/4] Installing PyTorch (+ CUDA $CUDA) and tracker requirements ..."
cd "$abs_tracker_dir"

# Upstream pins PyTorch 1.12.1; install via conda to match CUDA cleanly.
conda install -y \
    pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
    "cudatoolkit=$CUDA" \
    -c pytorch -c nvidia

# Prefer the upstream install.sh if present (handles MICA assets too).
if [ -x install.sh ]; then
  echo "[3b] Running upstream install.sh ..."
  bash install.sh || {
    echo
    echo "upstream install.sh exited non-zero. That step often fails when"
    echo "FLAME / MICA assets have not been downloaded yet. Follow the"
    echo "instructions printed above (register at flame.is.tue.mpg.de, place"
    echo "the assets, rerun install.sh) and re-run this script if needed."
    exit_code=$?
  }
elif [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

# ---------- 5. done ----------
cat <<EOM

[4/4] metrical-tracker environment ready.

Next steps:

  # Activate the tracker env and run it on a FlashAvatar identity's frames.
  conda activate $ENV_NAME
  cd $TRACKER_DIR
  python tracker.py \\
      --input_dir $repo_root/dataset/<idname>/raw/imgs \\
      --output_dir $repo_root/metrical-tracker/output/<idname>

  # Then back in the FlashAvatar env:
  cd $repo_root
  python scripts/preprocess.py finalize --idname <idname>

Or use the convenience wrapper:

  bash scripts/run_tracker.sh <idname>

EOM
