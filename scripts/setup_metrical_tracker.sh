#!/usr/bin/env bash
# Set up a conda environment for metrical-tracker (the FLAME tracker used
# by FlashAvatar). This follows the upstream instructions at
# https://github.com/Zielon/metrical-tracker and keeps the tracker env
# fully separate from FlashAvatar's env.
#
# Usage:
#   bash scripts/setup_metrical_tracker.sh
#
# The script is idempotent: re-running it repairs missing dependencies
# (e.g. opencv-python) without re-cloning or recreating the conda env.
#
# Overridable environment variables:
#   TRACKER_DIR   target dir for the clone (default: external/metrical-tracker)
#   ENV_NAME      conda env name          (default: tracker)
#   PY_VERSION    python version          (default: 3.9)
#   CUDA          cudatoolkit spec         (default: 11.3)
#   SKIP_ASSETS   if set, skip running upstream install.sh (asset download)
#
# Notes on license-gated assets (FLAME 2020 etc.):
# metrical-tracker / MICA need assets from flame.is.tue.mpg.de that require
# registration. The upstream install.sh will prompt / document how to supply
# them. You can rerun just the asset step manually:
#     cd <TRACKER_DIR>; conda activate <ENV_NAME>; bash install.sh

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
conda_base=$(conda info --base)

# ---------- 2. clone ----------
abs_tracker_dir="$repo_root/$TRACKER_DIR"
if [ ! -d "$abs_tracker_dir" ]; then
  echo "[1/5] Cloning metrical-tracker into $TRACKER_DIR ..."
  mkdir -p "$(dirname "$abs_tracker_dir")"
  git clone --recurse-submodules \
      https://github.com/Zielon/metrical-tracker.git \
      "$abs_tracker_dir"
else
  echo "[1/5] $TRACKER_DIR already exists, skipping clone."
  (cd "$abs_tracker_dir" && git submodule update --init --recursive || true)
fi

# ---------- 3. conda env ----------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[2/5] Creating conda env '$ENV_NAME' (python $PY_VERSION) ..."
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
else
  echo "[2/5] Conda env '$ENV_NAME' already exists, reusing."
fi

conda activate "$ENV_NAME"
env_prefix="$CONDA_PREFIX"

# ---------- 4. pytorch + tracker pip dependencies ----------
echo "[3/5] Installing PyTorch (+ CUDA $CUDA) ..."
cd "$abs_tracker_dir"

conda install -y \
    pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
    "cudatoolkit=$CUDA" \
    -c pytorch -c nvidia

# Always install the upstream pip deps. Previously we let install.sh do
# this implicitly, but upstream's install.sh may fail partway (e.g. on
# licence-gated asset downloads) before installing cv2/mediapipe/etc.
echo "[4/5] Installing pip dependencies ..."
# Note: we deliberately do NOT run `conda env update -f environment.yml`.
# metrical-tracker's environment.yml pins its own torch / cudatoolkit /
# python versions that collide with the conda install above, and the
# resulting conflict solve can hang or fail. `requirements.txt` carries
# every pure-python dependency the tracker actually needs at runtime.
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

# Safety net: the three packages that have caused the most frequent
# "ModuleNotFoundError" reports regardless of whether requirements.txt
# covered them. Explicit versions match metrical-tracker's README as of
# 2024-11; adjust if upstream changes.
pip install --upgrade \
    opencv-python \
    mediapipe \
    face-alignment \
    pyyaml loguru trimesh \
    || {
  echo "warning: pip safety-net install failed; tracker may still lack deps."
}

# chumpy needs special handling: the PyPI sdist (0.70) has a setup.py that
# does `import pip`, which fails inside PEP 517 isolated build envs with
# "ModuleNotFoundError: No module named 'pip'". The mattloper fork fixes
# this and is also what FlashAvatar's own install_128.sh uses (see
# requirements_128.txt:75). `--no-build-isolation` as a fallback lets
# chumpy's setup.py see the env's own pip.
if ! python -c "import chumpy" >/dev/null 2>&1; then
  echo "[4/5] Installing chumpy (mattloper fork) ..."
  pip install "git+https://github.com/mattloper/chumpy.git" \
    || pip install --no-build-isolation chumpy \
    || echo "warning: chumpy install failed; tracker may still lack it."
fi

# pytorch3d: tracker.py imports `from pytorch3d.io import load_obj`. Upstream
# requirements.txt does NOT list pytorch3d; metrical-tracker's install.sh
# tries to build it from source which is slow and often fails on CUDA
# mismatch. Use Facebook's prebuilt wheel for py39 + torch 1.12.1 + cu113,
# with a source build as a last-resort fallback.
if ! python -c "import pytorch3d" >/dev/null 2>&1; then
  echo "[4/5] Installing pytorch3d (prebuilt wheel for py39/torch1.12.1/cu113) ..."
  pip install fvcore iopath
  pip install --no-index --no-cache-dir pytorch3d \
      -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html \
    || {
      echo "prebuilt wheel unavailable; falling back to source build (slow)."
      pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2" \
        || echo "warning: pytorch3d install failed; tracker.py will error on import."
    }
fi

# ---------- 5. asset download (upstream install.sh) ----------
if [ -n "${SKIP_ASSETS:-}" ]; then
  echo "[5/5] SKIP_ASSETS set; skipping upstream install.sh (assets)."
elif [ -x install.sh ] || [ -f install.sh ]; then
  echo "[5/5] Running upstream install.sh (downloads FLAME / MICA assets) ..."
  bash install.sh || {
    echo
    echo "upstream install.sh exited non-zero. This usually means FLAME /"
    echo "MICA assets require registration at https://flame.is.tue.mpg.de/"
    echo "and a manual download. The conda env itself is already usable;"
    echo "just rerun \`cd $TRACKER_DIR && conda activate $ENV_NAME && \\"
    echo "bash install.sh\` after you have placed the assets."
  }
else
  echo "[5/5] no install.sh in $TRACKER_DIR; skipping asset step."
fi

# ---------- 6. locations ----------
cat <<EOM

================================================================================
[done] metrical-tracker environment ready.

Installed locations:

  Source (cloned repo)  : $abs_tracker_dir
  Conda env (binaries)  : $env_prefix
  Site-packages         : $env_prefix/lib/python$PY_VERSION/site-packages

Next steps:

  bash scripts/run_tracker.sh <idname>

or manually:

  conda activate $ENV_NAME
  cd $abs_tracker_dir
  python tracker.py \\
      --input_dir $repo_root/dataset/<idname>/raw/imgs \\
      --output_dir $repo_root/metrical-tracker/output/<idname>

Then back in the FlashAvatar env:

  python scripts/preprocess.py finalize --idname <idname>

================================================================================
EOM
