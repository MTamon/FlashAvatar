#!/usr/bin/env bash
# Install MTamon/metrical-tracker (cuda128 branch) into the ACTIVE
# FlashAvatar env.
#
# The cuda128 fork shares FlashAvatar's pinned stack (Python 3.11 /
# PyTorch 2.9.1 / CUDA 12.8 / numpy 2.2.6), so the tracker runs in the
# same environment — no separate conda/venv needed. Run this AFTER
# `bash install_128.sh` has set up FlashAvatar's env, and with that env
# active.
#
# Usage (activate your FlashAvatar env first — venv OR conda):
#   source .venv/bin/activate          # if you used a venv
#   conda activate <envname>           # or if you use conda
#   bash scripts/setup_metrical_tracker.sh
#
# Overridable environment variables:
#   TRACKER_REPO    git url    (default: https://github.com/MTamon/metrical-tracker.git)
#   TRACKER_BRANCH  git branch (default: cuda128)
#   TRACKER_DIR    clone dir  (default: external/metrical-tracker)
#   SKIP_ASSETS    if set, skip FLAME asset download
#   FLAME_USER     FLAME account username (prompted if unset and assets missing)
#   FLAME_PASS     FLAME account password (prompted if unset and assets missing)
#
# Notes on license-gated assets: the FLAME 2020 / texture / masks archives
# are gated behind a registration at https://flame.is.tue.mpg.de/. Re-runs
# skip the download if data/FLAME2020/generic_model.pkl already exists.

set -euo pipefail

TRACKER_REPO=${TRACKER_REPO:-https://github.com/MTamon/metrical-tracker.git}
TRACKER_BRANCH=${TRACKER_BRANCH:-cuda128}
TRACKER_DIR=${TRACKER_DIR:-external/metrical-tracker}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
abs_tracker_dir="$repo_root/$TRACKER_DIR"

# ---------- 1. precondition check: FlashAvatar env must be active ----------
if ! command -v python >/dev/null 2>&1; then
  echo "error: no 'python' on PATH. Activate FlashAvatar's env first:" >&2
  echo "    source .venv/bin/activate        (venv)" >&2
  echo "    conda activate <envname>         (conda)" >&2
  exit 1
fi
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "error: 'torch' is not importable in the active python env." >&2
  echo "Run FlashAvatar's install first (venv or conda; either is fine):" >&2
  echo "    bash install_128.sh" >&2
  exit 1
fi
torch_ver=$(python -c "import torch; print(torch.__version__)")
case "$torch_ver" in
  2.9.*) ;;
  *)
    echo "warning: tracker (cuda128) is pinned to torch==2.9.1;" >&2
    echo "         active env has torch $torch_ver." >&2
    echo "         pip install -r requirements.txt may try to reinstall torch." >&2
    ;;
esac

# ---------- 2. clone / update the fork ----------
if [ ! -d "$abs_tracker_dir" ]; then
  echo "[1/3] Cloning $TRACKER_REPO ($TRACKER_BRANCH) into $TRACKER_DIR ..."
  mkdir -p "$(dirname "$abs_tracker_dir")"
  git clone --branch "$TRACKER_BRANCH" --recurse-submodules \
      "$TRACKER_REPO" "$abs_tracker_dir"
else
  echo "[1/3] $TRACKER_DIR exists; updating to $TRACKER_BRANCH ..."
  (
    cd "$abs_tracker_dir"
    # Migration path: earlier revisions of this script cloned Zielon's
    # upstream, which has no cuda128 branch. If origin still points there
    # (or anywhere other than $TRACKER_REPO), rewrite it so fetch works.
    current_url=$(git remote get-url origin 2>/dev/null || echo "")
    if [ "$current_url" != "$TRACKER_REPO" ]; then
      echo "    rewriting origin: $current_url -> $TRACKER_REPO"
      git remote set-url origin "$TRACKER_REPO"
    fi
    git fetch origin "$TRACKER_BRANCH"
    git checkout "$TRACKER_BRANCH"
    git pull --ff-only origin "$TRACKER_BRANCH" || true
    git submodule update --init --recursive || true
  )
fi

# ---------- 3. tracker-only pip extras ----------
# The cuda128 fork's requirements.txt largely overlaps with FlashAvatar's
# install_128.sh pin set (torch 2.9.1, numpy 2.2.6, nvidia-cu12-*, etc.).
# `pip install -r` is idempotent: already-installed packages at the right
# version are skipped, tracker-only extras (mediapipe, tensorboard,
# trimesh, matplotlib, PyWavelets, ...) are added.
echo "[2/3] Installing tracker deps into the active env ..."
pip install -r "$abs_tracker_dir/requirements.txt"

# chumpy is not in requirements.txt but both the tracker and FlashAvatar
# need it. install_128.sh already installs it, but repair if missing.
if ! python -c "import chumpy" >/dev/null 2>&1; then
  echo "[2/3] Installing chumpy (mattloper git main; numpy 2.x compatible) ..."
  pip install "git+https://github.com/mattloper/chumpy.git"
fi

# pytorch3d v0.7.8 is source-built by install_128.sh against torch 2.9.1.
# Verify it's present and compiled against the active torch.
if ! python -c "import pytorch3d" >/dev/null 2>&1; then
  echo "warning: pytorch3d is not importable. It is source-built by" >&2
  echo "         install_128.sh against torch 2.9.1 + CUDA 12.8." >&2
  echo "         Re-run \`bash install_128.sh\` to build it." >&2
fi

# ---------- 4. FLAME assets ----------
asset_sentinel="$abs_tracker_dir/data/FLAME2020/generic_model.pkl"
if [ -n "${SKIP_ASSETS:-}" ]; then
  echo "[3/3] SKIP_ASSETS set; skipping FLAME asset download."
elif [ -f "$asset_sentinel" ]; then
  echo "[3/3] FLAME assets already present at $abs_tracker_dir/data/FLAME2020/, skipping."
else
  echo "[3/3] Downloading FLAME assets (requires https://flame.is.tue.mpg.de/ account) ..."
  if [ -z "${FLAME_USER:-}" ]; then
    read -p "FLAME username: " FLAME_USER
  fi
  if [ -z "${FLAME_PASS:-}" ]; then
    read -rsp "FLAME password: " FLAME_PASS
    echo
  fi
  # URL-encode credentials (same urle() as the fork's install.sh).
  urle() {
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
      x="${1:i:1}"
      if [[ "${x}" == [a-zA-Z0-9.~-] ]]; then
        printf '%s' "${x}"
      else
        printf '%%%02X' "'${x}"
      fi
    done
    echo
  }
  user_enc=$(urle "$FLAME_USER")
  pass_enc=$(urle "$FLAME_PASS")

  (
    cd "$abs_tracker_dir"
    mkdir -p data/FLAME2020
    wget --post-data "username=$user_enc&password=$pass_enc" \
        'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
        -O FLAME2020.zip --no-check-certificate --continue
    unzip -o FLAME2020.zip -d data/FLAME2020/ && rm -f FLAME2020.zip
    [ -f data/FLAME2020/Readme.pdf ] && \
        mv data/FLAME2020/Readme.pdf data/FLAME2020/Readme_FLAME.pdf || true

    wget --post-data "username=$user_enc&password=$pass_enc" \
        'https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=TextureSpace.zip' \
        -O TextureSpace.zip --no-check-certificate --continue
    unzip -o TextureSpace.zip -d data/FLAME2020/ && rm -f TextureSpace.zip

    wget 'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' \
        -O FLAME_masks.zip --no-check-certificate --continue
    unzip -o FLAME_masks.zip -d data/FLAME2020/ && rm -f FLAME_masks.zip

    # Head template mesh bundle (no auth required).
    wget -O mesh.zip 'https://keeper.mpdl.mpg.de/f/f158a430ef754edba5ec/?dl=1'
    unzip -o mesh.zip -d data/
    if [ -d data/mesh ]; then
      mv data/mesh/* data/ && rmdir data/mesh
    fi
    rm -f mesh.zip
  )
fi

cat <<EOM

================================================================================
[done] metrical-tracker (cuda128) set up in the active FlashAvatar env.

Source     : $abs_tracker_dir
Python env : $(python -c "import sys; print(sys.prefix)")

Next:
    bash scripts/run_tracker.sh <idname>

or manually:
    cd $abs_tracker_dir
    python tracker.py \\
        --input_dir $repo_root/dataset/<idname>/raw/imgs \\
        --output_dir $repo_root/metrical-tracker/output/<idname>

Then:
    python scripts/preprocess.py finalize --idname <idname>

================================================================================
EOM
