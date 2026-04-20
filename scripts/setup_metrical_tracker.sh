#!/usr/bin/env bash
# Install MTamon/metrical-tracker (cuda128) and MTamon/MICA into the
# ACTIVE FlashAvatar env.
#
# Both forks share FlashAvatar's pinned stack (Python 3.11 / PyTorch
# 2.9.1 / CUDA 12.8 / numpy 2.2.6), so tracker AND MICA run in the same
# environment — no separate conda/venv needed. Run this AFTER
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
#   TRACKER_DIR     clone dir  (default: external/metrical-tracker)
#   MICA_REPO       git url    (default: https://github.com/MTamon/MICA.git)
#   MICA_BRANCH     git branch (default: claude/cuda128-pytorch29-update-ZxnsN)
#   MICA_DIR        clone dir  (default: external/MICA)
#   SKIP_ASSETS     if set, skip FLAME / MICA / insightface asset downloads
#   FLAME_USER      FLAME account username (prompted if unset and assets missing)
#   FLAME_PASS      FLAME account password (prompted if unset and assets missing)
#
# Notes on license-gated assets: the FLAME 2020 / texture / masks archives
# are gated behind a registration at https://flame.is.tue.mpg.de/. Re-runs
# skip the download if data/FLAME2020/generic_model.pkl already exists.
# MICA re-uses the tracker's FLAME2020 via a symlink, so FLAME is fetched
# exactly once.

set -euo pipefail

TRACKER_REPO=${TRACKER_REPO:-https://github.com/MTamon/metrical-tracker.git}
TRACKER_BRANCH=${TRACKER_BRANCH:-cuda128}
TRACKER_DIR=${TRACKER_DIR:-external/metrical-tracker}
MICA_REPO=${MICA_REPO:-https://github.com/MTamon/MICA.git}
MICA_BRANCH=${MICA_BRANCH:-claude/cuda128-pytorch29-update-ZxnsN}
MICA_DIR=${MICA_DIR:-external/MICA}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
abs_tracker_dir="$repo_root/$TRACKER_DIR"
abs_mica_dir="$repo_root/$MICA_DIR"

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
  echo "[1/5] Cloning $TRACKER_REPO ($TRACKER_BRANCH) into $TRACKER_DIR ..."
  mkdir -p "$(dirname "$abs_tracker_dir")"
  git clone --branch "$TRACKER_BRANCH" --recurse-submodules \
      "$TRACKER_REPO" "$abs_tracker_dir"
else
  echo "[1/5] $TRACKER_DIR exists; updating to $TRACKER_BRANCH ..."
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
echo "[2/5] Installing tracker deps into the active env ..."
pip install -r "$abs_tracker_dir/requirements.txt"

# Promote local datasets/ to a regular package. Upstream tracker ships
# datasets/ without __init__.py, so it's treated as a PEP 420 namespace
# package — and any `datasets` regular package already installed in the
# active env's site-packages (e.g. HuggingFace datasets, pulled in as a
# transitive dep of transformers / ML tooling) will shadow it. A zero-byte
# __init__.py makes the tracker's folder a regular package, giving it
# priority over site-packages when run from the tracker dir.
touch "$abs_tracker_dir/datasets/__init__.py"

# chumpy is not in requirements.txt but both the tracker and FlashAvatar
# need it. install_128.sh already installs it, but repair if missing.
if ! python -c "import chumpy" >/dev/null 2>&1; then
  echo "[2/5] Installing chumpy (mattloper git main; numpy 2.x compatible) ..."
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
  echo "[3/5] SKIP_ASSETS set; skipping FLAME asset download."
elif [ -f "$asset_sentinel" ]; then
  echo "[3/5] FLAME assets already present at $abs_tracker_dir/data/FLAME2020/, skipping."
else
  echo "[3/5] Downloading FLAME assets (requires https://flame.is.tue.mpg.de/ account) ..."
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

  # FLAME2020.zip / TextureSpace.zip / FLAME_masks.zip are all packaged
  # with a top-level `FLAME2020/` directory, so unzipping directly into
  # data/FLAME2020/ yields data/FLAME2020/FLAME2020/... . Flatten after
  # each extraction so generic_model.pkl ends up at the expected path.
  flatten_flame() {
    local target="$1"
    if [ -d "$target/FLAME2020" ]; then
      # shellcheck disable=SC2012
      (shopt -s dotglob nullglob; mv "$target/FLAME2020"/* "$target/" 2>/dev/null || true)
      rmdir "$target/FLAME2020" 2>/dev/null || true
    fi
  }

  (
    cd "$abs_tracker_dir"
    mkdir -p data/FLAME2020
    wget --post-data "username=$user_enc&password=$pass_enc" \
        'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
        -O FLAME2020.zip --no-check-certificate --continue
    unzip -o FLAME2020.zip -d data/FLAME2020/ && rm -f FLAME2020.zip
    flatten_flame data/FLAME2020
    [ -f data/FLAME2020/Readme.pdf ] && \
        mv data/FLAME2020/Readme.pdf data/FLAME2020/Readme_FLAME.pdf || true

    wget --post-data "username=$user_enc&password=$pass_enc" \
        'https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=TextureSpace.zip' \
        -O TextureSpace.zip --no-check-certificate --continue
    unzip -o TextureSpace.zip -d data/FLAME2020/ && rm -f TextureSpace.zip
    flatten_flame data/FLAME2020

    # FLAME_masks.zip has no top-level directory (files at root), so
    # extract directly into data/FLAME2020/FLAME_masks/ to produce the
    # data/FLAME2020/FLAME_masks/FLAME_masks.pkl path that MICA expects.
    wget 'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' \
        -O FLAME_masks.zip --no-check-certificate --continue
    mkdir -p data/FLAME2020/FLAME_masks
    unzip -o FLAME_masks.zip -d data/FLAME2020/FLAME_masks/ && rm -f FLAME_masks.zip

    # Head template mesh bundle (no auth required).
    wget -O mesh.zip 'https://keeper.mpdl.mpg.de/f/f158a430ef754edba5ec/?dl=1'
    unzip -o mesh.zip -d data/
    if [ -d data/mesh ]; then
      mv data/mesh/* data/ && rmdir data/mesh
    fi
    rm -f mesh.zip
  )
fi

# ---------- 5. MICA: clone + pip extras ----------
# MICA (Metrical Implicit Conditioned Avatars) produces the 300-dim
# FLAME shape code that tracker.py consumes as <actor>/identity.npy.
# The MTamon/MICA fork (claude/cuda128-pytorch29-update-ZxnsN) is
# pin-aligned with FlashAvatar + tracker, so it installs into the same
# env; only a handful of extras beyond tracker's requirements are needed.
if [ ! -d "$abs_mica_dir" ]; then
  echo "[4/5] Cloning $MICA_REPO ($MICA_BRANCH) into $MICA_DIR ..."
  mkdir -p "$(dirname "$abs_mica_dir")"
  git clone --branch "$MICA_BRANCH" --recurse-submodules \
      "$MICA_REPO" "$abs_mica_dir"
else
  echo "[4/5] $MICA_DIR exists; updating to $MICA_BRANCH ..."
  (
    cd "$abs_mica_dir"
    current_url=$(git remote get-url origin 2>/dev/null || echo "")
    if [ "$current_url" != "$MICA_REPO" ]; then
      echo "    rewriting origin: $current_url -> $MICA_REPO"
      git remote set-url origin "$MICA_REPO"
    fi
    git fetch origin "$MICA_BRANCH"
    git checkout "$MICA_BRANCH"
    git pull --ff-only origin "$MICA_BRANCH" || true
    git submodule update --init --recursive || true
  )
fi

# MICA-only pip extras. Installed WITHOUT --no-deps because insightface /
# onnxruntime-gpu pull their own transitive deps that don't conflict with
# the FlashAvatar pin set. Re-runs are idempotent.
echo "[4/5] Installing MICA deps into the active env ..."
pip install insightface==0.7.3 onnx==1.17.0 onnxruntime-gpu==1.22.0 gdown==5.2.0

# Share FLAME2020 between tracker and MICA. Previously this was done
# with a symlink, but the MICA fork ships its own committed files in
# data/FLAME2020/ (landmark_embedding.npy, head_template.obj, the
# FLAME_masks/ subdir, ...) — a symlink destroys those. Instead, mirror
# tracker's downloads into MICA's tree non-destructively (MICA wins on
# every filename conflict). Migration for earlier revisions that did
# symlink: restore MICA's committed directory first.
mica_flame_dir="$abs_mica_dir/data/FLAME2020"
tracker_flame_dir="$abs_tracker_dir/data/FLAME2020"

if [ -L "$mica_flame_dir" ]; then
  rm -f "$mica_flame_dir"
  (cd "$abs_mica_dir" && git checkout -- data/FLAME2020 >/dev/null 2>&1 || true)
  echo "[4/5] removed stale FLAME2020 symlink in MICA and restored committed tree"
fi

if [ -d "$tracker_flame_dir" ]; then
  mkdir -p "$mica_flame_dir"
  (
    shopt -s dotglob nullglob
    for src in "$tracker_flame_dir"/*; do
      dst="$mica_flame_dir/$(basename "$src")"
      [ -e "$dst" ] && continue
      if [ -d "$src" ] && [ ! -L "$src" ]; then
        cp -rp "$src" "$dst"
      else
        cp -p "$src" "$dst"
      fi
    done
  )
  # MICA's masking.py expects data/FLAME2020/FLAME_masks/FLAME_masks.pkl,
  # but FLAME_masks.zip has no top-level FLAME_masks/ directory, so the
  # tracker-side extraction lands it flat at data/FLAME2020/FLAME_masks.pkl.
  # Bridge the gap so MICA finds it at the nested path.
  if [ -f "$tracker_flame_dir/FLAME_masks.pkl" ] && \
     [ ! -f "$mica_flame_dir/FLAME_masks/FLAME_masks.pkl" ]; then
    mkdir -p "$mica_flame_dir/FLAME_masks"
    cp -p "$tracker_flame_dir/FLAME_masks.pkl" \
          "$mica_flame_dir/FLAME_masks/FLAME_masks.pkl"
  fi
fi

# ---------- 6. MICA model assets ----------
# mica.tar (Google Drive) + insightface antelopev2/buffalo_l (Google Drive).
# insightface looks up models under ~/.insightface/models/<name>/.
if [ -n "${SKIP_ASSETS:-}" ]; then
  echo "[5/5] SKIP_ASSETS set; skipping MICA asset download."
else
  (
    cd "$abs_mica_dir"
    if [ ! -f data/pretrained/mica.tar ]; then
      echo "[5/5] Downloading MICA checkpoint (mica.tar) ..."
      mkdir -p data/pretrained
      gdown --id 1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt -O data/pretrained/mica.tar
    else
      echo "[5/5] MICA checkpoint already present, skipping."
    fi
  )

  insight_dir="$HOME/.insightface/models"
  mkdir -p "$insight_dir"
  if [ ! -d "$insight_dir/antelopev2" ]; then
    echo "[5/5] Downloading insightface antelopev2 ..."
    gdown --id 16PWKI_RjjbE4_kqpElG-YFqe8FpXjads -O "$insight_dir/antelopev2.zip"
    unzip -o "$insight_dir/antelopev2.zip" -d "$insight_dir/"
    rm -f "$insight_dir/antelopev2.zip"
  else
    echo "[5/5] insightface antelopev2 already present, skipping."
  fi
  if [ ! -d "$insight_dir/buffalo_l" ]; then
    echo "[5/5] Downloading insightface buffalo_l ..."
    gdown --id 1navJMy0DTr1_DHjLWu1i48owCPvXWfYc -O "$insight_dir/buffalo_l.zip"
    unzip -o "$insight_dir/buffalo_l.zip" -d "$insight_dir/"
    rm -f "$insight_dir/buffalo_l.zip"
  else
    echo "[5/5] insightface buffalo_l already present, skipping."
  fi
fi

cat <<EOM

================================================================================
[done] metrical-tracker (cuda128) + MICA set up in the active FlashAvatar env.

Tracker    : $abs_tracker_dir
MICA       : $abs_mica_dir
Python env : $(python -c "import sys; print(sys.prefix)")

Next:
    bash scripts/run_tracker.sh <idname>

(run_tracker.sh automatically invokes MICA on the first frame if
 <actor>/identity.npy is missing, then runs tracker.py.)

Then:
    python scripts/preprocess.py finalize --idname <idname>

================================================================================
EOM
