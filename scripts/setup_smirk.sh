#!/usr/bin/env bash
# Optional installer: clone MTamon/smirk@release/cuda128 under
# external/smirk and install its deps + weights into the ACTIVE
# FlashAvatar env. Provides a more robust FLAME feature extractor for
# high-motion / motion-blurred videos, as an alternative to
# metrical-tracker.
#
# This is intentionally SEPARATE from install_128.sh — SMIRK is opt-in.
# Run this AFTER `bash install_128.sh` has set up FlashAvatar's env,
# and with that env active.
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/setup_smirk.sh
#
# Overridable environment variables:
#   SMIRK_REPO    git url    (default: https://github.com/MTamon/smirk.git)
#   SMIRK_BRANCH  git branch (default: release/cuda128)
#   SMIRK_DIR     clone dir  (default: external/smirk)
#   SKIP_DEPS     if set, skip running SMIRK's install_128.sh
#   SKIP_WEIGHTS  if set, skip running SMIRK's quick_install.sh
#   FLAME_USER    FLAME account (prompted by SMIRK's quick_install.sh)
#   FLAME_PASS    FLAME password

set -euo pipefail

SMIRK_REPO=${SMIRK_REPO:-https://github.com/MTamon/smirk.git}
SMIRK_BRANCH=${SMIRK_BRANCH:-release/cuda128}
SMIRK_DIR=${SMIRK_DIR:-external/smirk}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
abs_smirk_dir="$repo_root/$SMIRK_DIR"

# ---------- 1. precondition check ----------
if ! command -v python >/dev/null 2>&1; then
  echo "error: no 'python' on PATH. Activate FlashAvatar's env first:" >&2
  echo "    source .venv/bin/activate        (venv)" >&2
  echo "    conda activate <envname>         (conda)" >&2
  exit 1
fi
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "error: 'torch' is not importable in the active python env." >&2
  echo "Run FlashAvatar's install first:" >&2
  echo "    bash install_128.sh" >&2
  exit 1
fi
torch_ver=$(python -c "import torch; print(torch.__version__)")
case "$torch_ver" in
  2.9.*) ;;
  *)
    echo "warning: SMIRK (release/cuda128) is pinned against torch==2.9.1;" >&2
    echo "         active env has torch $torch_ver. Dep install may rewrite" >&2
    echo "         torch in-place." >&2
    ;;
esac

# ---------- 2. clone / update ----------
if [ ! -d "$abs_smirk_dir" ]; then
  echo "[1/3] Cloning $SMIRK_REPO ($SMIRK_BRANCH) into $SMIRK_DIR ..."
  mkdir -p "$(dirname "$abs_smirk_dir")"
  git clone --branch "$SMIRK_BRANCH" --recurse-submodules \
      "$SMIRK_REPO" "$abs_smirk_dir"
else
  echo "[1/3] $SMIRK_DIR exists; updating to $SMIRK_BRANCH ..."
  (
    cd "$abs_smirk_dir"
    current_url=$(git remote get-url origin 2>/dev/null || echo "")
    if [ "$current_url" != "$SMIRK_REPO" ]; then
      echo "    rewriting origin: $current_url -> $SMIRK_REPO"
      git remote set-url origin "$SMIRK_REPO"
    fi
    git fetch origin "$SMIRK_BRANCH"
    git checkout "$SMIRK_BRANCH"
    git pull --ff-only origin "$SMIRK_BRANCH" || true
    git submodule update --init --recursive || true
  )
fi

# ---------- 3. SMIRK deps ----------
if [ -n "${SKIP_DEPS:-}" ]; then
  echo "[2/3] SKIP_DEPS set; skipping SMIRK dep install."
else
  if [ ! -f "$abs_smirk_dir/install_128.sh" ]; then
    echo "error: $abs_smirk_dir/install_128.sh missing — wrong branch?" >&2
    exit 1
  fi
  echo "[2/3] Running SMIRK's install_128.sh (pinned torch 2.9.1 / cu128)..."
  (
    cd "$abs_smirk_dir"
    # SMIRK's install_128.sh is idempotent — already-installed packages at
    # the right version are skipped. It does NOT rebuild pytorch3d (which
    # FlashAvatar's install_128.sh already built); it only adds the SMIRK
    # extras (timm, albumentations, mediapipe, ...).
    bash install_128.sh
  )
fi

# ---------- 4. SMIRK weights (FLAME, SMIRK checkpoint, MediaPipe .task) ----------
if [ -n "${SKIP_WEIGHTS:-}" ]; then
  echo "[3/3] SKIP_WEIGHTS set; skipping SMIRK weight download."
else
  smirk_ckpt="$abs_smirk_dir/pretrained_models/SMIRK_em1.pt"
  flame_pkl="$abs_smirk_dir/assets/FLAME2020/generic_model.pkl"
  if [ -f "$smirk_ckpt" ] && [ -f "$flame_pkl" ]; then
    echo "[3/3] SMIRK + FLAME assets already present, skipping quick_install.sh."
  else
    if [ ! -f "$abs_smirk_dir/quick_install.sh" ]; then
      echo "error: $abs_smirk_dir/quick_install.sh missing — wrong branch?" >&2
      exit 1
    fi
    echo "[3/3] Running SMIRK's quick_install.sh (FLAME + SMIRK ckpt + MP .task)..."
    (
      cd "$abs_smirk_dir"
      # quick_install.sh prompts for FLAME creds if FLAME_USER / FLAME_PASS
      # are not already set; we just forward the env.
      bash quick_install.sh
    )
  fi
fi

cat <<EOM

================================================================================
[done] SMIRK (release/cuda128) installed at:
    $abs_smirk_dir

Python env : $(python -c "import sys; print(sys.prefix)")

Next:
    # Generate FLAME .frame files with SMIRK (instead of metrical-tracker):
    bash scripts/run_tracker.sh <idname> --smirk
    # or, equivalently:
    python scripts/preprocess.py smirk --idname <idname>

Then, as usual:
    python scripts/preprocess.py finalize --idname <idname>

================================================================================
EOM
