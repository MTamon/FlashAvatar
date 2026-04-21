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

# ---------- 5. post-install sanity check ----------
# SMIRK's install_128.sh pulls a handful of packages with dependency
# resolution enabled, so in principle it could reinstall something
# FlashAvatar depends on (torch, pytorch3d, the two CUDA extensions, ...).
# Catch that here loudly with a one-shot import check — much nicer than
# finding out mid-training.
echo "[post-install] verifying FlashAvatar + SMIRK imports still resolve ..."
export SMIRK_ROOT="$abs_smirk_dir"
check_out=$(python - <<'PY'
import importlib
import sys

probes = [
    ("torch",                       "2.9."),
    ("pytorch3d",                   None),
    ("diff_gaussian_rasterization", None),
    ("simple_knn",                  None),
    ("numpy",                       "2."),
    ("mediapipe",                   None),
]

rc = 0
for mod, want in probes:
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print(f"  FAIL  {mod}: {e}")
        rc = 1
        continue
    ver = getattr(m, "__version__", "?")
    note = ""
    if want and not str(ver).startswith(want):
        note = f"  (expected {want}*, got {ver})"
        rc = 1
    print(f"  OK    {mod} {ver}{note}")

# SMIRK-specific probe: load the encoder module via the cloned repo path.
import os
smirk_root = os.environ["SMIRK_ROOT"]
sys.path.insert(0, smirk_root)
try:
    from src.smirk_encoder import SmirkEncoder  # noqa: F401
    print(f"  OK    src.smirk_encoder (via {smirk_root})")
except Exception as e:
    print(f"  FAIL  src.smirk_encoder: {e}")
    rc = 1

sys.exit(rc)
PY
) && sanity_rc=0 || sanity_rc=$?
echo "$check_out"
if [ "$sanity_rc" -ne 0 ]; then
  echo ""
  echo "error: post-install sanity check failed. SMIRK's installer may have" >&2
  echo "       reinstalled something FlashAvatar needs at a different pin." >&2
  echo "       Re-run: bash install_128.sh   (then re-run this script)" >&2
  exit 1
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
