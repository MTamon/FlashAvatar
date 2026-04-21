#!/usr/bin/env bash
# Run SMIRK (MTamon/smirk@release/cuda128) as a FLAME feature extractor
# over `dataset/<idname>/raw/imgs/`, producing FlashAvatar-compatible
# `.frame` files at `metrical-tracker/output/<idname>/checkpoint_raw/`.
#
# Drop-in alternative to `scripts/run_tracker.sh <idname>` (which calls
# metrical-tracker). Use this path when metrical-tracker fails on
# high-motion / motion-blurred inputs.
#
# Requires a prior `bash scripts/setup_smirk.sh`.
#
# Usage (activate the FlashAvatar env first):
#   source .venv/bin/activate
#   bash scripts/run_smirk_tracker.sh <idname> [extra preprocess smirk args...]
#
# Common extras:
#   --verify-dir dataset/<idname>/smirk_verify
#   --batch-size 16
#   --focal-px 8000
#   --shape-frames 300

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <idname> [extra args forwarded to 'preprocess smirk']" >&2
  exit 1
fi

IDNAME=$1
shift || true

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

# Sanity-check the env has the SMIRK deps the runner imports.
missing=()
for mod in torch numpy skimage PIL mediapipe pytorch3d; do
  if ! python -c "import $mod" >/dev/null 2>&1; then
    missing+=("$mod")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "error: the active python env is missing: ${missing[*]}" >&2
  echo "Repair with:" >&2
  echo "    bash install_128.sh" >&2
  echo "    bash scripts/setup_smirk.sh" >&2
  exit 1
fi

cd "$repo_root"
python scripts/preprocess.py smirk --idname "$IDNAME" "$@"
