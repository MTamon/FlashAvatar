#!/bin/bash
# ============================================================
# run_pipeline.sh  —  Full preprocess → train → test pipeline
#
# Usage:
#   bash run_pipeline.sh --idname <ID> --video <VIDEO_PATH>
#
# Options:
#   --idname  <ID>         : Subject / session ID  (e.g. Mikawa7)
#   --video   <VIDEO_PATH> : Input video file path (e.g. dataset/src/mikawa7.mp4)
#
# Example:
#   bash run_pipeline.sh --idname Mikawa7 --video dataset/src/mikawa7.mp4
# ============================================================

# ---- Defaults ----
IDNAME=""
VIDEO_PATH=""

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --idname)
            IDNAME="$2"
            shift 2
            ;;
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Usage: bash $0 --idname <ID> --video <VIDEO_PATH>"
            exit 1
            ;;
    esac
done

# ---- Validation ----
if [ -z "${IDNAME}" ] || [ -z "${VIDEO_PATH}" ]; then
    echo "Error: --idname and --video are both required."
    echo "Usage: bash $0 --idname <ID> --video <VIDEO_PATH>"
    exit 1
fi

# ---- Pipeline ----
python scripts/preprocess.py prepare --idname "${IDNAME}" --video "${VIDEO_PATH}"

bash scripts/run_tracker.sh "${IDNAME}" \
    --smirk --bbox-mode offline --bbox-fps 30 \
    --verify-dir dataset/"${IDNAME}"/smirk_verify

python scripts/preprocess.py filter-blur --idname "${IDNAME}" --percentile 15

python scripts/preprocess.py finalize --idname "${IDNAME}"

python train.py --idname "${IDNAME}"

python scripts/preprocess.py smirk --idname "${IDNAME}" \
    --bbox-mode offline --bbox-fps 30 \
    --demo-video dataset/"${IDNAME}"/smirk_demo/demo_lbs.mp4 --demo-fps 30 \
    --demo-lbs-pose

python test.py --idname "${IDNAME}" \
    --checkpoint dataset/"${IDNAME}"/log/ckpt/chkpnt150000.pth