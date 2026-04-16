#!/usr/bin/env bash
# FlashAvatar install script for Python 3.11 / PyTorch 2.9.1 / CUDA 12.8.
#
# Preconditions (verified before running this script):
#   - Python 3.11 is active (e.g. `python3.11 -m venv .venv && source .venv/bin/activate`).
#   - System CUDA Toolkit 12.8 is installed (nvcc must be on PATH, or CUDA_HOME set).
#   - gcc-11 / g++-11 are installed (Ubuntu 22.04: `sudo apt install gcc-11 g++-11`).
#   - Git submodules are initialized:
#         git submodule update --init --recursive
#
# This script mirrors MTamon/DECA128/install_128.sh as closely as possible to
# keep library versions aligned. The `--no-deps` flag is used everywhere to
# prevent pip from mutating the pin set via transitive resolution.
#
# After pinned deps, it additionally:
#   1. installs chumpy from GitHub (numpy 2.x compatible main branch),
#   2. source-builds pytorch3d v0.7.8 against torch 2.9.1 + CUDA 12.8,
#   3. builds the two local CUDA extensions under submodules/.

set -euo pipefail

# ----------------------------------------------------------------------------
# Toolchain setup for CUDA extensions.
# ----------------------------------------------------------------------------
# PyTorch 2.9 + CUDA 12.8 needs gcc <= 13; gcc-11 is the DECA128-tested choice.
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"

# Point CUDA_HOME at the system CUDA 12.8 install (Ubuntu standard path).
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/usr/local/cuda-12.8" ]; then
        export CUDA_HOME="/usr/local/cuda-12.8"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        echo "[install_128.sh] WARNING: CUDA_HOME is not set and /usr/local/cuda-12.8 was not found."
        echo "[install_128.sh]          Set CUDA_HOME manually to your CUDA 12.8 install path before rerunning."
        exit 1
    fi
fi
export PATH="${CUDA_HOME}/bin:${PATH}"

# Force nvcc to emit code for common modern arches; extend here if needed.
# (Ampere sm_80/86, Ada sm_89, Hopper sm_90. Add 12.0 later for Blackwell.)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"

# Force pytorch3d / submodule setups to build with CUDA support.
export FORCE_CUDA=1

echo "[install_128.sh] CC=${CC} CXX=${CXX}"
echo "[install_128.sh] CUDA_HOME=${CUDA_HOME}"
echo "[install_128.sh] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
nvcc --version || { echo "[install_128.sh] nvcc not found on PATH"; exit 1; }

# ----------------------------------------------------------------------------
# 1. Upgrade pip and install pinned dependencies (DECA128-aligned).
# ----------------------------------------------------------------------------
python -m pip install --upgrade pip==25.2

# chumpy (git main — numpy 2.x friendly; DECA128 uses the exact same source).
pip install git+https://github.com/mattloper/chumpy.git

# Pinned deps. Installed with --no-deps to avoid pip rewriting the pin set.
pip install --no-deps Cython==0.29.35
pip install --no-deps face-alignment==1.4.1
pip install --no-deps filelock==3.20.0
pip install --no-deps fsspec==2025.10.0
pip install --no-deps fvcore==0.1.5.post20221221
pip install --no-deps ImageIO==2.37.2
pip install --no-deps iopath==0.1.10
pip install --no-deps Jinja2==3.1.6
pip install --no-deps joblib==1.5.2
pip install --no-deps kornia==0.8.2
pip install --no-deps kornia_rs==0.1.10
pip install --no-deps lazy_loader==0.4
pip install --no-deps llvmlite==0.45.1
pip install --no-deps MarkupSafe==3.0.3
pip install --no-deps mpmath==1.3.0
pip install --no-deps networkx==3.5
pip install --no-deps ninja==1.13.0
pip install --no-deps numba==0.62.1
pip install --no-deps numpy==2.2.6
pip install --no-deps nvidia-cublas-cu12==12.8.4.1
pip install --no-deps nvidia-cuda-cupti-cu12==12.8.90
pip install --no-deps nvidia-cuda-nvrtc-cu12==12.8.93
pip install --no-deps nvidia-cuda-runtime-cu12==12.8.90
pip install --no-deps nvidia-cudnn-cu12==9.10.2.21
pip install --no-deps nvidia-cufft-cu12==11.3.3.83
pip install --no-deps nvidia-cufile-cu12==1.13.1.3
pip install --no-deps nvidia-curand-cu12==10.3.9.90
pip install --no-deps nvidia-cusolver-cu12==11.7.3.90
pip install --no-deps nvidia-cusparse-cu12==12.5.8.93
pip install --no-deps nvidia-cusparselt-cu12==0.7.1
pip install --no-deps nvidia-nccl-cu12==2.27.5
pip install --no-deps nvidia-nvjitlink-cu12==12.8.93
pip install --no-deps nvidia-nvshmem-cu12==3.3.20
pip install --no-deps nvidia-nvtx-cu12==12.8.90
pip install --no-deps opencv-python==4.12.0.88
pip install --no-deps packaging==25.0
pip install --no-deps pillow==12.0.0
pip install --no-deps portalocker==3.2.0
pip install --no-deps PyYAML==6.0.3
pip install --no-deps scikit-image==0.25.2
pip install --no-deps scikit-learn==1.7.2
pip install --no-deps scipy==1.16.3
pip install --no-deps six==1.17.0
pip install --no-deps sympy==1.14.0
pip install --no-deps tabulate==0.9.0
pip install --no-deps termcolor==3.2.0
pip install --no-deps threadpoolctl==3.6.0
pip install --no-deps tifffile==2025.10.16
pip install --no-deps torch==2.9.1
pip install --no-deps torchvision==0.24.1
pip install --no-deps tqdm==4.67.1
pip install --no-deps triton==3.5.1
pip install --no-deps typing_extensions==4.15.0
pip install --no-deps yacs==0.1.8

# FlashAvatar-only additions. These three are not in DECA128's set but are
# required by FlashAvatar's Python code (train.py / scene / flame).
pip install --no-deps lpips==0.1.4
pip install --no-deps plyfile==1.1.2
pip install --no-deps loguru==0.7.3

# ----------------------------------------------------------------------------
# 2. pytorch3d v0.7.8 (source build against torch 2.9.1 + CUDA 12.8).
# ----------------------------------------------------------------------------
# NOTE: this step compiles a large CUDA extension and can take 10+ minutes.
# pip's --filter=blob:none clone doesn't fetch tags, so we clone manually.
PYTORCH3D_TMP="$(mktemp -d)"
git clone --branch v0.7.8 --depth 1 https://github.com/facebookresearch/pytorch3d.git "${PYTORCH3D_TMP}"
pip install --no-deps "${PYTORCH3D_TMP}"
rm -rf "${PYTORCH3D_TMP}"

# ----------------------------------------------------------------------------
# 3. Local CUDA extensions (diff-gaussian-rasterization and simple-knn).
# ----------------------------------------------------------------------------
# The submodules MUST be populated before running this script:
#     git submodule update --init --recursive
#
# simple-knn needs a newer commit than the submodule pointer (the original
# camenduru commit uses torch's deprecated .data API and is missing <float.h>,
# both of which break on PyTorch 2.x / CUDA 12.x). Force-check out the known-
# good commit before building. Safe no-op if already on that commit.
SIMPLE_KNN_PIN="60f461ff977c577ab43e29c9e4f7a4480eecc87a"
if [ -d submodules/simple-knn/.git ] || [ -f submodules/simple-knn/.git ]; then
    (
        cd submodules/simple-knn
        git fetch origin "${SIMPLE_KNN_PIN}" 2>/dev/null || git fetch origin
        git checkout --detach "${SIMPLE_KNN_PIN}"
    )
else
    echo "[install_128.sh] submodules/simple-knn not initialized."
    echo "[install_128.sh] Run: git submodule update --init --recursive"
    exit 1
fi

pip install --no-deps ./submodules/diff-gaussian-rasterization
pip install --no-deps ./submodules/simple-knn

echo "[install_128.sh] done. Quick sanity check:"
python - <<'PY'
import torch
print("torch            :", torch.__version__)
print("torch.cuda       :", torch.version.cuda)
print("cuda available   :", torch.cuda.is_available())
try:
    import pytorch3d
    print("pytorch3d        :", pytorch3d.__version__)
except Exception as e:
    print("pytorch3d import :", repr(e))
try:
    import diff_gaussian_rasterization  # noqa: F401
    print("diff_gaussian_rasterization : ok")
except Exception as e:
    print("diff_gaussian_rasterization :", repr(e))
try:
    from simple_knn._C import distCUDA2  # noqa: F401
    print("simple_knn       : ok")
except Exception as e:
    print("simple_knn       :", repr(e))
PY
