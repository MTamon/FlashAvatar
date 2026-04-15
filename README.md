# FlashAvatar
**[Paper](https://arxiv.org/abs/2312.02214)|[Project Page](https://ustc3dv.github.io/FlashAvatar/)**

![teaser](exhibition/teaser.png)
Given a monocular video sequence, our proposed FlashAvatar can reconstruct a high-fidelity digital avatar in minutes which can be animated and rendered over 300FPS at the resolution of 512×512 with an Nvidia RTX 3090.

## Setup

### Supported environment (128 branch)

This branch is maintained for modern PyTorch / CUDA stacks.

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 (other Linux distros likely work) |
| Python | 3.11 |
| PyTorch | 2.9.1 (torchvision 0.24.1) |
| CUDA Toolkit | 12.8 (system-installed, `nvcc` on PATH) |
| GCC | gcc-11 / g++-11 (required by the CUDA extensions) |
| GPU | RTX 30 / 40 / Hopper class (sm_80 / sm_86 / sm_89 / sm_90) |

Support for PyTorch 2.12 + CUDA 13.0 / 13.2 is planned as a follow-up once
PyTorch 2.12 is released.

### Install (pip-only)

0. Prerequisites on the host:

   ```bash
   sudo apt install -y gcc-11 g++-11
   # Install CUDA Toolkit 12.8 system-wide so that nvcc is available,
   # then export CUDA_HOME if it is not at /usr/local/cuda-12.8 already.
   export CUDA_HOME=/usr/local/cuda-12.8
   export PATH=$CUDA_HOME/bin:$PATH
   ```

1. Create a Python 3.11 virtual environment and activate it:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Initialize git submodules (diff-gaussian-rasterization / simple-knn):

   ```bash
   git submodule update --init --recursive
   ```

3. Run the install script. It mirrors MTamon/DECA128/install_128.sh, installs
   the pinned dependency set from `requirements_128.txt`, builds pytorch3d
   v0.7.8 from source, and builds the two local CUDA extensions under
   `submodules/`:

   ```bash
   bash install_128.sh
   ```

The script sanity-checks the environment at the end (torch, pytorch3d,
diff_gaussian_rasterization, simple_knn).

#### Troubleshooting the CUDA extension builds

If `pip install ./submodules/diff-gaussian-rasterization` or
`./submodules/simple-knn` fail against PyTorch 2.9.1 / CUDA 12.8, first try:

- Double-check `nvcc --version` reports 12.8 and is actually the one on PATH.
- Make sure `TORCH_CUDA_ARCH_LIST` is set to the arch list of your local GPU
  (the script defaults to `7.5;8.0;8.6;8.9;9.0`; you can narrow it to e.g.
  `8.6` for an RTX 3090 to speed up the build).
- Re-run with verbose output: `pip install -v --no-deps ./submodules/simple-knn`.

If the build still fails, save the full compiler log and report back; the
expected follow-up is to apply targeted patches (API deprecations, GLM header
fixes) in a follow-up commit on this same branch.

### Legacy conda environment (original FlashAvatar, for reference only)

The original FlashAvatar environment targeted CUDA 11.6 / PyTorch 1.12.1 /
Python 3.7.13 and is kept around for reproducibility of the upstream paper:

```
conda env create --file environment.yml
conda activate FlashAvatar
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

This path is no longer actively maintained.
## Data Convention
The data is organized in the following form：
```
dataset
├── <id1_name>
    ├── alpha # raw alpha prediction
    ├── imgs # extracted video frames
    ├── parsing # semantic segmentation
├── <id2_name>
...
metrical-tracker
├── output
    ├── <id1_name>
        ├── checkpoint
    ├── <id2_name>
...
```
## Running
- **Evaluating pre-trained model**
```shell
python test.py --idname <id_name> --checkpoint dataset/<id_name>/log/ckpt/chkpnt.pth
```
-  **Training on your own data** 
```shell
python train.py --idname <id_name>
```
Download the [example](https://drive.google.com/file/d/1_WLvlmHD73jOAO178N7eX5UQqlrL2ghD/view?usp=drive_link) with pre-processed data and pre-trained model for a try!

## Citation
```
@inproceedings{xiang2024flashavatar,
      author    = {Jun Xiang and Xuan Gao and Yudong Guo and Juyong Zhang},
      title     = {FlashAvatar: High-fidelity Head Avatar with Efficient Gaussian Embedding},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2024},
  }
```
