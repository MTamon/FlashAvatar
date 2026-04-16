# FLAME Parameter Conversion Rationale

This document explains **why** the parameter conversions between face
trackers (DECA, EMOCA, SMIRK, SPARK) and FlashAvatar are mathematically
valid.  It is intended for collaborators and reviewers who want to
understand the theoretical basis before trusting the implementation.

## Background

FlashAvatar's deformation model (`src/deform_model.py`) conditions a
per-vertex MLP on a 120-dimensional FLAME parameter vector:

```
condition = expr(100) ‖ jaw_pose(6) ‖ eyes_pose(12) ‖ eyelids(2)   [120D]
```

The trackers listed below all fit the **same** FLAME 2020 model
(`generic_model.pkl`) but produce outputs in different shapes and
rotation representations.

| Tracker | Expression | Jaw Pose | Eyes | Eyelids | Camera |
|---------|-----------|----------|------|---------|--------|
| **DECA / EMOCA** | 50D | axis-angle 3D | not estimated | not estimated | weak perspective (scale, tx, ty) |
| **SMIRK** | 50D | axis-angle 3D | not estimated | not estimated | weak perspective |
| **SPARK** | variable | axis-angle 3D | variable | variable | weak perspective |
| **FlashAvatar** | 100D | 6D rotation | 12D (6D × 2) | 2D | full perspective (K, R, t) |

---

## 1. Expression Zero-Padding (50D → 100D)

### Claim

Appending 50 zeros to a 50-dimensional expression vector produces the
**exact same mesh deformation** as passing only the first 50 coefficients
to a 100-basis FLAME model.

### Proof

FLAME stores expression blendshapes as a matrix
`B_exp ∈ R^{V×3×100}` derived from PCA on an expression displacement
dataset.  The columns are the principal components, ordered by
eigenvalue (variance explained):

```
λ₁ ≥ λ₂ ≥ … ≥ λ₅₀ ≥ λ₅₁ ≥ … ≥ λ₁₀₀
```

The expression deformation is a linear combination:

```
Δv = Σ_{i=1}^{100} ψ_i · B_i
```

where `ψ_i` is the i-th coefficient and `B_i` is the i-th blendshape.

**PCA guarantees orthogonality**: `B_i · B_j = 0` for `i ≠ j`.
Therefore each term contributes independently:

```
Δv = Σ_{i=1}^{50} ψ_i · B_i  +  Σ_{i=51}^{100} 0 · B_i
   = Σ_{i=1}^{50} ψ_i · B_i
```

Setting `ψ_{51..100} = 0` has **zero effect** on the first 50 terms.
The mesh deformation is identical to using only the 50 estimated
coefficients.

### Why trackers only estimate 50

Components 51–100 have very small eigenvalues.  Estimating them from a
single RGB image is numerically ill-conditioned (the signal is below
the noise floor of monocular reconstruction).  Truncation at 50 is
standard practice across DECA, EMOCA, SMIRK, and MICA.

---

## 2. Jaw Pose: Axis-Angle (3D) → 6D Rotation

### Claim

The conversion `axis-angle → rotation matrix → first-two-columns` is a
**lossless, bijective** mapping (for rotation angles < π).

### Proof

**Step 1 — Rodrigues' formula** converts an axis-angle vector
`ω ∈ R³` to a rotation matrix `R ∈ SO(3)`:

```
θ = ‖ω‖
k = ω / θ
R = I + sin(θ) [k]× + (1 − cos(θ)) [k]×²
```

This is a smooth bijection for `θ ∈ [0, π)`.

**Step 2 — 6D extraction**: Take columns 0 and 1 of `R`:

```
rot6d = [R₀₀, R₁₀, R₂₀, R₀₁, R₁₁, R₂₁]
```

Since `R` is orthogonal, column 2 is uniquely determined by the cross
product of columns 0 and 1:

```
R[:,2] = R[:,0] × R[:,1]
```

Therefore the 6D representation carries **exactly the same information**
as the full 3×3 matrix.

### Why FlashAvatar uses 6D

Zhou et al. ("On the Continuity of Rotation Representations in Neural
Networks", CVPR 2019) proved that 6D is the minimum-dimensional
**continuous** representation of SO(3).  Axis-angle, Euler angles, and
quaternions all have discontinuities that impair gradient-based
optimization.  Since the deformation MLP is trained with gradient
descent, 6D is the natural choice.

### Implementation

```python
from pytorch3d.transforms import axis_angle_to_matrix

def axis_angle_to_rot6d(aa: torch.Tensor) -> torch.Tensor:
    """(B, 3) axis-angle → (B, 6) continuous rotation."""
    R = axis_angle_to_matrix(aa)       # (B, 3, 3)
    return R[:, :, :2].reshape(-1, 6)  # first two columns
```

---

## 3. Eyes Pose → Identity Rotation (12D)

### Claim

Substituting the identity rotation for missing eye-pose parameters
does **not** degrade the trained model, provided the same substitution
is used during both training and inference.

### Reasoning

The deformation MLP receives:

```
condition = [expr | jaw | eyes | eyelids]
```

If `eyes` is always `[1,0,0,0,1,0, 1,0,0,0,1,0]` (identity × 2)
during training, the MLP's learned weights for the eyes-related input
dimensions will have near-zero gradient.  Effectively, those 12
dimensions become a constant bias that the MLP factors out.

Any eye-related facial deformation (e.g. squinting) is then captured
by the expression PCA coefficients, which do encode eye-region shape
changes — the PCA was computed over the full face mesh including the
eye region.

**Requirement**: the same constant must be used at inference.  If a
future tracker provides real eye-pose estimates, the model must be
**retrained** with those estimates to benefit from them.

### Optional enhancement

MediaPipe Face Mesh provides real-time iris tracking from which eye
rotation can be estimated.  This is noted in the FLARE design spec
as a Phase 2 enhancement.

---

## 4. Eyelids → Zero (2D)

### Claim

Setting eyelid blend weights to zero is safe under the same
train-time = inference-time consistency requirement.

### Reasoning

Identical to Section 3.  The 2D eyelid input becomes a learned
constant.  Blink and eyelid closure are partially captured by expression
PCA modes (particularly modes related to the AU45 action unit).

---

## 5. Camera Parameters

### Why camera conversion is separate

FlashAvatar uses **full perspective** camera parameters (K, R, t) for
Gaussian splatting rendering.  DECA/EMOCA/SMIRK use **weak perspective**
(scale, tx, ty).

The deformation MLP does **not** receive camera parameters — they are
used only in the rendering pipeline.  This means:

1. Camera conversion errors do not affect the learned deformation model.
2. In a real-time pipeline (FLARE), camera intrinsics come from the
   actual camera calibration, not from the tracker's estimate.
3. Head pose (R, t) can come from the tracker or from a separate
   PnP solver.

The conversion script provides a `weak_perspective_to_full()` utility
for offline `.frame` file generation, but in the FLARE real-time
pipeline this is expected to be replaced by the actual camera system.

---

## 6. Shape Parameters

### Handling

Shape parameters are loaded **once** from the first frame and shared
across all frames.  DECA outputs 100D shape; FlashAvatar expects 300D.

The same zero-padding logic from Section 1 applies: FLAME shape
blendshapes use the same PCA decomposition, and components 101–300
have progressively smaller eigenvalues.

### Important constraint

The shape estimate must be **consistent** between training and
inference.  If training uses DECA's shape estimate for person A, then
inference must also use DECA's shape estimate (not metrical-tracker's).
Mixing trackers for shape estimation will cause base-geometry mismatch.

---

## Summary

| Conversion | Type | Mathematical basis | Learned component |
|------------|------|-------------------|-------------------|
| Expression 50→100 | Zero-padding | PCA orthogonality | None |
| Shape 100→300 | Zero-padding | PCA orthogonality | None |
| Jaw axis-angle→6D | Rodrigues + column extraction | SO(3) representation theory | None |
| Eyes → identity | Constant substitution | MLP absorbs constant at train time | Implicit |
| Eyelids → 0 | Constant substitution | MLP absorbs constant at train time | Implicit |
| Camera weak→full | Geometric projection | Pinhole camera model | None |

All FLAME parameter conversions are **deterministic, stateless, and
zero-cost** — suitable for frame-by-frame real-time processing.

## References

- Li et al., "Learning a model of facial shape and expression from 4D
  scans", SIGGRAPH Asia 2017 (FLAME model)
- Feng et al., "Learning an Animatable Detailed 3D Face Model from
  In-The-Wild Images", SIGGRAPH 2021 (DECA)
- Zhou et al., "On the Continuity of Rotation Representations in
  Neural Networks", CVPR 2019 (6D rotation)
- Xiang et al., "FlashAvatar: High-fidelity Head Avatar with
  Efficient Gaussian Embedding", CVPR 2024
