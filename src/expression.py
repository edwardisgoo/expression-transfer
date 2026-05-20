"""
Phase 2: Expression Parameterization
Owner: Member A

Computes displacement vectors representing a facial expression,
normalized by face scale for robustness across different image sizes.

Changes vs original:
    - _interocular_distance: now uses mean of 6 eye landmark points per side
      instead of two single outer-corner points (36, 45) — more stable against
      per-point detection noise.
    - Added _eye_centers() and _align_landmarks() helpers.
    - Full mode: aligns driver_neutral onto driver before computing delta so
      minor head movement between shots does not pollute the expression delta.
    - Direct mode: replaced mean(lm)-based alignment (biased by jaw/chin points)
      with eye-anchor alignment via _align_landmarks().
    - Added _auto_scale(): when scale=None, normalises P95 displacement to
      target_ratio × IOD bidirectionally (both up and down).  Previously
      scale defaulted to a fixed float with no auto option.
    - Added _validate_landmarks() for early shape / NaN checks.
    - Warning printed when auto scale ≥ 1.8 (likely near pipeline ceiling).
"""

import numpy as np

# ── Landmark index ranges (dlib 68-point model) ──────────────────────────────
_LEFT_EYE  = slice(36, 42)
_RIGHT_EYE = slice(42, 48)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _eye_centers(landmarks: np.ndarray) -> tuple:
    """Return (left_eye_center, right_eye_center) averaged over 6 pts each."""
    return landmarks[_LEFT_EYE].mean(axis=0), landmarks[_RIGHT_EYE].mean(axis=0)


def _interocular_distance(landmarks: np.ndarray) -> float:
    """
    Inter-ocular distance based on mean eye-centre points.

    Original used single outer-corner points (landmarks[36], landmarks[45]).
    Averaging over all 6 eye points per side is more robust to detection noise.
    """
    lc, rc = _eye_centers(landmarks)
    return float(np.linalg.norm(rc - lc))


def _align_landmarks(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Align src landmarks onto ref via scale + translation using eye anchors.

    Original direct mode used lm.mean(axis=0) as the centre, which is biased
    toward the lower face (jaw, chin have more points).  Eye-anchor alignment
    is more stable and consistent with how align_face() works in demo.py.

    Only corrects scale + translation (NOT rotation) — rotation is already
    handled by align_face() before this function is called.

    Args:
        src: (68, 2) landmarks to align
        ref: (68, 2) reference landmarks

    Returns:
        (68, 2) src landmarks rescaled and translated to ref space
    """
    src_lc, src_rc = _eye_centers(src)
    ref_lc, ref_rc = _eye_centers(ref)

    src_iod    = np.linalg.norm(src_rc - src_lc) + 1e-8
    ref_iod    = np.linalg.norm(ref_rc - ref_lc) + 1e-8
    scale_ratio = ref_iod / src_iod

    src_center = (src_lc + src_rc) / 2.0
    ref_center = (ref_lc + ref_rc) / 2.0

    return ((src - src_center) * scale_ratio + ref_center).astype(np.float32)


def _auto_scale(
    raw_displacement: np.ndarray,
    source_iod: float,
    target_ratio: float = 0.35,
    min_scale: float = 0.3,
    max_scale: float = 2.5,
) -> float:
    """
    Normalise P95 displacement magnitude to target_ratio × IOD.

    Bidirectional:
      - P95 > target → scale DOWN (prevents extreme warping)
      - P95 < target → scale UP   (prevents expression disappearing after alignment)

    Clamped to [min_scale, max_scale] as a hard safety net.
    max_scale=2.5 because direct mode alignment compresses expression vectors
    heavily; ETR tests showed scale ~2.0 was needed for strong expressions.

    Args:
        raw_displacement: (68, 2) unscaled displacement
        source_iod:       inter-ocular distance of source face (pixels)
        target_ratio:     target P95 as fraction of IOD (default 0.35)
        min_scale:        lower clamp (default 0.3)
        max_scale:        upper clamp (default 2.5)

    Returns:
        float in [min_scale, max_scale]
    """
    magnitudes = np.linalg.norm(raw_displacement, axis=1)
    p95 = float(np.percentile(magnitudes, 95))
    if p95 < 1e-6:
        return 1.0
    computed = (source_iod * target_ratio) / p95
    return float(np.clip(computed, min_scale, max_scale))


def _validate_landmarks(lm: np.ndarray, name: str) -> None:
    if lm.shape != (68, 2):
        raise ValueError(f"{name}: expected shape (68, 2), got {lm.shape}")
    if not np.isfinite(lm).all():
        raise ValueError(f"{name}: contains NaN or Inf values")


# ── Public API ────────────────────────────────────────────────────────────────

def compute_displacement(
    source_lm: np.ndarray,
    driver_lm: np.ndarray,
    driver_neutral_lm: np.ndarray = None,
    scale: float = None,
    auto_scale_ratio: float = 0.35,
    auto_scale_min: float = 0.3,
    auto_scale_max: float = 2.5,
) -> np.ndarray:
    """
    Compute expression displacement vectors.

    Two modes
    ---------
    Full mode (driver_neutral_lm provided):
        delta = driver_expressive - driver_neutral, rescaled to source face.
        Best quality — requires a neutral photo of the driver.

    Direct mode (no driver_neutral_lm):
        Warp source landmarks toward aligned driver positions.
        No neutral photo needed; captures pose + expression together.

    Args:
        source_lm:         (68, 2) landmarks of the source (target) face
        driver_lm:         (68, 2) landmarks of the driver (expressive)
        driver_neutral_lm: (68, 2) optional — driver neutral baseline
        scale:             float to manually set expression strength, or
                           None (default) to auto-compute from displacement stats.
                           Typical range 0.5–1.5; auto usually lands in 1.0–2.0
                           for direct mode.
        auto_scale_ratio:  target P95 displacement as fraction of IOD (0.35).
        auto_scale_min:    auto-scale lower clamp (default 0.3).
        auto_scale_max:    auto-scale upper clamp (default 2.5).

    Returns:
        displacement: (68, 2) float32 array of (dx, dy) vectors
    """
    _validate_landmarks(source_lm,  "source_lm")
    _validate_landmarks(driver_lm,  "driver_lm")
    if driver_neutral_lm is not None:
        _validate_landmarks(driver_neutral_lm, "driver_neutral_lm")

    source_iod = _interocular_distance(source_lm)
    if source_iod < 1e-6:
        raise ValueError("Source landmarks degenerate — interocular distance near zero.")

    if driver_neutral_lm is not None:
        # ── Full mode ─────────────────────────────────────────────────────────
        # Align driver_neutral onto driver first so minor head movement between
        # the two shots does not contaminate the expression delta.
        # Original: raw_delta = driver_lm - driver_neutral_lm  (no alignment)
        driver_neutral_aligned = _align_landmarks(driver_neutral_lm, driver_lm)
        raw_delta = driver_lm - driver_neutral_aligned

        driver_iod = _interocular_distance(driver_lm)
        if driver_iod < 1e-6:
            raise ValueError("Driver landmarks degenerate — interocular distance near zero.")

        raw_displacement = raw_delta * (source_iod / driver_iod)

    else:
        # ── Direct mode ───────────────────────────────────────────────────────
        # align_face() in demo.py already removed rotation.
        # Use eye-anchor alignment (not mean-based) to map driver → source space.
        driver_aligned   = _align_landmarks(driver_lm, source_lm)
        raw_displacement = driver_aligned - source_lm

    # ── Scale ─────────────────────────────────────────────────────────────────
    if scale is None:
        scale = _auto_scale(
            raw_displacement, source_iod,
            target_ratio = auto_scale_ratio,
            min_scale    = auto_scale_min,
            max_scale    = auto_scale_max,
        )
        p95_px = float(np.percentile(np.linalg.norm(raw_displacement, axis=1), 95))
        print(f"[expression] P95 displacement = {p95_px:.1f} px  |  "
              f"IOD = {source_iod:.1f} px  |  auto scale = {scale:.3f}")
        if scale >= 1.8:
            print("[expression] Note: scale is high — check ETR after running; "
                  "if ETR < 0.65 the pipeline ceiling has been reached.")
    else:
        if not (0.0 < scale <= 3.0):
            print(f"[expression] Warning: scale={scale:.2f} is outside typical range (0.3–2.5)")

    return (raw_displacement * scale).astype(np.float32)


def apply_displacement(landmarks: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """
    Apply displacement vectors to a set of landmarks.

    Note: warp_face() computes target_lm = source_lm + displacement internally.
    Use this for computing target_lm in demo.py (for ETR) or for debugging.

    Args:
        landmarks:    (68, 2) source landmark positions
        displacement: (68, 2) displacement vectors

    Returns:
        new_landmarks: (68, 2) displaced landmark positions
    """
    return (landmarks + displacement).astype(np.float32)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    src = rng.uniform(100, 300, (68, 2)).astype(np.float32)
    # Give it a realistic eye layout so IOD is meaningful
    src[36] = [150, 200]; src[41] = [170, 200]
    src[42] = [210, 200]; src[47] = [230, 200]

    drv   = src + rng.uniform(-10, 10, (68, 2)).astype(np.float32)
    drv_n = src + rng.uniform(-2,   2, (68, 2)).astype(np.float32)

    print("=== Full mode ===")
    disp = compute_displacement(src, drv, drv_n)
    print(f"  shape={disp.shape}  max={np.abs(disp).max():.2f} px\n")

    print("=== Direct mode (auto scale) ===")
    disp2 = compute_displacement(src, drv)
    print(f"  shape={disp2.shape}  max={np.abs(disp2).max():.2f} px\n")

    new_lm = apply_displacement(src, disp)
    print(f"apply_displacement → shape={new_lm.shape}")
    print("expression.py OK")