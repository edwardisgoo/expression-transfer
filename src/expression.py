from __future__ import annotations
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
    - Added lm_cfg parameter to compute_displacement() so that the same code
      can handle both MediaPipe 478-point and dlib 68-point landmarks.
"""

import numpy as np

# ── Default landmark index ranges (MediaPipe 478-point model) ─────────────────
_LEFT_EYE_DEFAULT  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_DEFAULT = [362, 385, 387, 263, 373, 380]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _eye_centers(landmarks: np.ndarray,
                 left_eye: list | None  = None,
                 right_eye: list | None = None) -> tuple:
    """Return (left_eye_center, right_eye_center) averaged over the given pts."""
    le = left_eye  if left_eye  is not None else _LEFT_EYE_DEFAULT
    re = right_eye if right_eye is not None else _RIGHT_EYE_DEFAULT
    return landmarks[le].mean(axis=0), landmarks[re].mean(axis=0)


def _interocular_distance(landmarks: np.ndarray,
                          left_eye: list | None  = None,
                          right_eye: list | None = None) -> float:
    """
    Inter-ocular distance based on mean eye-centre points.

    Averaging over all 6 eye points per side is more robust to detection noise
    than using a single outer-corner point.
    """
    lc, rc = _eye_centers(landmarks, left_eye, right_eye)
    return float(np.linalg.norm(rc - lc))


def _align_landmarks(src: np.ndarray,
                     ref: np.ndarray,
                     left_eye: list | None  = None,
                     right_eye: list | None = None) -> np.ndarray:
    """
    Align src landmarks onto ref via scale + translation using eye anchors.

    Only corrects scale + translation (NOT rotation) — rotation is already
    handled by align_face() before this function is called.

    Args:
        src:       (N, 2) landmarks to align
        ref:       (N, 2) reference landmarks
        left_eye:  list of left-eye landmark indices (defaults to MP478)
        right_eye: list of right-eye landmark indices (defaults to MP478)

    Returns:
        (N, 2) src landmarks rescaled and translated to ref space
    """
    src_lc, src_rc = _eye_centers(src, left_eye, right_eye)
    ref_lc, ref_rc = _eye_centers(ref, left_eye, right_eye)

    src_iod     = np.linalg.norm(src_rc - src_lc) + 1e-8
    ref_iod     = np.linalg.norm(ref_rc - ref_lc) + 1e-8
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
        raw_displacement: (478, 2) unscaled displacement
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


def _validate_landmarks(lm: np.ndarray, name: str, n_points: int = 478) -> None:
    if lm.shape != (n_points, 2):
        raise ValueError(
            f"{name}: expected shape ({n_points}, 2), got {lm.shape}"
        )
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
    lm_cfg: dict | None = None,
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
        source_lm:         (N, 2) landmarks of the source (target) face
        driver_lm:         (N, 2) landmarks of the driver (expressive)
        driver_neutral_lm: (N, 2) optional — driver neutral baseline
        scale:             float to manually set expression strength, or
                           None (default) to auto-compute from displacement stats.
                           Typical range 0.5–1.5; auto usually lands in 1.0–2.0
                           for direct mode.
        auto_scale_ratio:  target P95 displacement as fraction of IOD (0.35).
        auto_scale_min:    auto-scale lower clamp (default 0.3).
        auto_scale_max:    auto-scale upper clamp (default 2.5).
        lm_cfg:            Optional landmark-mode config dict from
                           ``src.landmark_config.get_config()``.
                           Keys used: ``n_points``, ``left_eye``, ``right_eye``.
                           When None (default), MediaPipe 478-point layout is used.

    Returns:
        displacement: (N, 2) float32 array of (dx, dy) vectors
    """
    cfg       = lm_cfg or {}
    n_pts     = cfg.get("n_points", 478)
    left_eye  = cfg.get("left_eye",  None)
    right_eye = cfg.get("right_eye", None)

    _validate_landmarks(source_lm,  "source_lm",  n_pts)
    _validate_landmarks(driver_lm,  "driver_lm",  n_pts)
    if driver_neutral_lm is not None:
        _validate_landmarks(driver_neutral_lm, "driver_neutral_lm", n_pts)

    source_iod = _interocular_distance(source_lm, left_eye, right_eye)
    if source_iod < 1e-6:
        raise ValueError("Source landmarks degenerate — interocular distance near zero.")

    if driver_neutral_lm is not None:
        # ── Full mode ─────────────────────────────────────────────────────────
        # Align driver_neutral onto driver first so minor head movement between
        # the two shots does not contaminate the expression delta.
        driver_neutral_aligned = _align_landmarks(
            driver_neutral_lm, driver_lm, left_eye, right_eye
        )
        raw_delta = driver_lm - driver_neutral_aligned

        driver_iod = _interocular_distance(driver_lm, left_eye, right_eye)
        if driver_iod < 1e-6:
            raise ValueError("Driver landmarks degenerate — interocular distance near zero.")

        raw_displacement = raw_delta * (source_iod / driver_iod)

    else:
        # ── Direct mode ───────────────────────────────────────────────────────
        # align_face() in demo.py already removed rotation.
        # Use eye-anchor alignment (not mean-based) to map driver → source space.
        driver_aligned   = _align_landmarks(driver_lm, source_lm, left_eye, right_eye)
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
        landmarks:    (N, 2) source landmark positions
        displacement: (N, 2) displacement vectors

    Returns:
        new_landmarks: (N, 2) displaced landmark positions
    """
    return (landmarks + displacement).astype(np.float32)


def _w2(v) -> tuple:
    """Return (wx, wy) from a scalar float or a (wx, wy) tuple/list."""
    return tuple(v) if isinstance(v, (tuple, list)) else (float(v), float(v))


def apply_region_weights(
    displacement: np.ndarray,
    source_lm: np.ndarray,
    lm_cfg: dict | None = None,
    *,
    eye_weight = (0.0, 0.5),
    nose_weight: float = 0.0,
    brow_weight: float = 0.75,
    mouth_weight: float = 1.0,
    jaw_weight: float = 0.20,
    outer_weight: float = 0.15,
    smooth_sigma_ratio: float = 0.25,
) -> np.ndarray:
    """
    Per-region displacement regularisation via Gaussian RBF interpolation.

    Assigns target weights to known facial regions, then smoothly interpolates
    to all other landmarks so there are no sharp seams in the warp field.

    Each weight can be a scalar (applied to both x and y displacement) or a
    ``(wx, wy)`` tuple to decouple horizontal and vertical motion independently.

    Default weights
    ---------------
    eyes        (0.0, 0.5) — x=0 preserves IOD; y=0.5 allows squinting / widening
    nose         0.0       — preserves nose proportions and position
    eyebrows     0.75      — partial expression (brows can raise / furrow)
    mouth        1.0       — full expression transfer
    jaw / chin   jaw_weight  — strong regularisation (default 0.20)
    face oval    outer_weight — strong regularisation along silhouette (default 0.15)

    Smooth transition
    -----------------
    sigma = IOD × smooth_sigma_ratio (default ≈ 25 % of inter-ocular distance).
    Each landmark's weight = Gaussian-weighted average of all seed weights.

    Args:
        displacement:       (N, 2) from compute_displacement()
        source_lm:          (N, 2) source landmark positions (aligned space)
        lm_cfg:             landmark config from get_config(); defaults to MP478
        eye_weight:         scalar or (wx, wy) for eye-ring landmarks
                            default (0.0, 0.5): freeze horizontal, allow 50 % vertical
        nose_weight:        scalar or (wx, wy) for nose bridge/tip/wings (default 0.0)
        brow_weight:        weight for eyebrow landmarks (default 0.75)
        mouth_weight:       weight for mouth / inner-lip landmarks (default 1.0)
        jaw_weight:         weight for jaw/chin arc (default 0.20)
        outer_weight:       weight for full face-oval perimeter (default 0.15)
        smooth_sigma_ratio: Gaussian sigma as fraction of IOD (default 0.25)

    Returns:
        (N, 2) float32 weighted displacement
    """
    cfg = lm_cfg or {}

    le       = cfg.get("left_eye",       [33, 160, 158, 133, 153, 144])
    re       = cfg.get("right_eye",      [362, 385, 387, 263, 373, 380])
    le_full  = cfg.get("left_eye_full",  le)
    re_full  = cfg.get("right_eye_full", re)
    nose     = cfg.get("nose_idx",       [1, 2, 4, 5, 6, 168, 195, 197])
    brow     = cfg.get("brow_idx",       [70, 63, 105, 66, 107, 336, 296, 334, 293, 300])
    mouth    = list(set(
        cfg.get("mouth_idx",     [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91])
        + cfg.get("inner_lip_idx", [78, 191, 80, 13, 308, 402, 14, 88])
    ))
    jaw      = cfg.get("jaw_idx",        [234, 93, 132, 58, 172, 136, 150, 149, 152,
                                           377, 400, 378, 379, 365, 397, 288, 454])
    oval     = cfg.get("face_oval_idx",  jaw)

    # IOD-based sigma
    lc    = source_lm[le].mean(0)
    rc    = source_lm[re].mean(0)
    iod   = float(np.linalg.norm(rc - lc))
    sigma = max(iod * smooth_sigma_ratio, 5.0)

    # Build per-axis seed weight lists
    seed_pts: list = []
    seed_wx:  list = []
    seed_wy:  list = []
    for idx_list, w in [
        (list(set(le_full + re_full)), eye_weight),
        (nose,                         nose_weight),
        (brow,                         brow_weight),
        (mouth,                        mouth_weight),
        (jaw,                          jaw_weight),
        (oval,                         outer_weight),
    ]:
        wx, wy = _w2(w)
        for i in idx_list:
            seed_pts.append(source_lm[i])
            seed_wx.append(wx)
            seed_wy.append(wy)

    seeds = np.array(seed_pts, dtype=np.float32)    # (M, 2)
    sw_x  = np.array(seed_wx,  dtype=np.float32)   # (M,)
    sw_y  = np.array(seed_wy,  dtype=np.float32)   # (M,)

    # Gaussian RBF interpolation (shared kernel, separate per-axis weights)
    diff     = source_lm[:, None, :] - seeds[None, :, :]   # (N, M, 2)
    dists_sq = (diff ** 2).sum(axis=2)                      # (N, M)
    gauss    = np.exp(-dists_sq / (2.0 * sigma * sigma))    # (N, M)
    denom    = gauss.sum(1) + 1e-10                         # (N,)

    w_x = (gauss * sw_x[None, :]).sum(1) / denom  # (N,)
    w_y = (gauss * sw_y[None, :]).sum(1) / denom  # (N,)

    result = displacement.copy()
    result[:, 0] *= w_x
    result[:, 1] *= w_y
    return result.astype(np.float32)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.landmark_config import get_config
    rng = np.random.default_rng(42)

    # ── MediaPipe 478-point ───────────────────────────────────────────────────
    cfg_mp = get_config("mp")
    src = rng.uniform(100, 300, (478, 2)).astype(np.float32)
    src[33]  = [150, 200]; src[144] = [170, 200]   # left eye
    src[362] = [210, 200]; src[263] = [230, 200]   # right eye
    drv   = src + rng.uniform(-10, 10, (478, 2)).astype(np.float32)
    drv_n = src + rng.uniform(-2,   2, (478, 2)).astype(np.float32)

    print("=== MP478 — Full mode ===")
    disp = compute_displacement(src, drv, drv_n, lm_cfg=cfg_mp)
    print(f"  shape={disp.shape}  max={np.abs(disp).max():.2f} px\n")

    print("=== MP478 — Direct mode (auto scale) ===")
    disp2 = compute_displacement(src, drv, lm_cfg=cfg_mp)
    print(f"  shape={disp2.shape}  max={np.abs(disp2).max():.2f} px\n")

    # ── dlib 68-point ─────────────────────────────────────────────────────────
    cfg_dl = get_config("dlib")
    src68 = rng.uniform(100, 300, (68, 2)).astype(np.float32)
    src68[36] = [150, 200]; src68[41] = [170, 200]  # left eye
    src68[42] = [210, 200]; src68[47] = [230, 200]  # right eye
    drv68 = src68 + rng.uniform(-10, 10, (68, 2)).astype(np.float32)

    print("=== DLIB68 — Direct mode ===")
    disp3 = compute_displacement(src68, drv68, lm_cfg=cfg_dl)
    print(f"  shape={disp3.shape}  max={np.abs(disp3).max():.2f} px\n")

    new_lm = apply_displacement(src, disp)
    print(f"apply_displacement → shape={new_lm.shape}")
    print("expression.py OK")