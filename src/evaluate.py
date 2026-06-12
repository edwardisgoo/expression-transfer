from __future__ import annotations
"""
evaluate.py — Expression Transfer Evaluation Metrics

Computes quantitative metrics to assess the quality of an expression transfer result.

Metrics
-------
1. SSIM (full image)
       Structural similarity between result and source.
       High = background and identity are well preserved.

2. SSIM (face region only)
       Restricted to the face mask bounding box.
       Lower than source is expected when a strong expression was transferred —
       this is a sign of success, not failure.

3. PSNR (full image)
       Peak signal-to-noise ratio (dB).
       < 30 dB = visible difference;  > 40 dB = nearly identical.

4. Color Drift (face region)
       Mean absolute per-channel colour difference between result and source
       inside the face mask (pixel units, 0–255).
       Note: some drift is inherent — geometric warping moves facial regions
       to new pixel positions, so result ≠ source pixel-for-pixel even with
       perfect colour preservation.  Strong expressions on different people
       typically produce 10–18 px drift.

5. Background Preservation
       SSIM on the non-face region, excluding a 20 px buffer around the mask
       boundary where seamlessClone's Poisson halo is expected.
       Threshold ≥ 0.98 on the *true* background (outside halo zone).

6. ETR  (Expression Transfer Ratio)
       Detects landmarks on the result image and measures how much of the
       intended landmark displacement was actually achieved.

           ETR = mean(||result_lm - source_lm||)
               / mean(||target_lm  - source_lm||)

       All three landmark sets must be in the same coordinate space.
       demo.py passes source_lm and target_lm in original (unaligned) space;
       detect_landmarks(result) also returns original space. ✓

       ETR ≈ 1.0  → displacement fully realised
       ETR 0.7–1.2 → good range
       ETR < 0.65 → scale may be past pipeline ceiling; try driver-neutral
       ETR > 1.2  → over-shoot (scale too high)

7. Landmark RMSE (result vs driver)
       After aligning result and driver to a common frame via full similarity
       transform (scale + rotation + translation), computes the RMS landmark
       distance.  Lower = result expression is closer to driver's expression.
       Uses similarity (not just scale+translation) to absorb any residual
       head-tilt between result space (original) and driver_lm space (aligned).
"""

import json
import numpy as np
import cv2

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    print("[evaluate] Warning: scikit-image not found — SSIM will use OpenCV fallback.")

# ── LPIPS (lazy-loaded on first use) ─────────────────────────────────────────
_LPIPS_MODEL = None

def _get_lpips_model():
    """Lazy-initialise AlexNet LPIPS model (loads once, stays in memory)."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        try:
            import lpips
            _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False)
            _LPIPS_MODEL.eval()
        except Exception as e:
            print(f"[evaluate] LPIPS not available ({e}); perceptual metric skipped.")
            _LPIPS_MODEL = False   # sentinel: don't retry
    return _LPIPS_MODEL if _LPIPS_MODEL is not False else None


def _lpips_face(src: np.ndarray, res: np.ndarray, mask: np.ndarray) -> float | None:
    """
    Compute LPIPS perceptual distance between source and result, restricted
    to the face bounding box.  Returns a float in [0, 1] (lower = more similar).
    Returns None if LPIPS is unavailable.
    """
    model = _get_lpips_model()
    if model is None:
        return None

    # ── Crop to face bounding box ──────────────────────────────────────────────
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_src = src[y0:y1, x0:x1]
    crop_res = res[y0:y1, x0:x1]

    # ── Resize to at least 64×64 (LPIPS minimum) ──────────────────────────────
    h, w = crop_src.shape[:2]
    side  = max(64, h, w)
    crop_src = cv2.resize(crop_src, (side, side), interpolation=cv2.INTER_AREA)
    crop_res = cv2.resize(crop_res, (side, side), interpolation=cv2.INTER_AREA)

    # ── Convert BGR→RGB, HWC→CHW, uint8→float32 in [-1, 1] ───────────────────
    try:
        import torch
        def _to_tensor(img_bgr):
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            t   = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0)
            return t.permute(2, 0, 1).unsqueeze(0)          # (1, 3, H, W)

        with torch.no_grad():
            d = model(_to_tensor(crop_src), _to_tensor(crop_res))
        return float(d.item())
    except Exception as e:
        print(f"[evaluate] LPIPS compute failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM between two BGR images (full frame)."""
    if _HAS_SKIMAGE:
        return float(_ssim_fn(_to_gray(a), _to_gray(b), data_range=255))
    # OpenCV fallback
    ga, gb   = _to_gray(a).astype(float), _to_gray(b).astype(float)
    mu_a, mu_b = ga.mean(), gb.mean()
    sig_a, sig_b = ga.std(), gb.std()
    sig_ab = ((ga - mu_a) * (gb - mu_b)).mean()
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(
        (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2) /
        ((mu_a ** 2 + mu_b ** 2 + C1) * (sig_a ** 2 + sig_b ** 2 + C2))
    )


def _ssim_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """SSIM computed only inside the mask bounding box."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return float("nan")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return _ssim_score(a[y0:y1, x0:x1], b[y0:y1, x0:x1])


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def _color_drift(source: np.ndarray, result: np.ndarray, mask: np.ndarray) -> dict:
    """
    Mean absolute per-channel colour difference inside the face mask.
    Returns {'B', 'G', 'R', 'mean'} in pixel units (0–255).
    """
    idx = mask > 0
    if not idx.any():
        nan = float("nan")
        return {"B": nan, "G": nan, "R": nan, "mean": nan}
    diffs: dict[str, float] = {}
    for i, ch in enumerate(["B", "G", "R"]):
        diffs[ch] = float(np.abs(
            result[:, :, i].astype(float)[idx] -
            source[:, :, i].astype(float)[idx]
        ).mean())
    diffs["mean"] = float(np.mean([diffs["B"], diffs["G"], diffs["R"]]))
    return diffs


_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Expression-relevant landmark groups (MediaPipe 478 defaults)
_BROW_IDX  = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
_MOUTH_IDX = [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
              78, 191, 80, 13, 308, 402, 14, 88]
# Jaw/chin: stable, expression-independent — used for identity preservation
_JAW_IDX   = [234, 93, 132, 58, 172, 136, 150, 149, 152,
              377, 400, 378, 379, 365, 397, 288, 454]
# Expression-relevant subset used for ETR (brows + outer lips + inner lips)
_EXPR_IDX  = (
    _BROW_IDX
    + [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91]   # outer lips
    + [78, 191, 80, 13, 308, 402, 14, 88]                     # inner lips
)  # 30 landmarks total

# MediaPipe blendshape indices that correspond to visible expression AUs.
# Index 0 (_neutral) is excluded; eye-gaze blendshapes are also excluded
# since they reflect look direction rather than expression.
_EXPR_BS_IDX = [
    1,  2,  3,  4,  5,   # browDown L/R, browInnerUp, browOuterUp L/R
    6,  7,  8,           # cheekPuff, cheekSquint L/R
    9, 10,               # eyeBlink L/R
    19, 20, 21, 22,      # eyeSquint L/R, eyeWide L/R
    25,                  # jawOpen
    27, 28, 29,          # mouthClose, mouthDimple L/R
    30, 31, 32,          # mouthFrown L/R, mouthFunnel
    34, 35,              # mouthLowerDown L/R
    38, 39,              # mouthPucker, mouthRight
    40, 41, 42, 43,      # mouthRollLower/Upper, mouthShrugLower/Upper
    44, 45,              # mouthSmile L/R
    46, 47,              # mouthStretch L/R
    48, 49,              # mouthUpperUp L/R
    50, 51,              # noseSneer L/R
]


def _eye_centers(lm: np.ndarray,
                 left_eye: list | None  = None,
                 right_eye: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_eye_center, right_eye_center) averaged over the given pts."""
    le = left_eye  if left_eye  is not None else _LEFT_EYE
    re = right_eye if right_eye is not None else _RIGHT_EYE
    return lm[le].mean(axis=0), lm[re].mean(axis=0)


def _region_rmse(aligned_result: np.ndarray, ref_lm: np.ndarray, idx: list) -> float:
    """RMSE between aligned_result and ref_lm restricted to landmark indices idx."""
    diff = aligned_result[idx] - ref_lm[idx]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def _blendshape_similarity(bs_driver: np.ndarray, bs_result: np.ndarray) -> dict:
    """
    Compare two (52,) blendshape score arrays on expression-relevant AUs only.

    Returns:
        cosine_sim : float  — 1.0 = identical AUs, 0 = orthogonal  (↑ better)
        mae        : float  — mean absolute error across AUs (0–1 scale)  (↓ better)
    """
    a = bs_driver[_EXPR_BS_IDX].astype(float)
    b = bs_result[_EXPR_BS_IDX].astype(float)
    norm_a = np.linalg.norm(a) + 1e-8
    norm_b = np.linalg.norm(b) + 1e-8
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    mae    = float(np.mean(np.abs(a - b)))
    return {"cosine_sim": cosine, "mae": mae}


def _align_lm_similarity(src_lm: np.ndarray,
                         ref_lm: np.ndarray,
                         left_eye: list | None  = None,
                         right_eye: list | None = None) -> np.ndarray:
    """
    Map src_lm into ref_lm's coordinate frame via similarity transform
    (scale + rotation + translation), anchored on eye-centre points.

    This is used for LM RMSE to neutralise both the scale difference between
    faces AND any residual head-tilt between result space (original, unaligned)
    and driver_lm space (align_face-rotated).  Using only scale+translation
    (_align_lm_to_ref) would leave rotation error that inflates the RMSE.
    """
    src_l, src_r = _eye_centers(src_lm, left_eye, right_eye)
    ref_l, ref_r = _eye_centers(ref_lm, left_eye, right_eye)

    src_vec  = src_r - src_l
    ref_vec  = ref_r - ref_l
    src_iod  = float(np.linalg.norm(src_vec)) + 1e-8
    ref_iod  = float(np.linalg.norm(ref_vec)) + 1e-8
    scale    = ref_iod / src_iod

    theta    = float(np.arctan2(ref_vec[1], ref_vec[0])) \
             - float(np.arctan2(src_vec[1], src_vec[0]))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]], dtype=np.float32)

    src_ctr = (src_l + src_r) / 2.0
    ref_ctr = (ref_l + ref_r) / 2.0

    return ((src_lm - src_ctr) @ R.T * scale + ref_ctr).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    source_img: np.ndarray,
    result_img: np.ndarray,
    face_mask:  np.ndarray,
    source_lm:  np.ndarray,
    target_lm:  np.ndarray,
    driver_lm:  np.ndarray,
    detect_fn=None,
    driver_blendshapes: np.ndarray = None,
    detect_bs_fn=None,
    lm_cfg: dict | None = None,
) -> dict:
    """
    Compute all evaluation metrics for one expression transfer result.

    Args:
        source_img: BGR source image (before transfer)
        result_img: BGR result image (after transfer)
        face_mask:  (H, W) uint8 mask — 255 = face region
        source_lm:  (N, 2) source landmarks in ORIGINAL (unaligned) space
        target_lm:  (N, 2) intended displaced landmarks in ORIGINAL space
                    (= source_lm + inv_transform(displacement), computed in demo.py)
        driver_lm:  (N, 2) driver landmarks in ALIGNED space (drv_lm_aligned)
        detect_fn:  callable(image) → (N, 2) or None.
                    Pass detect_landmarks from landmark.py to enable ETR + RMSE.
                    If None, both metrics are reported as None.
        lm_cfg:     Optional landmark-mode config dict from
                    ``src.landmark_config.get_config()``.
                    Keys used: ``left_eye``, ``right_eye``, ``brow_idx``,
                    ``mouth_idx``, ``jaw_idx``, ``expr_idx``.
                    When None (default), MediaPipe 478-point indices are used.

    Returns:
        metrics: dict — keys listed in module docstring
    """
    cfg       = lm_cfg or {}
    left_eye  = cfg.get("left_eye",  None)   # None → _eye_centers uses _LEFT_EYE
    right_eye = cfg.get("right_eye", None)
    brow_idx  = cfg.get("brow_idx",  _BROW_IDX)
    mouth_idx = cfg.get("mouth_idx", _MOUTH_IDX)
    jaw_idx   = cfg.get("jaw_idx",   _JAW_IDX)
    expr_idx  = cfg.get("expr_idx",  _EXPR_IDX)
    metrics: dict = {}

    # ── 1. SSIM full ─────────────────────────────────────────────────────────
    metrics["ssim_full"]     = _ssim_score(source_img, result_img)

    # ── 2. SSIM face ─────────────────────────────────────────────────────────
    metrics["ssim_face"]     = _ssim_masked(source_img, result_img, face_mask)

    # ── 3. PSNR full ─────────────────────────────────────────────────────────
    metrics["psnr_full_db"]  = _psnr(source_img, result_img)

    # ── 4. Color drift ───────────────────────────────────────────────────────
    metrics["color_drift"]   = _color_drift(source_img, result_img, face_mask)

    # ── 5. Background SSIM (excluding Poisson halo band) ─────────────────────
    # seamlessClone propagates gradients ~15–20 px outside the mask boundary.
    # We dilate the mask by 20 px and evaluate only pixels outside that zone
    # so the metric reflects true background preservation, not expected halos.
    dilate_k = np.ones((41, 41), np.uint8)          # radius ≈ 20 px
    dilated  = cv2.dilate(face_mask, dilate_k, iterations=1)
    bg_mask  = (dilated == 0).astype(np.uint8) * 255
    metrics["ssim_background"] = _ssim_masked(source_img, result_img, bg_mask)

    # ── 5b. LPIPS (perceptual face quality) ──────────────────────────────────
    # AlexNet-based perceptual distance on the face crop.
    # 0 = identical, ~1 = very different.  Correlates better with human
    # "looks natural" judgement than pixel-level SSIM/PSNR.
    metrics["lpips_face"] = _lpips_face(source_img, result_img, face_mask)

    # ── 6. ETR  ──────────────────────────────────────────────────────────────
    # Only measure on expression-relevant landmarks (brows + mouth).
    # Jaw, nose, and cheeks barely move with expression; including them
    # shrinks the denominator and inflates ETR unpredictably.
    # expr_idx comes from lm_cfg when provided, else module-level _EXPR_IDX.

    metrics["etr"]               = None
    metrics["lm_rmse_vs_driver"] = None
    metrics["brow_rmse"]         = None
    metrics["mouth_rmse"]        = None
    metrics["jaw_rmse"]          = None
    metrics["blendshape"]        = None

    if detect_fn is None:
        return metrics

    result_lm = detect_fn(result_img)
    if result_lm is None:
        print("[evaluate] Warning: landmark detection failed on result — "
              "ETR and LM RMSE skipped.")
        return metrics

    # ── 6. ETR ───────────────────────────────────────────────────────────────
    sl = source_lm[expr_idx]
    tl = target_lm[expr_idx]
    rl = result_lm[expr_idx]
    intended = np.linalg.norm(tl - sl, axis=1).mean()
    achieved = np.linalg.norm(rl - sl, axis=1).mean()
    if intended > 1e-6:
        metrics["etr"] = float(achieved / intended)

    # ── 7–9. Per-region RMSE vs driver ───────────────────────────────────────
    # Align result into driver's coordinate frame once, then evaluate each
    # region separately.  Three regions tell three different stories:
    #   brow_rmse   — how well brow expression matches driver
    #   mouth_rmse  — how well mouth expression matches driver
    #   jaw_rmse    — how much non-expression geometry shifted (identity cost)
    result_in_drv = _align_lm_similarity(result_lm, driver_lm, left_eye, right_eye)
    metrics["lm_rmse_vs_driver"] = float(np.sqrt(np.mean(
        np.sum((result_in_drv - driver_lm) ** 2, axis=1)
    )))
    metrics["brow_rmse"]  = _region_rmse(result_in_drv, driver_lm, brow_idx)
    metrics["mouth_rmse"] = _region_rmse(result_in_drv, driver_lm, mouth_idx)
    metrics["jaw_rmse"]   = _region_rmse(result_in_drv, driver_lm, jaw_idx)

    # ── 10. Blendshape similarity ─────────────────────────────────────────────
    # Compare MediaPipe AU scores between driver and result.
    # cosine_sim ≈ 1.0 → AUs match; mae ≈ 0 → AU intensities match.
    # This is independent of the landmark pipeline and measures perceived
    # expression directly.
    if driver_blendshapes is not None and detect_bs_fn is not None:
        result_bs = detect_bs_fn(result_img)
        if result_bs is not None:
            metrics["blendshape"] = _blendshape_similarity(
                driver_blendshapes, result_bs
            )

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dict to stdout."""
    cd   = metrics.get("color_drift", {})
    etr  = metrics.get("etr")
    rmse = metrics.get("lm_rmse_vs_driver")

    sep = "═" * 52
    print(f"\n{sep}")
    print("  Evaluation Metrics")
    print(sep)

    # ── Image quality ─────────────────────────────────────────────────────────
    print(f"  SSIM  full image   : {metrics['ssim_full']:.4f}"
          f"   (^ higher = less overall change)")
    print(f"  SSIM  face region  : {metrics['ssim_face']:.4f}"
          f"   (lower = more expression transferred)")
    print(f"  PSNR  full image   : {metrics['psnr_full_db']:.2f} dB")

    bg = metrics.get("ssim_background", float("nan"))
    if bg != bg:   # NaN
        print(f"  SSIM  background   : N/A  (no background pixels outside halo zone)")
    else:
        ssim_full = metrics.get("ssim_full", float("nan"))
        if abs(bg - ssim_full) < 1e-4:
            bg_note = "~= full SSIM (face is small relative to image -- normal)"
        elif bg >= 0.98:
            bg_note = "OK"
        else:
            bg_note = "v warp leaking outside face"
        print(f"  SSIM  background   : {bg:.4f}   ({bg_note})")

    # ── Colour ────────────────────────────────────────────────────────────────
    if isinstance(cd, dict) and "mean" in cd:
        drift_note = ""
        if cd["mean"] > 18:
            drift_note = "  <- high; partly from expression geometry"
        elif cd["mean"] > 12:
            drift_note = "  <- moderate; expected for strong expressions"
        print(f"  Color drift face   : {cd['mean']:.2f} px"
              f"  (B={cd['B']:.1f} G={cd['G']:.1f} R={cd['R']:.1f})"
              f"{drift_note}")

    # ── Expression transfer ───────────────────────────────────────────────────
    if etr is not None:
        if etr > 1.2:
            etr_note = "^ over-shoot -- lower scale"
        elif etr >= 0.7:
            etr_note = "OK good"
        elif etr >= 0.5:
            etr_note = "v under-transfer -- scale may be past pipeline ceiling"
        else:
            etr_note = "v under-transfer -- try driver-neutral or better source"
        print(f"  ETR transfer ratio : {etr:.3f}   ({etr_note})")
    else:
        print(f"  ETR transfer ratio : N/A  (pass detect_fn to enable)")

    if rmse is not None:
        rmse_note = "OK close" if rmse < 15 else ("moderate" if rmse < 25 else "^ far from driver")
        print(f"  LM RMSE vs driver  : {rmse:.2f} px  ({rmse_note})")
    else:
        print(f"  LM RMSE vs driver  : N/A")

    # ── Per-region RMSE ───────────────────────────────────────────────────────
    brow_r  = metrics.get("brow_rmse")
    mouth_r = metrics.get("mouth_rmse")
    jaw_r   = metrics.get("jaw_rmse")
    if brow_r is not None:
        print(f"  Brow  RMSE         : {brow_r:.2f} px"
              f"  ({'OK' if brow_r < 8 else '^ high'})")
    if mouth_r is not None:
        print(f"  Mouth RMSE         : {mouth_r:.2f} px"
              f"  ({'OK' if mouth_r < 8 else '^ high'})")
    if jaw_r is not None:
        jaw_note = "OK identity preserved" if jaw_r < 6 else (
                   "moderate drift" if jaw_r < 12 else "^ identity distorted")
        print(f"  Jaw   RMSE         : {jaw_r:.2f} px  ({jaw_note})")

    # ── Blendshape similarity ─────────────────────────────────────────────────
    bs = metrics.get("blendshape")
    if bs is not None:
        cos  = bs["cosine_sim"]
        mae  = bs["mae"]
        cos_note = "OK good" if cos > 0.90 else ("moderate" if cos > 0.75 else "v low -- expression mismatch")
        mae_note = "OK" if mae < 0.05 else ("moderate" if mae < 0.10 else "^ high AU divergence")
        print(f"  Blendshape cosine  : {cos:.3f}   ({cos_note})")
        print(f"  Blendshape MAE     : {mae:.3f}   ({mae_note})")

    print(f"{sep}\n")


def save_metrics_json(metrics: dict, path: str) -> None:
    """Serialise metrics to JSON (NaN → null, nested dicts handled)."""
    def _serial(obj):
        if isinstance(obj, float):
            return None if obj != obj else obj     # NaN → null
        if isinstance(obj, dict):
            return {k: _serial(v) for k, v in obj.items()}
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serial(metrics), f, indent=2)
    print(f"[evaluate] Saved metrics → {path}")