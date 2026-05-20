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

from __future__ import annotations
import json
import numpy as np
import cv2

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    print("[evaluate] Warning: scikit-image not found — SSIM will use OpenCV fallback.")


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


def _eye_centers(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_eye_center, right_eye_center) averaged over 6 pts each."""
    return lm[36:42].mean(axis=0), lm[42:48].mean(axis=0)


def _align_lm_similarity(src_lm: np.ndarray, ref_lm: np.ndarray) -> np.ndarray:
    """
    Map src_lm into ref_lm's coordinate frame via similarity transform
    (scale + rotation + translation), anchored on eye-centre points.

    This is used for LM RMSE to neutralise both the scale difference between
    faces AND any residual head-tilt between result space (original, unaligned)
    and driver_lm space (align_face-rotated).  Using only scale+translation
    (_align_lm_to_ref) would leave rotation error that inflates the RMSE.
    """
    src_l, src_r = _eye_centers(src_lm)
    ref_l, ref_r = _eye_centers(ref_lm)

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
) -> dict:
    """
    Compute all evaluation metrics for one expression transfer result.

    Args:
        source_img: BGR source image (before transfer)
        result_img: BGR result image (after transfer)
        face_mask:  (H, W) uint8 mask — 255 = face region
        source_lm:  (68, 2) source landmarks in ORIGINAL (unaligned) space
        target_lm:  (68, 2) intended displaced landmarks in ORIGINAL space
                    (= source_lm + inv_transform(displacement), computed in demo.py)
        driver_lm:  (68, 2) driver landmarks in ALIGNED space (drv_lm_aligned)
        detect_fn:  callable(image) → (68, 2) or None.
                    Pass detect_landmarks from landmark.py to enable ETR + RMSE.
                    If None, both metrics are reported as None.

    Returns:
        metrics: dict — keys listed in module docstring
    """
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

    # ── 6. ETR  ──────────────────────────────────────────────────────────────
    metrics["etr"]               = None
    metrics["lm_rmse_vs_driver"] = None

    if detect_fn is None:
        return metrics

    result_lm = detect_fn(result_img)
    if result_lm is None:
        print("[evaluate] Warning: landmark detection failed on result — "
              "ETR and LM RMSE skipped.")
        return metrics

    # ETR — all in original (unaligned) space
    intended = np.linalg.norm(target_lm - source_lm, axis=1).mean()
    achieved = np.linalg.norm(result_lm - source_lm, axis=1).mean()
    if intended > 1e-6:
        metrics["etr"] = float(achieved / intended)

    # ── 7. LM RMSE vs driver ─────────────────────────────────────────────────
    # result_lm is in original space; driver_lm is in aligned space.
    # _align_lm_similarity maps result_lm into driver_lm's frame using
    # scale + rotation + translation so neither scale nor tilt inflate RMSE.
    result_in_drv = _align_lm_similarity(result_lm, driver_lm)
    metrics["lm_rmse_vs_driver"] = float(np.sqrt(np.mean(
        np.sum((result_in_drv - driver_lm) ** 2, axis=1)
    )))

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
          f"   (↑ higher = less overall change)")
    print(f"  SSIM  face region  : {metrics['ssim_face']:.4f}"
          f"   (lower = more expression transferred)")
    print(f"  PSNR  full image   : {metrics['psnr_full_db']:.2f} dB")

    bg = metrics.get("ssim_background", float("nan"))
    if bg == bg:   # not NaN
        bg_note = "✓" if bg >= 0.98 else "↓ warp leaking outside face"
        print(f"  SSIM  background   : {bg:.4f}   ({bg_note})")
    else:
        print(f"  SSIM  background   : N/A  (no background pixels outside halo zone)")

    # ── Colour ────────────────────────────────────────────────────────────────
    if isinstance(cd, dict) and "mean" in cd:
        drift_note = ""
        if cd["mean"] > 18:
            drift_note = "  ← high; partly from expression geometry"
        elif cd["mean"] > 12:
            drift_note = "  ← moderate; expected for strong expressions"
        print(f"  Color drift face   : {cd['mean']:.2f} px"
              f"  (B={cd['B']:.1f} G={cd['G']:.1f} R={cd['R']:.1f})"
              f"{drift_note}")

    # ── Expression transfer ───────────────────────────────────────────────────
    if etr is not None:
        if etr > 1.2:
            etr_note = "↑ over-shoot — lower scale"
        elif etr >= 0.7:
            etr_note = "✓ good"
        elif etr >= 0.5:
            etr_note = "↓ under-transfer — scale may be past pipeline ceiling"
        else:
            etr_note = "↓ under-transfer — try driver-neutral or better source"
        print(f"  ETR transfer ratio : {etr:.3f}   ({etr_note})")
    else:
        print(f"  ETR transfer ratio : N/A  (pass detect_fn to enable)")

    if rmse is not None:
        rmse_note = "✓ close" if rmse < 15 else ("moderate" if rmse < 25 else "↑ far from driver")
        print(f"  LM RMSE vs driver  : {rmse:.2f} px  ({rmse_note})")
    else:
        print(f"  LM RMSE vs driver  : N/A")

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