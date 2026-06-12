"""
demo.py — End-to-end expression transfer demo
Usage: python demo.py --source <path> --driver <path> [--driver-neutral <path>] [--scale 0.9]
"""
from __future__ import annotations

import argparse
import sys
import os

# Print immediately (before slow library imports) so the user knows the
# script is alive.  flush=True is important on Windows where stdout is
# line-buffered and output can be delayed otherwise.
print("Starting — loading libraries (scipy / mediapipe may take ~20 s on first run)...",
      flush=True)

import cv2
import numpy as np

# Make repo root importable, so we can import src.* as a package
sys.path.insert(0, os.path.dirname(__file__))

from src.landmark_config import get_config as _get_lm_config
from src.expression import compute_displacement, apply_displacement, apply_region_weights
from src.warp import warp_face
from src.blend import blend, save_comparison, color_correct
from src.align import align_face
from src.evaluate import compute_metrics, print_metrics, save_metrics_json


def _load_image(path: str) -> np.ndarray | None:
    """
    Load image with defensive handling for common real-world issues:

    1. RGBA PNG (4-channel) — converted to BGR; ignores transparency.
    2. Grayscale image (1-channel) — promoted to 3-channel BGR.
    3. EXIF auto-rotation — JPEG photos from phones are often stored
       rotated 90°/180°; cv2.imread ignores EXIF, so we correct it here
       via PIL when available.  If PIL is absent we proceed as-is and
       print a warning so the caller knows.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # ── Channel normalisation ──────────────────────────────────────────────
    if img.ndim == 2:
        # Pure grayscale (e.g. some medical / passport photos)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        # RGBA PNG — drop alpha, keep BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] != 3:
        print(f"[load_image] Unexpected channel count {img.shape[2]} in {path} — skipping.")
        return None

    # ── EXIF rotation correction (JPEG only) ──────────────────────────────
    if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg"):
        try:
            from PIL import Image as _PIL, ExifTags as _ExifTags
            _ORIENT_TAG = next(k for k, v in _ExifTags.TAGS.items() if v == "Orientation")
            with _PIL.open(path) as _pil:
                orient = (_pil.getexif() or {}).get(_ORIENT_TAG, 1)
            _ROT = {
                3: cv2.ROTATE_180,
                6: cv2.ROTATE_90_CLOCKWISE,
                8: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            if orient in _ROT:
                img = cv2.rotate(img, _ROT[orient])
        except ImportError:
            print("[load_image] PIL not found — EXIF rotation not corrected. "
                  "Install Pillow if images appear rotated.")
        except Exception:
            pass  # No EXIF or unreadable — proceed as-is

    return img


def _transform_landmarks(lm: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2×3 affine matrix M to a (N, 2) landmark array."""
    ones = np.ones((lm.shape[0], 1), dtype=np.float32)
    pts  = np.hstack([lm.astype(np.float32), ones])
    return (pts @ M.T).astype(np.float32)


def _build_detector(landmark_mode: str):
    """
    Return (detect_fn, detect_bs_fn) for the requested landmark mode.

    detect_fn     : image → (N, 2) ndarray or None
    detect_bs_fn  : image → (52,) ndarray or None   (None for dlib mode)
    """
    if landmark_mode == "mp":
        from src.landmark import detect_landmarks, detect_blendshapes
        return detect_landmarks, detect_blendshapes
    elif landmark_mode == "dlib":
        from src.landmark_dlib import detect_landmarks as _det
        return _det, None
    else:
        raise ValueError(
            f"Unknown landmark mode {landmark_mode!r}. Choose 'mp' or 'dlib'."
        )


def run(source_path, driver_path=None, driver_neutral_path=None, scale=0.7, output_dir="output",
        run_eval=True, save_metrics=False, landmark_mode="mp", file_prefix=None,
        atlas_path=None, expr_name=None, region_warp=True,
        jaw_reg=0.20, outer_reg=0.15):
    os.makedirs(output_dir, exist_ok=True)

    # ── Landmark backend ──────────────────────────────────────────────────────
    lm_cfg = _get_lm_config(landmark_mode)
    detect_landmarks, detect_blendshapes = _build_detector(landmark_mode)
    print(f"[landmark] mode = {landmark_mode}  ({lm_cfg['n_points']} points)")

    _atlas_mode = atlas_path is not None and expr_name is not None

    print("[1/5] Loading images...")
    source_img = _load_image(source_path)
    if source_img is None:
        print("Error: could not load source image.")
        sys.exit(1)

    if _atlas_mode:
        driver_img = None
        driver_neutral_img = None
        print(f"[driver] Atlas mode — expression: '{expr_name}'  ({atlas_path})")
    else:
        driver_img = _load_image(driver_path)
        if driver_img is None:
            print("Error: could not load driver image.")
            sys.exit(1)
        if driver_neutral_path:
            driver_neutral_img = _load_image(driver_neutral_path)
            if driver_neutral_img is None:
                print("Error: could not load driver-neutral image.")
                sys.exit(1)
        else:
            print("[!] No driver neutral provided — using direct warp mode.")
            print("    Source landmarks will be warped directly toward driver face geometry.")
            driver_neutral_img = None

    print("[2/5] Detecting landmarks...")
    source_lm = detect_landmarks(source_img)
    if source_lm is None:
        print("Error: landmark detection failed on source image.")
        sys.exit(1)

    if _atlas_mode:
        # Load pre-computed canonical landmarks from atlas (no per-image detection needed)
        _atlas_data = np.load(atlas_path)
        _lm_key     = f"{expr_name}_landmarks"
        _bs_key     = f"{expr_name}_blendshapes"
        if _lm_key not in _atlas_data.files:
            available = [k.replace("_landmarks", "") for k in _atlas_data.files
                         if k.endswith("_landmarks") and not k.endswith("_std")]
            print(f"Error: expression '{expr_name}' not found in atlas.")
            print(f"  Available: {available}")
            sys.exit(1)
        drv_lm_aligned    = _atlas_data[_lm_key]
        driver_bs         = _atlas_data[_bs_key] if _bs_key in _atlas_data.files else None
        driver_neutral_lm = None
        drv_aligned       = None
        drvN_lm_aligned   = None
        print(f"[landmark] Loaded '{expr_name}' from atlas  ({drv_lm_aligned.shape[0]} pts)")
    else:
        driver_lm = detect_landmarks(driver_img)
        if driver_lm is None:
            print("Error: landmark detection failed on driver image.")
            sys.exit(1)
        driver_neutral_lm = detect_landmarks(driver_neutral_img) if driver_neutral_img is not None else None
        driver_bs = detect_blendshapes(driver_img) if detect_blendshapes is not None else None

    print("[3/5] Face alignment (stabilization)...")
    src_aligned, src_lm_aligned, M_src, M_src_inv = align_face(source_img, source_lm, lm_cfg=lm_cfg)

    if not _atlas_mode:
        drv_aligned, drv_lm_aligned, _, _ = align_face(driver_img, driver_lm, lm_cfg=lm_cfg)
        if driver_neutral_img is not None and driver_neutral_lm is not None:
            _, drvN_lm_aligned, _, _ = align_face(driver_neutral_img, driver_neutral_lm, lm_cfg=lm_cfg)
        else:
            drvN_lm_aligned = None

    print("[4/5] Computing displacement & warping in aligned space...")
    displacement = compute_displacement(src_lm_aligned, drv_lm_aligned, drvN_lm_aligned, scale=scale, lm_cfg=lm_cfg)

    if region_warp:
        displacement = apply_region_weights(
            displacement, src_lm_aligned, lm_cfg=lm_cfg,
            jaw_weight=jaw_reg, outer_weight=outer_reg,
        )
        print(f"[region-warp] eye=(x=0.00,y=0.50)  nose=0.00  brow=0.75  mouth=1.00  "
              f"jaw={jaw_reg:.2f}  outer={outer_reg:.2f}")

    # target_lm in aligned space
    target_lm_aligned = apply_displacement(src_lm_aligned, displacement)

    # For ETR, compute the target landmarks back in original space
    target_lm_orig = _transform_landmarks(target_lm_aligned, M_src_inv)

    warped_aligned, face_mask_aligned = warp_face(src_aligned, src_lm_aligned, displacement,
                                                   lm_cfg=lm_cfg)

    # Mouth inner-lip mask (used both for face_mask carve-out and mouth clone gate)
    try:
        inner_poly = target_lm_aligned[lm_cfg["inner_lip_idx"]].astype(np.int32)
        mouth_inner = np.zeros(face_mask_aligned.shape, dtype=np.uint8)
        cv2.fillPoly(mouth_inner, [inner_poly], 255)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mouth_inner_eroded = cv2.erode(mouth_inner, k3, iterations=1)
        face_mask_aligned_nouth = face_mask_aligned.copy()
        inv_inner = cv2.bitwise_not(mouth_inner_eroded)
        face_mask_aligned_nouth = cv2.bitwise_and(face_mask_aligned_nouth, inv_inner)
    except Exception:
        face_mask_aligned_nouth = face_mask_aligned
        mouth_inner = None

    # Region-aware compositing — warp driver image onto target landmarks for mouth clone
    # (skipped in atlas mode: no driver image available to warp)
    if not _atlas_mode:
        try:
            from src.warp import warp_image_by_landmarks  # type: ignore
            driver_to_target_aligned = warp_image_by_landmarks(drv_aligned, drv_lm_aligned, target_lm_aligned)
        except Exception as e:
            print(f"[demo] Driver warp error (non-fatal): {e}")
            driver_to_target_aligned = None
    else:
        driver_to_target_aligned = None

    h, w = source_img.shape[:2]
    warped_img = cv2.warpAffine(warped_aligned, M_src_inv, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # Use face mask with mouth carved out for global blend
    face_mask  = cv2.warpAffine(face_mask_aligned_nouth, M_src_inv, (w, h),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    print("[5/5] Blending...")
    base_result = blend(source_img, warped_img, face_mask)

    # Second pass: paste driver mouth in original space
    # ── Gate: only clone when driver mouth is visibly open ───────────────────
    # Measure inner-lip vertical gap normalised by inter-ocular distance (IOD).
    # If the driver's mouth is closed (or barely open), cloning would transplant
    # closed-lip texture / phantom teeth onto the result — so we skip it and let
    # the Delaunay warp carry the mouth region on its own.
    _CLONE_OPEN_THRESH = 0.08   # 8 % of IOD ≈ clearly parted lips
    _mouth_open_ratio  = 0.0
    if driver_to_target_aligned is not None and mouth_inner is not None:
        _inner_lm = drv_lm_aligned[lm_cfg["inner_lip_idx"]]
        _le       = drv_lm_aligned[lm_cfg["left_eye"]]
        _re       = drv_lm_aligned[lm_cfg["right_eye"]]
        _iod      = float(np.linalg.norm(_le.mean(0) - _re.mean(0)))
        _mouth_open_ratio = float(_inner_lm[:, 1].max() - _inner_lm[:, 1].min()) / max(_iod, 1.0)

    if driver_to_target_aligned is not None and mouth_inner is not None \
            and _mouth_open_ratio >= _CLONE_OPEN_THRESH:
        driver_to_target_orig = cv2.warpAffine(driver_to_target_aligned, M_src_inv, (w, h),
                                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mouth_mask_orig = cv2.warpAffine(mouth_inner, M_src_inv, (w, h),
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        Mmouth = cv2.moments(mouth_mask_orig)
        if Mmouth["m00"] > 0:
            cx = int(Mmouth["m10"] / Mmouth["m00"])
            cy = int(Mmouth["m01"] / Mmouth["m00"])
            # Restrict driver content to a dilated mouth zone only.
            # driver_to_target_orig contains the entire driver face warped onto
            # the source — including the driver's chin.  seamlessClone's Poisson
            # diffusion bleeds gradients outside the mask, so the driver's chin
            # texture leaks into the source's lower face (double-chin artefact).
            # Replacing everything outside the dilated mouth region with
            # base_result pixels gives the solver zero gradient to spread there.
            _k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            _mouth_zone = cv2.dilate(mouth_mask_orig, _k_dil, iterations=3)
            _driver_mouth_only = driver_to_target_orig.copy()
            _driver_mouth_only[_mouth_zone == 0] = base_result[_mouth_zone == 0]
            result = cv2.seamlessClone(_driver_mouth_only, base_result, mouth_mask_orig, (cx, cy), cv2.MIXED_CLONE)
            print(f"[demo] Mouth clone applied   (open_ratio={_mouth_open_ratio:.3f})")
        else:
            print("[demo] Warning: empty mouth mask — skipping mouth clone")
            result = base_result
    else:
        if _mouth_open_ratio > 0:
            print(f"[demo] Mouth clone skipped   (open_ratio={_mouth_open_ratio:.3f} < {_CLONE_OPEN_THRESH}  driver mouth closed)")
        result = base_result

    result = color_correct(result, source_img, face_mask)

    prefix          = file_prefix or "result"
    result_path     = os.path.join(output_dir, f"{prefix}_result.jpg"   if file_prefix else "result.jpg")
    comparison_path = os.path.join(output_dir, f"{prefix}_results.jpg"  if file_prefix else "comparison.jpg")
    cv2.imwrite(result_path, result)
    if _atlas_mode:
        # No driver image — create a labelled placeholder panel for the comparison
        _ph = np.full_like(source_img, 200)
        cv2.putText(_ph, f"Atlas: {expr_name}", (10, _ph.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 200), 2)
        save_comparison(source_img, _ph, result, comparison_path)
    else:
        save_comparison(source_img, driver_img, result, comparison_path)

    print(f"\nDone!")
    print(f"  Result:     {result_path}")
    print(f"  Comparison: {comparison_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    # ETR uses base_result (before mouth seamlessClone) intentionally.
    # After mouth clone, result landmarks shift to the driver's mouth position,
    # which inflates ETR far above 1.0 even at low scale — making it meaningless.
    # base_result reflects the pure warp displacement and is the correct signal.
    metrics = None
    if run_eval:
        print("\n[eval] Computing metrics...")
        metrics = compute_metrics(
            source_img         = source_img,
            result_img         = base_result,      # before mouth clone — ETR stays meaningful
            face_mask          = face_mask,
            source_lm          = source_lm,        # original space ✓
            target_lm          = target_lm_orig,   # original space ✓
            driver_lm          = drv_lm_aligned,   # aligned space  ✓ (used for LM RMSE only)
            detect_fn          = detect_landmarks,
            driver_blendshapes = driver_bs,
            detect_bs_fn       = detect_blendshapes,
            lm_cfg             = lm_cfg,
        )
        print_metrics(metrics)

        if save_metrics:
            metrics_path = os.path.join(output_dir, "metrics.json")
            save_metrics_json(metrics, metrics_path)
            print(f"  Metrics:    {metrics_path}")

    return result, metrics


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Scale-search defaults ─────────────────────────────────────────────────────
# 5 candidates: from conservative (0.4) to mild amplification (1.2).
# Rationale: observed ETR already averages 1.0–1.5× even at scale=0.7,
# so we keep the upper bound modest to avoid extreme distortion.
SEARCH_SCALES: list = [0.4, 0.6, 0.8, 1.0, 1.2]


def _score_metrics(metrics: dict) -> float:
    """
    Perceptual composite quality score for scale selection (higher = better).

    Designed to approximate "looks best" as a human would judge it:

      blendshape_cosine  40%  — semantic expression accuracy (AU match)
      1 - LPIPS_face     35%  — perceptual naturalness (AlexNet-based)
      color_naturalness  15%  — max(0, 1 – ΔE/30); penalises heavy colour shift
      ssim_face          10%  — structural similarity in face region

    LPIPS distance is in [0, ~1]; inverting gives a "looks-natural" score.
    color_naturalness clamps colour drift > 30 LAB units to 0 (extreme shift).

    Fallback (dlib / no blendshapes, bs == 0):
      perceptual  50% + color_naturalness 30% + ssim_face 20%

    Fallback (LPIPS unavailable):
      falls back to ssim_face in place of perceptual component.
    """
    bs         = float(metrics.get("blendshape_cosine") or 0.0)
    ssim       = float(metrics.get("ssim_face") or metrics.get("ssim_full") or 0.0)
    # Perceptual naturalness: LPIPS face crop (0 = identical, ~1 = very different)
    lpips_raw  = metrics.get("lpips_face")          # None if LPIPS unavailable
    if lpips_raw is not None:
        perceptual = max(0.0, 1.0 - float(lpips_raw))
    else:
        perceptual = ssim                            # graceful fallback

    # Colour naturalness: penalise heavy hue shift (LAB ΔE > 30 → 0)
    cd_raw         = (metrics.get("color_drift") or {}).get("mean")
    color_nat      = max(0.0, 1.0 - float(cd_raw) / 30.0) if cd_raw is not None else 0.5

    if bs == 0.0:
        # dlib mode — no blendshapes
        return perceptual * 0.50 + color_nat * 0.30 + ssim * 0.20
    return bs * 0.40 + perceptual * 0.35 + color_nat * 0.15 + ssim * 0.10


def _run_one_scale(source_path, driver_path, driver_neutral_path,
                   scale, landmark_mode, region_warp=True,
                   jaw_reg=0.20, outer_reg=0.15):
    """
    Run expression transfer at a single scale and return
    (result_img, source_img, driver_img, metrics) WITHOUT writing any files.
    Mirrors the logic of run() but stripped of all I/O.
    """
    lm_cfg = _get_lm_config(landmark_mode)
    detect_landmarks, detect_blendshapes = _build_detector(landmark_mode)

    source_img = _load_image(source_path)
    driver_img = _load_image(driver_path)
    if source_img is None or driver_img is None:
        raise ValueError("Could not load image(s)")

    driver_neutral_img = _load_image(driver_neutral_path) if driver_neutral_path else None

    source_lm         = detect_landmarks(source_img)
    driver_lm         = detect_landmarks(driver_img)
    driver_neutral_lm = detect_landmarks(driver_neutral_img) if driver_neutral_img else None
    driver_bs         = detect_blendshapes(driver_img) if detect_blendshapes else None

    if source_lm is None or driver_lm is None:
        raise ValueError("Landmark detection failed")

    src_aligned, src_lm_aligned, M_src, M_src_inv = align_face(source_img, source_lm, lm_cfg=lm_cfg)
    drv_aligned, drv_lm_aligned, _, _              = align_face(driver_img, driver_lm, lm_cfg=lm_cfg)

    drvN_lm_aligned = None
    if driver_neutral_img is not None and driver_neutral_lm is not None:
        _, drvN_lm_aligned, _, _ = align_face(driver_neutral_img, driver_neutral_lm, lm_cfg=lm_cfg)

    displacement      = compute_displacement(src_lm_aligned, drv_lm_aligned, drvN_lm_aligned,
                                             scale=scale, lm_cfg=lm_cfg)
    if region_warp:
        displacement = apply_region_weights(
            displacement, src_lm_aligned, lm_cfg=lm_cfg,
            jaw_weight=jaw_reg, outer_weight=outer_reg,
        )
    target_lm_aligned = apply_displacement(src_lm_aligned, displacement)
    target_lm_orig    = _transform_landmarks(target_lm_aligned, M_src_inv)

    warped_aligned, face_mask_aligned = warp_face(src_aligned, src_lm_aligned, displacement,
                                                   lm_cfg=lm_cfg)

    try:
        from src.warp import warp_image_by_landmarks
        driver_to_target_aligned = warp_image_by_landmarks(
            drv_aligned, drv_lm_aligned, target_lm_aligned)
        inner_poly         = target_lm_aligned[lm_cfg["inner_lip_idx"]].astype(np.int32)
        mouth_inner        = np.zeros(face_mask_aligned.shape, dtype=np.uint8)
        cv2.fillPoly(mouth_inner, [inner_poly], 255)
        k3                 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mouth_inner_eroded = cv2.erode(mouth_inner, k3, iterations=1)
        face_mask_no_mouth = cv2.bitwise_and(face_mask_aligned,
                                              cv2.bitwise_not(mouth_inner_eroded))
    except Exception:
        driver_to_target_aligned = None
        face_mask_no_mouth       = face_mask_aligned
        mouth_inner              = None

    h, w   = source_img.shape[:2]
    warped = cv2.warpAffine(warped_aligned, M_src_inv, (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    fmask  = cv2.warpAffine(face_mask_no_mouth, M_src_inv, (w, h),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    base_result = blend(source_img, warped, fmask)

    # ── Gate mouth clone on driver open-mouth ratio (same logic as run()) ────
    _CLONE_OPEN_THRESH = 0.08
    _mouth_open_ratio  = 0.0
    if driver_to_target_aligned is not None and mouth_inner is not None:
        _il  = drv_lm_aligned[lm_cfg["inner_lip_idx"]]
        _iod = float(np.linalg.norm(
            drv_lm_aligned[lm_cfg["left_eye"]].mean(0) -
            drv_lm_aligned[lm_cfg["right_eye"]].mean(0)))
        _mouth_open_ratio = float(_il[:, 1].max() - _il[:, 1].min()) / max(_iod, 1.0)

    if driver_to_target_aligned is not None and mouth_inner is not None \
            and _mouth_open_ratio >= _CLONE_OPEN_THRESH:
        dtarget_orig    = cv2.warpAffine(driver_to_target_aligned, M_src_inv, (w, h),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mouth_mask_orig = cv2.warpAffine(mouth_inner, M_src_inv, (w, h),
                                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        Mm = cv2.moments(mouth_mask_orig)
        if Mm["m00"] > 0:
            cx, cy = int(Mm["m10"] / Mm["m00"]), int(Mm["m01"] / Mm["m00"])
            _k_dil2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            _mouth_zone2 = cv2.dilate(mouth_mask_orig, _k_dil2, iterations=3)
            _dtarget_mouth = dtarget_orig.copy()
            _dtarget_mouth[_mouth_zone2 == 0] = base_result[_mouth_zone2 == 0]
            result = cv2.seamlessClone(_dtarget_mouth, base_result, mouth_mask_orig,
                                        (cx, cy), cv2.MIXED_CLONE)
        else:
            result = base_result
    else:
        result = base_result

    result = color_correct(result, source_img, fmask)

    metrics = compute_metrics(
        source_img         = source_img,
        result_img         = base_result,     # before mouth clone — ETR stays meaningful
        face_mask          = fmask,
        source_lm          = source_lm,
        target_lm          = target_lm_orig,
        driver_lm          = drv_lm_aligned,
        detect_fn          = detect_landmarks,
        driver_blendshapes = driver_bs,
        detect_bs_fn       = detect_blendshapes,
        lm_cfg             = lm_cfg,
    )
    return result, source_img, driver_img, metrics


def run_scale_search(source_path, driver_dir, driver_neutral_path=None,
                     scales=None, output_dir="output", landmark_mode="mp",
                     region_warp=True, jaw_reg=0.20, outer_reg=0.15):
    """
    Scale-search batch mode.

    For every driver image (handles both flat dirs and expression sub-folders):
      1. Run expression transfer at each scale in `scales`
      2. Score with _score_metrics() (blendshape × 0.5 + ssim × 0.3 + etr × 0.2)
      3. Save ONLY the best-scale result; filename encodes the winning scale
      4. Write per-driver JSON: all scale scores + winning scale + metrics
    Per expression folder: aggregated JSON of best-scale metrics.
    Global:               cross-expression summary JSON.

    Output layout:
      {output_dir}/scale_search/
        {expr}/
          {stem}_scale{X.XX}_result.jpg       ← best only
          {stem}_scale{X.XX}_results.jpg      ← best comparison only
          {stem}_scale_search.json            ← all scales + winner for this driver
          metrics_{expr}_scale_search.json    ← aggregated (best per driver)
        scale_search_summary.json             ← cross-expression summary
    """
    if scales is None:
        scales = SEARCH_SCALES

    root_out = os.path.join(output_dir, "scale_search")
    os.makedirs(root_out, exist_ok=True)

    # ── Discover groups (expression sub-folders or flat) ─────────────────────
    expr_entries = sorted(
        [e for e in os.scandir(driver_dir) if e.is_dir()],
        key=lambda e: e.name,
    )
    if expr_entries:
        groups = [
            (e.name,
             sorted([f for f in os.scandir(e.path)
                     if os.path.splitext(f.name)[1].lower() in _IMG_EXTS],
                    key=lambda f: f.name),
             os.path.join(root_out, e.name))
            for e in expr_entries
        ]
    else:
        flat_imgs = sorted(
            [f for f in os.scandir(driver_dir)
             if os.path.splitext(f.name)[1].lower() in _IMG_EXTS],
            key=lambda f: f.name,
        )
        groups = [("flat", flat_imgs, root_out)]

    print(f"\n[scale-search] Source     : {source_path}")
    print(f"[scale-search] Driver dir : {driver_dir}")
    print(f"[scale-search] Scales     : {scales}  ({len(scales)} candidates/image)")
    print(f"[scale-search] Groups     : {[g[0] for g in groups]}")

    all_group_summaries = {}

    for expr_name, img_files, expr_output in groups:
        os.makedirs(expr_output, exist_ok=True)
        print(f"\n{'='*62}")
        print(f"  Expression : {expr_name}   ({len(img_files)} images × {len(scales)} scales)")
        print(f"{'='*62}")

        group_best_metrics = []
        failed             = []

        for idx, img_entry in enumerate(img_files, 1):
            driver_stem = os.path.splitext(img_entry.name)[0]
            print(f"\n  [{idx:>3}/{len(img_files)}] {img_entry.name}")

            scale_results = []   # (scale, score, result_img, src_img, drv_img, metrics)

            for sc in scales:
                print(f"    scale={sc:.2f} … ", end="", flush=True)
                try:
                    result, src_img, drv_img, metrics = _run_one_scale(
                        source_path, img_entry.path, driver_neutral_path,
                        sc, landmark_mode,
                        region_warp=region_warp, jaw_reg=jaw_reg, outer_reg=outer_reg,
                    )
                    score = _score_metrics(metrics)
                    scale_results.append((sc, score, result, src_img, drv_img, metrics))
                    print(f"score={score:.4f}  "
                          f"(bs={metrics.get('blendshape_cosine') or 0:.3f}, "
                          f"ssim={metrics.get('ssim_face') or 0:.3f}, "
                          f"etr={metrics.get('etr') or 0:.3f})")
                except Exception as exc:
                    print(f"FAILED ({exc})")

            if not scale_results:
                print(f"  [!] All scales failed — skipping {img_entry.name}")
                failed.append(img_entry.name)
                continue

            # ── Select winner from this run ───────────────────────────────────
            best_sc, best_score, best_result, best_src, best_drv, best_metrics = max(
                scale_results, key=lambda x: x[1]
            )

            # ── Load existing JSON (if any) — compare & optionally overwrite ──
            per_driver_path  = os.path.join(expr_output,
                                             f"{driver_stem}_scale_search.json")
            old_best_score   = -1.0
            old_best_sc      = None
            old_scales_tried = []
            old_best_metrics = None

            if os.path.exists(per_driver_path):
                try:
                    import json as _json
                    with open(per_driver_path) as _f:
                        _old = _json.load(_f)
                    old_best_score   = float(_old.get("best_score", -1.0))
                    old_best_sc      = _old.get("best_scale")
                    old_scales_tried = _old.get("scales_tried", [])
                    old_best_metrics = _old.get("best_metrics")
                except Exception:
                    pass   # corrupted JSON — treat as fresh

            # Merge scale records: new results overwrite same-scale old entries
            new_scale_values = {sc for sc, *_ in scale_results}
            merged_scales = (
                [s for s in old_scales_tried if s.get("scale") not in new_scale_values]
                + [{"scale": sc, "score": round(_score_metrics(m), 6), **m}
                   for sc, _, _, _, _, m in scale_results]
            )
            merged_scales.sort(key=lambda x: x.get("scale", 0))

            # ── Decide whether to save images ─────────────────────────────────
            if best_score > old_best_score:
                # New result is better → write images (remove old if scale changed)
                if old_best_sc is not None:
                    old_tag = f"scale{old_best_sc:.2f}"
                    if old_tag != f"scale{best_sc:.2f}":
                        for suf in ("_result.jpg", "_results.jpg"):
                            old_img = os.path.join(expr_output,
                                                    f"{driver_stem}_{old_tag}{suf}")
                            if os.path.exists(old_img):
                                os.remove(old_img)

                scale_tag       = f"scale{best_sc:.2f}"
                result_path     = os.path.join(expr_output,
                                                f"{driver_stem}_{scale_tag}_result.jpg")
                comparison_path = os.path.join(expr_output,
                                                f"{driver_stem}_{scale_tag}_results.jpg")
                cv2.imwrite(result_path, best_result)
                save_comparison(best_src, best_drv, best_result, comparison_path)
                verdict = f"NEW BEST  scale={best_sc:.2f}  score={best_score:.4f}"
                if old_best_score >= 0:
                    verdict += f"  (was {old_best_score:.4f})"
                print(f"    ★ {verdict}  → saved")
            else:
                # Old result is still better → keep old images, just log
                best_sc      = old_best_sc if old_best_sc is not None else best_sc
                best_score   = old_best_score
                best_metrics = old_best_metrics if old_best_metrics else best_metrics
                new_best_in_run = max(scale_results, key=lambda x: x[1])[1]
                print(f"    = KEPT OLD  scale={best_sc:.2f}  score={best_score:.4f}"
                      f"  (new run best={new_best_in_run:.4f} — no improvement)")

            # ── Per-driver JSON: merged scales + current winner ───────────────
            driver_report = {
                "driver"      : img_entry.name,
                "best_scale"  : best_sc,
                "best_score"  : round(best_score, 6),
                "scales_tried": merged_scales,
                "best_metrics": best_metrics,
            }
            save_metrics_json(driver_report, per_driver_path)

            best_metrics["driver"]     = img_entry.name
            best_metrics["best_scale"] = best_sc
            group_best_metrics.append(best_metrics)

        # ── Aggregated (best-scale) metrics for this expression ───────────────
        if group_best_metrics:
            agg           = _aggregate_metrics(group_best_metrics)
            agg["failed"] = failed
            agg_path      = os.path.join(expr_output,
                                          f"metrics_{expr_name}_scale_search.json")
            save_metrics_json(agg, agg_path)
            print(f"\n  Aggregated best-scale metrics → {agg_path}")
            all_group_summaries[expr_name] = agg["summary"]
        elif failed:
            print(f"  All {len(failed)} image(s) failed — no aggregated metrics written.")

    # ── Cross-expression summary ──────────────────────────────────────────────
    if all_group_summaries:
        summary_path = os.path.join(root_out, "scale_search_summary.json")
        save_metrics_json(all_group_summaries, summary_path)
        print(f"\n  Cross-expression summary → {summary_path}")

    print(f"\n{'='*62}")
    print(f"Scale search complete.  Results in: {root_out}")
    print(f"{'='*62}")


def _aggregate_metrics(metrics_list: list) -> dict:
    """
    Aggregate individual metric dicts into:
      - summary: mean ± std for every numeric field (dot-notation keys)
      - per_driver: original per-image metric dicts

    NaN / None values are excluded from stats but preserved in per_driver.
    """
    n = len(metrics_list)
    if n == 0:
        return {"n": 0, "summary": {}, "per_driver": []}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from _flatten(v, key)
            elif v is not None and isinstance(v, (int, float)) and v == v:  # skip NaN
                yield key, v

    flat_records = [dict(_flatten(m)) for m in metrics_list]
    all_keys     = sorted({k for r in flat_records for k in r})

    summary = {}
    for k in all_keys:
        vals = [r[k] for r in flat_records if k in r]
        if vals:
            arr = np.array(vals, dtype=float)
            summary[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}

    return {"n": n, "summary": summary, "per_driver": metrics_list}


def run_batch(source_path, driver_dir, driver_neutral_path=None, scale=0.7,
              output_dir="output", run_eval=True, landmark_mode="mp",
              region_warp=True, jaw_reg=0.20, outer_reg=0.15):
    """
    Batch mode: driver_dir contains one sub-folder per expression.
    Each sub-folder holds N driver images.

    Output layout:
      output/
        {expr}/
          {driver_stem}_results.jpg   (side-by-side comparison)
          {driver_stem}_result.jpg    (result only)
          metrics_{expr}.json         (aggregated across all drivers)
    """
    expr_dirs = sorted(
        [e for e in os.scandir(driver_dir) if e.is_dir()],
        key=lambda e: e.name,
    )
    if not expr_dirs:
        print(f"[batch] No sub-folders found in {driver_dir}")
        return

    print(f"[batch] Source  : {source_path}")
    print(f"[batch] Driver  : {driver_dir}")
    print(f"[batch] Expressions ({len(expr_dirs)}): {[e.name for e in expr_dirs]}")

    for expr_entry in expr_dirs:
        expr_name   = expr_entry.name
        expr_output = os.path.join(output_dir, expr_name)
        os.makedirs(expr_output, exist_ok=True)

        img_files = sorted(
            [f for f in os.scandir(expr_entry.path)
             if os.path.splitext(f.name)[1].lower() in _IMG_EXTS],
            key=lambda f: f.name,
        )
        print(f"\n{'='*62}")
        print(f"  Expression : {expr_name}   ({len(img_files)} images)")
        print(f"{'='*62}")

        all_metrics = []
        failed      = []

        for idx, img_entry in enumerate(img_files, 1):
            driver_stem = os.path.splitext(img_entry.name)[0]
            print(f"\n  [{idx:>3}/{len(img_files)}] {img_entry.name}")
            try:
                _, metrics = run(
                    source_path         = source_path,
                    driver_path         = img_entry.path,
                    driver_neutral_path = driver_neutral_path,
                    scale               = scale,
                    output_dir          = expr_output,
                    run_eval            = run_eval,
                    save_metrics        = False,
                    landmark_mode       = landmark_mode,
                    file_prefix         = driver_stem,
                    region_warp         = region_warp,
                    jaw_reg             = jaw_reg,
                    outer_reg           = outer_reg,
                )
                if metrics is not None:
                    metrics["driver"] = img_entry.name
                    all_metrics.append(metrics)
            except Exception as exc:
                print(f"  [!] Skipped ({exc})")
                failed.append(img_entry.name)

        # ── Aggregated metrics for this expression ────────────────────────────
        if run_eval:
            agg      = _aggregate_metrics(all_metrics)
            agg["failed"] = failed
            agg_path = os.path.join(expr_output, f"metrics_{expr_name}.json")
            save_metrics_json(agg, agg_path)
            print(f"\n  Aggregated metrics ({len(all_metrics)} drivers) → {agg_path}")
            if failed:
                print(f"  Skipped {len(failed)} image(s): {failed}")

    print(f"\n{'='*62}")
    print(f"Batch complete.  Results in: {output_dir}")
    print(f"{'='*62}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Expression Transfer Demo")
    parser.add_argument("--source",         required=True,  help="Source face image (neutral)")
    parser.add_argument("--driver",         default=None,
                        help="Driver image OR directory of drivers.  "
                             "Directory → batch mode (or scale-search with --scale-search).  "
                             "Omit when using --atlas.")
    parser.add_argument("--driver-neutral", default=None,   help="Driver neutral baseline image")
    # ── Atlas mode (pre-computed expression data) ─────────────────────────────
    parser.add_argument("--atlas",          default=None,
                        help="Pre-computed expression atlas .npz built by build_expression_atlas.py")
    parser.add_argument("--expr",           default=None,
                        help="Expression name to load from --atlas (e.g. angry, happy)")
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--scale",          type=float, default=0.7,
                        help="Expression scale factor for single/batch mode (default: 0.7)")
    parser.add_argument("--landmark-mode",  default="mp", choices=["mp", "dlib"],
                        help="Landmark backend: 'mp' = MediaPipe 478-pt (default), "
                             "'dlib' = dlib 68-pt (lighter, legacy)")
    parser.add_argument("--output",         default="output",  help="Output directory")
    parser.add_argument("--no-eval",        action="store_true", help="Skip evaluation metrics")
    parser.add_argument("--save-metrics",   action="store_true", help="Save metrics.json (single mode)")
    # ── Scale-search arguments ────────────────────────────────────────────────
    parser.add_argument("--scale-search",   action="store_true",
                        help="Scale-search mode (requires --driver to be a directory): "
                             "try each scale in --scales, keep only the best per image.")
    parser.add_argument("--scales",         type=float, nargs="+",
                        default=SEARCH_SCALES,
                        help=f"Scale candidates for --scale-search "
                             f"(default: {SEARCH_SCALES})")
    # ── Region-warp arguments ─────────────────────────────────────────────────
    parser.add_argument("--no-region-warp", action="store_true",
                        help="Disable per-region displacement regularisation (on by default). "
                             "When off, all landmarks receive identical scale weighting.")
    parser.add_argument("--jaw-reg",  type=float, default=0.20,
                        help="Jaw/chin regularisation weight: 0=frozen, 1=full expression "
                             "(default: 0.20)")
    parser.add_argument("--outer-reg", type=float, default=0.15,
                        help="Outer face-oval regularisation weight (default: 0.15)")
    args = parser.parse_args()

    # Validate: need either --driver or (--atlas + --expr)
    if args.driver is None and not (args.atlas and args.expr):
        parser.error("Provide either --driver <image/dir> or both --atlas <file> and --expr <name>.")

    _region_warp = not args.no_region_warp
    _jaw_reg     = args.jaw_reg
    _outer_reg   = args.outer_reg

    if args.driver and os.path.isdir(args.driver):
        if args.scale_search:
            run_scale_search(
                source_path         = args.source,
                driver_dir          = args.driver,
                driver_neutral_path = args.driver_neutral,
                scales              = args.scales,
                output_dir          = args.output,
                landmark_mode       = args.landmark_mode,
                region_warp         = _region_warp,
                jaw_reg             = _jaw_reg,
                outer_reg           = _outer_reg,
            )
        else:
            run_batch(
                source_path         = args.source,
                driver_dir          = args.driver,
                driver_neutral_path = args.driver_neutral,
                scale               = args.scale,
                output_dir          = args.output,
                run_eval            = not args.no_eval,
                landmark_mode       = args.landmark_mode,
                region_warp         = _region_warp,
                jaw_reg             = _jaw_reg,
                outer_reg           = _outer_reg,
            )
    else:
        if args.scale_search:
            print("[!] --scale-search requires --driver to be a directory. Ignored.")
        run(args.source, args.driver, args.driver_neutral, args.scale, args.output,
            run_eval=not args.no_eval, save_metrics=args.save_metrics,
            landmark_mode=args.landmark_mode,
            atlas_path=args.atlas, expr_name=args.expr,
            region_warp=_region_warp, jaw_reg=_jaw_reg, outer_reg=_outer_reg)