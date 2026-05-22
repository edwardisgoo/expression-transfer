"""
demo.py — End-to-end expression transfer demo
Usage: python demo.py --source <path> --driver <path> [--driver-neutral <path>] [--scale 0.9]
"""

import argparse
import cv2
import numpy as np
import os
import sys

# Make repo root importable, so we can import src.* as a package
sys.path.insert(0, os.path.dirname(__file__))

from src.landmark import detect_landmarks
from src.expression import compute_displacement, apply_displacement
from src.warp import warp_face
from src.blend import blend, save_comparison
from src.align import align_face
from src.evaluate import compute_metrics, print_metrics, save_metrics_json


def _transform_landmarks(lm: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2×3 affine matrix M to a (N, 2) landmark array."""
    ones = np.ones((lm.shape[0], 1), dtype=np.float32)
    pts  = np.hstack([lm.astype(np.float32), ones])
    return (pts @ M.T).astype(np.float32)


def run(source_path, driver_path, driver_neutral_path=None, scale=1.0, output_dir="output",
        run_eval=True, save_metrics=False):
    os.makedirs(output_dir, exist_ok=True)

    print("[1/5] Loading images...")
    source_img = cv2.imread(source_path)
    driver_img = cv2.imread(driver_path)
    if source_img is None or driver_img is None:
        print("Error: could not load one or both images.")
        sys.exit(1)

    if driver_neutral_path:
        driver_neutral_img = cv2.imread(driver_neutral_path)
    else:
        print("[!] No driver neutral provided — using direct warp mode.")
        print("    Source landmarks will be warped directly toward driver face geometry.")
        driver_neutral_img = None

    print("[2/5] Detecting landmarks...")
    source_lm = detect_landmarks(source_img)
    driver_lm = detect_landmarks(driver_img)
    driver_neutral_lm = detect_landmarks(driver_neutral_img) if driver_neutral_img is not None else None

    if any(lm is None for lm in [source_lm, driver_lm]):
        print("Error: landmark detection failed on one or more images.")
        sys.exit(1)

    print("[3/5] Face alignment (stabilization)...")
    src_aligned, src_lm_aligned, M_src, M_src_inv = align_face(source_img, source_lm)
    drv_aligned, drv_lm_aligned, _, _ = align_face(driver_img, driver_lm)
    if driver_neutral_img is not None and driver_neutral_lm is not None:
        drvN_aligned, drvN_lm_aligned, _, _ = align_face(driver_neutral_img, driver_neutral_lm)
    else:
        drvN_aligned = None
        drvN_lm_aligned = None

    print("[4/5] Computing displacement & warping in aligned space...")
    displacement = compute_displacement(src_lm_aligned, drv_lm_aligned, drvN_lm_aligned, scale=scale)

    # target_lm in aligned space
    target_lm_aligned = apply_displacement(src_lm_aligned, displacement)

    # For ETR, compute the target landmarks back in original space
    target_lm_orig = _transform_landmarks(target_lm_aligned, M_src_inv)

    warped_aligned, face_mask_aligned = warp_face(src_aligned, src_lm_aligned, displacement)

    # Region-aware compositing — mouth handled separately (avoid lip fill)
    try:
        from src.warp import warp_image_by_landmarks  # type: ignore
        driver_to_target_aligned = warp_image_by_landmarks(drv_aligned, drv_lm_aligned, target_lm_aligned)

        # Mouth masks (aligned space)
        inner_poly = target_lm_aligned[60:68].astype(np.int32)
        mouth_inner = np.zeros(face_mask_aligned.shape, dtype=np.uint8)
        cv2.fillPoly(mouth_inner, [inner_poly], 255)
        # Slightly erode inner so we don't overlap lip rim; keep outer in global mask
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mouth_inner_eroded = cv2.erode(mouth_inner, k3, iterations=1)

        # Carve ONLY the inner mouth from the global face mask; keep outer lips warped
        face_mask_aligned_nouth = face_mask_aligned.copy()
        inv_inner = cv2.bitwise_not(mouth_inner_eroded)
        face_mask_aligned_nouth = cv2.bitwise_and(face_mask_aligned_nouth, inv_inner)

    except Exception as e:
        print(f"[demo] Mouth preprocessing error (non-fatal): {e}")
        driver_to_target_aligned = None
        face_mask_aligned_nouth = face_mask_aligned
        mouth_inner = None

    h, w = source_img.shape[:2]
    warped_img = cv2.warpAffine(warped_aligned, M_src_inv, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # Use face mask with mouth carved out for global blend
    face_mask  = cv2.warpAffine(face_mask_aligned_nouth, M_src_inv, (w, h),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    print("[5/5] Blending...")
    base_result = blend(source_img, warped_img, face_mask)

    # Second pass: paste driver mouth in original space
    if driver_to_target_aligned is not None and mouth_inner is not None:
        driver_to_target_orig = cv2.warpAffine(driver_to_target_aligned, M_src_inv, (w, h),
                                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mouth_mask_orig = cv2.warpAffine(mouth_inner, M_src_inv, (w, h),
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        Mmouth = cv2.moments(mouth_mask_orig)
        if Mmouth["m00"] > 0:
            cx = int(Mmouth["m10"] / Mmouth["m00"])
            cy = int(Mmouth["m01"] / Mmouth["m00"])
            result = cv2.seamlessClone(driver_to_target_orig, base_result, mouth_mask_orig, (cx, cy), cv2.MIXED_CLONE)
        else:
            print("[demo] Warning: empty mouth mask in orig space; skipping mouth clone")
            result = base_result
    else:
        result = base_result

    result_path     = os.path.join(output_dir, "result.jpg")
    comparison_path = os.path.join(output_dir, "comparison.jpg")
    cv2.imwrite(result_path, result)
    save_comparison(source_img, driver_img, result, comparison_path)

    print(f"\nDone!")
    print(f"  Result:     {result_path}")
    print(f"  Comparison: {comparison_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    if run_eval:
        print("\n[eval] Computing metrics...")
        metrics = compute_metrics(
            source_img = source_img,
            result_img = result,
            face_mask  = face_mask,
            source_lm  = source_lm,        # original space ✓
            target_lm  = target_lm_orig,   # original space ✓
            driver_lm  = drv_lm_aligned,   # aligned space  ✓ (used for LM RMSE only)
            detect_fn  = detect_landmarks,
        )
        print_metrics(metrics)

        if save_metrics:
            metrics_path = os.path.join(output_dir, "metrics.json")
            save_metrics_json(metrics, metrics_path)
            print(f"  Metrics:    {metrics_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Expression Transfer Demo")
    parser.add_argument("--source",         required=True,      help="Source face image (neutral)")
    parser.add_argument("--driver",         required=True,      help="Driver face image (expressive)")
    parser.add_argument("--driver-neutral", default=None,       help="Driver neutral baseline image")
    parser.add_argument("--scale",          type=float, default=0.7,
                        help="Expression scale factor (0.5–1.0)")
    parser.add_argument("--output",         default="output",   help="Output directory")
    parser.add_argument("--no-eval",        action="store_true", help="Skip evaluation metrics")
    parser.add_argument("--save-metrics",   action="store_true", help="Save metrics.json to output dir")
    args = parser.parse_args()

    run(args.source, args.driver, args.driver_neutral, args.scale, args.output,
        run_eval=not args.no_eval, save_metrics=args.save_metrics)