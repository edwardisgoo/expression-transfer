"""
demo.py — End-to-end expression transfer demo
Usage: python demo.py --source <path> --driver <path> [--driver-neutral <path>] [--scale 0.9]
"""

import argparse
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from landmark import detect_landmarks
from expression import compute_displacement
from warp import warp_face
from blend import blend, save_comparison


def run(source_path, driver_path, driver_neutral_path=None, scale=1.0, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    print("[1/4] Loading images...")
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

    print("[2/4] Detecting landmarks...")
    source_lm = detect_landmarks(source_img)
    driver_lm = detect_landmarks(driver_img)
    driver_neutral_lm = detect_landmarks(driver_neutral_img) if driver_neutral_img is not None else None

    if any(lm is None for lm in [source_lm, driver_lm]):
        print("Error: landmark detection failed on one or more images.")
        sys.exit(1)

    print("[3/4] Computing displacement & warping...")
    from expression import compute_displacement, apply_displacement
    displacement = compute_displacement(source_lm, driver_lm, driver_neutral_lm, scale=scale)
    warped_img, face_mask = warp_face(source_img, source_lm, displacement)

    print("[4/4] Blending...")
    result = blend(source_img, warped_img, face_mask)

    # Save outputs
    result_path = os.path.join(output_dir, "result.jpg")
    comparison_path = os.path.join(output_dir, "comparison.jpg")
    cv2.imwrite(result_path, result)
    save_comparison(source_img, driver_img, result, comparison_path)

    print(f"\nDone!")
    print(f"  Result:     {result_path}")
    print(f"  Comparison: {comparison_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Expression Transfer Demo")
    parser.add_argument("--source", required=True, help="Source face image (neutral)")
    parser.add_argument("--driver", required=True, help="Driver face image (expressive)")
    parser.add_argument("--driver-neutral", default=None, help="Driver neutral baseline image")
    parser.add_argument("--scale", type=float, default=0.7, help="Expression scale factor (0.5–1.0)")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    run(args.source, args.driver, args.driver_neutral, args.scale, args.output)
