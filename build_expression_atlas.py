"""
build_expression_atlas.py — Pre-compute per-expression landmark/blendshape statistics.

Usage:
  python build_expression_atlas.py [--sample-dir data/sample_images]
                                   [--output data/expression_atlas.npz]
                                   [--outlier-k 2.5]

For each expression sub-folder (e.g. angry/, happy/, ...):
  1. Detect MediaPipe 478-pt landmarks + 52 blendshapes in a single pass per image
  2. Rotate-align landmarks (remove head tilt) then normalise to canonical space:
       IOD = 200 px,  eye midpoint = (256, 256)
  3. Remove outliers: faces whose flattened-landmark Euclidean distance from the
     group centroid exceeds  mean + k * std  (default k=2.5)
  4. Compute mean canonical landmarks and mean blendshapes from the inlier set

Output:
  data/expression_atlas.npz       — arrays keyed <expr>_landmarks / <expr>_blendshapes
  data/expression_atlas_meta.json — per-expression statistics and file lists

The atlas is consumed by demo.py:
  python demo.py --source face.jpg --atlas data/expression_atlas.npz --expr angry
"""
from __future__ import annotations

import argparse
import json
import os
import sys

print("Loading libraries...", flush=True)

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.landmark_config import get_config
from src.align import align_face
from src.landmark_mp import _run_detector  # single-pass: landmarks + blendshapes

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Canonical space constants (resolution-independent representation)
_CANON_IOD    = 200.0
_CANON_CENTER = np.array([256.0, 256.0], dtype=np.float32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] != 3:
        return None

    # EXIF rotation correction — phone JPEGs are often stored rotated.
    # Without this, a 90-rotated driver image produces a completely wrong
    # canonical landmark set that poisons the atlas mean.
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
            pass  # Pillow absent — proceed without EXIF correction
        except Exception:
            pass

    return img


def _normalize_to_canonical(lm_aligned: np.ndarray, lm_cfg: dict) -> np.ndarray | None:
    """
    Map rotation-aligned landmarks into canonical space.

    Canonical space: inter-ocular distance = 200 px, eye midpoint = (256, 256).
    This representation is resolution-independent and consistent across subjects,
    so landmark arrays from different images can be meaningfully averaged.
    """
    le_ctr = lm_aligned[lm_cfg["left_eye"]].mean(0)
    re_ctr = lm_aligned[lm_cfg["right_eye"]].mean(0)
    iod = float(np.linalg.norm(re_ctr - le_ctr))
    if iod < 1.0:
        return None
    eye_mid = (le_ctr + re_ctr) * 0.5
    lm_unit = (lm_aligned - eye_mid) / iod
    return (lm_unit * _CANON_IOD + _CANON_CENTER).astype(np.float32)


def _detect_one(img_path: str, lm_cfg: dict) -> tuple:
    """
    Single-pass detection of landmarks + blendshapes for one image.

    Returns (canonical_lm, blendshapes, error_str).
    canonical_lm  : (478, 2) float32 in canonical space, or None on failure
    blendshapes   : (52,) float32, or None if not available
    error_str     : description of failure, or None on success
    """
    img = _load_image(img_path)
    if img is None:
        return None, None, "failed to load"

    lm, bs = _run_detector(img, blendshapes=True)
    if lm is None:
        return None, None, "no face detected"

    _, lm_aligned, _, _ = align_face(img, lm, lm_cfg=lm_cfg)
    lm_canon = _normalize_to_canonical(lm_aligned, lm_cfg)
    if lm_canon is None:
        return None, None, "IOD near zero — degenerate landmark set"

    return lm_canon, bs, None


def _filter_outliers(lm_list: list, k: float = 2.5) -> tuple:
    """
    Identify outlier faces by Euclidean distance in flattened canonical landmark space.

    A face is an outlier when its distance from the group centroid exceeds
    mean + k * std. With k=2.5, roughly 1% of a Gaussian population would
    be removed under the null; in practice this catches badly detected or
    atypical expression faces.

    Returns:
        inlier_mask : (N,) bool — True = keep
        dists       : (N,) float — distance from centroid for each face
    """
    stack    = np.stack(lm_list, axis=0)       # (N, 478, 2)
    flat     = stack.reshape(len(lm_list), -1) # (N, 956)
    centroid = flat.mean(0)
    dists    = np.linalg.norm(flat - centroid, axis=1)

    if len(dists) < 3:
        # Too few samples to compute a meaningful threshold — keep all
        return np.ones(len(dists), dtype=bool), dists

    threshold = dists.mean() + k * dists.std()
    return dists <= threshold, dists


# ── Main ─────────────────────────────────────────────────────────────────────

def build_atlas(sample_dir: str, output_path: str, outlier_k: float = 2.5) -> None:
    lm_cfg = get_config("mp")

    expr_dirs = sorted(
        [e for e in os.scandir(sample_dir) if e.is_dir()],
        key=lambda e: e.name,
    )
    if not expr_dirs:
        print(f"[!] No expression sub-folders found in: {sample_dir}")
        sys.exit(1)

    print(f"Found {len(expr_dirs)} expression folders: {[e.name for e in expr_dirs]}\n")

    arrays: dict[str, np.ndarray] = {}
    meta:   dict[str, dict]       = {}

    for expr_entry in expr_dirs:
        expr = expr_entry.name
        img_files = sorted(
            [f for f in os.scandir(expr_entry.path)
             if os.path.splitext(f.name)[1].lower() in _IMG_EXTS],
            key=lambda f: f.name,
        )
        print(f"\n{'='*62}")
        print(f"  Expression : {expr}   ({len(img_files)} images)")
        print(f"{'='*62}")

        lm_list: list[np.ndarray]        = []
        bs_list: list[np.ndarray | None] = []
        names:   list[str]               = []
        failed:  list[str]               = []

        for i, img_entry in enumerate(img_files, 1):
            print(f"  [{i:>4}/{len(img_files)}] {img_entry.name:<35}", end="", flush=True)
            lm_canon, bs, err = _detect_one(img_entry.path, lm_cfg)
            if err:
                print(f"SKIP  ({err})")
                failed.append(img_entry.name)
                continue
            lm_list.append(lm_canon)
            bs_list.append(bs)
            names.append(img_entry.name)
            print("OK")

        n_detected = len(lm_list)
        if n_detected == 0:
            print(f"\n  [!] No valid faces — skipping expression '{expr}'")
            continue

        # ── Outlier filtering ─────────────────────────────────────────────────
        inlier_mask, dists = _filter_outliers(lm_list, k=outlier_k)
        n_inliers  = int(inlier_mask.sum())
        n_outliers = n_detected - n_inliers

        inlier_names  = [names[i] for i in range(n_detected) if inlier_mask[i]]
        outlier_names = [names[i] for i in range(n_detected) if not inlier_mask[i]]

        print(f"\n  Inlier faces : {n_inliers}/{n_detected}"
              f"  (k={outlier_k}, threshold={dists.mean() + outlier_k * dists.std():.1f})")
        if outlier_names:
            print(f"  Removed as outliers ({n_outliers}):")
            for fn in outlier_names:
                idx = names.index(fn)
                print(f"    {fn}  (dist={dists[idx]:.2f})")

        # ── Guard: zero inliers can happen if outlier_k is very small ────────
        if n_inliers == 0:
            print(f"\n  [!] All {n_detected} face(s) removed as outliers — skipping '{expr}'.")
            print(f"      Tip: raise --outlier-k (current k={outlier_k}) to keep more faces.")
            continue

        # ── Mean canonical landmarks from inliers ─────────────────────────────
        inlier_lm_stack = np.stack(
            [lm_list[i] for i in range(n_detected) if inlier_mask[i]], axis=0
        )
        mean_lm = inlier_lm_stack.mean(0).astype(np.float32)

        # ── Mean blendshapes from inliers that have valid blendshapes ─────────
        inlier_bs_arrays = [
            bs_list[i] for i in range(n_detected)
            if inlier_mask[i] and bs_list[i] is not None
        ]
        if inlier_bs_arrays:
            mean_bs = np.stack(inlier_bs_arrays, axis=0).mean(0).astype(np.float32)
        else:
            mean_bs = None

        # ── Std of canonical landmarks (useful for debugging expression spread) ─
        if n_inliers > 1:
            std_lm = inlier_lm_stack.std(0).astype(np.float32)
        else:
            std_lm = np.zeros_like(mean_lm)

        arrays[f"{expr}_landmarks"] = mean_lm
        arrays[f"{expr}_landmarks_std"] = std_lm
        if mean_bs is not None:
            arrays[f"{expr}_blendshapes"] = mean_bs

        meta[expr] = {
            "n_total_images" : len(img_files),
            "n_detected"     : n_detected,
            "n_inliers"      : n_inliers,
            "n_outliers"     : n_outliers,
            "n_failed"       : len(failed),
            "inlier_files"   : inlier_names,
            "outlier_files"  : outlier_names,
            "failed_files"   : failed,
            "dist_mean"      : float(dists.mean()),
            "dist_std"       : float(dists.std()),
            "dist_threshold" : float(dists.mean() + outlier_k * dists.std()),
            "has_blendshapes": mean_bs is not None,
            "canon_iod"      : _CANON_IOD,
            "canon_center"   : list(_CANON_CENTER),
        }

    if not arrays:
        print("\n[!] No expressions processed — atlas not saved.")
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    np.savez(output_path, **arrays)

    meta_path = os.path.splitext(output_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*62}")
    print(f"Atlas saved   : {os.path.abspath(output_path)}")
    print(f"Metadata saved: {os.path.abspath(meta_path)}")
    print(f"\nPer-expression summary (k={outlier_k}):")
    max_name = max(len(e) for e in meta)
    for expr, m in meta.items():
        bs_tag = "  [+blendshapes]" if m["has_blendshapes"] else ""
        print(f"  {expr:<{max_name}} : {m['n_inliers']:>3}/{m['n_detected']:>3} inliers"
              f"  ({m['n_outliers']:>2} removed,  {m['n_failed']:>2} failed){bs_tag}")
    print(f"\nUsage:")
    print(f"  python demo.py --source <face.jpg> "
          f"--atlas {output_path} --expr <expression>")
    print(f"{'='*62}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build per-expression landmark/blendshape atlas from sample_images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample-dir", default="data/sample_images",
        help="Root dir with one expression sub-folder each (default: data/sample_images)",
    )
    parser.add_argument(
        "--output", default="data/expression_atlas.npz",
        help="Output .npz file path (default: data/expression_atlas.npz)",
    )
    parser.add_argument(
        "--outlier-k", type=float, default=2.5,
        help="Z-score multiplier for outlier rejection (default: 2.5; lower → stricter)",
    )
    args = parser.parse_args()

    build_atlas(
        sample_dir  = args.sample_dir,
        output_path = args.output,
        outlier_k   = args.outlier_k,
    )
