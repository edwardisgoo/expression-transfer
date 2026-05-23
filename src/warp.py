"""
Phase 3: Face Warping via Delaunay Triangulation
Owner: Member B

Warps the source face geometry using per-triangle affine transforms
to match the displaced landmark positions.
"""

import numpy as np
import cv2
from scipy.spatial import Delaunay


def _get_face_rect(image: np.ndarray) -> tuple:
    """Returns bounding rect covering the full image for Delaunay subdivision."""
    h, w = image.shape[:2]
    return (0, 0, w, h)


def _apply_affine_to_triangle(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray
):
    """Warp one triangle from src_img into dst_img."""
    # Skip degenerate (zero/near-zero area) triangles.
    # cv2.getAffineTransform produces a singular matrix for collinear points,
    # causing warpAffine to fill the entire patch with a single colour.
    # This can happen when nearby landmarks collapse to the same pixel after
    # clipping, or when the dlib→MP index mapping produces duplicate coords.
    if abs(cv2.contourArea(src_tri.astype(np.float32))) < 1.0:
        return
    if abs(cv2.contourArea(dst_tri.astype(np.float32))) < 1.0:
        return

    # Bounding rect of destination triangle
    x, y, w, h = cv2.boundingRect(dst_tri.astype(np.float32))
    x, y = max(x, 0), max(y, 0)
    Hd, Wd = dst_img.shape[:2]
    w = min(w, Wd - x)
    h = min(h, Hd - y)
    if w <= 0 or h <= 0:
        return

    # Crop source patch and compute offsets in the same integer-rect coords
    Hs, Ws = src_img.shape[:2]
    sx, sy, sw, sh = cv2.boundingRect(src_tri.astype(np.float32))
    sx, sy = max(sx, 0), max(sy, 0)
    sw = min(sw, Ws - sx)
    sh = min(sh, Hs - sy)
    if sw <= 0 or sh <= 0:
        return

    # Offset triangles to their respective bounding rect coordinate systems
    src_tri_offset = src_tri - np.array([sx, sy])
    dst_tri_offset = dst_tri - np.array([x, y])

    # Compute affine transform mapping src triangle to dst triangle
    M = cv2.getAffineTransform(
        src_tri_offset.astype(np.float32),
        dst_tri_offset.astype(np.float32)
    )

    # Crop source patch
    src_patch = src_img[sy:sy+sh, sx:sx+sw]

    # Warp patch
    warped_patch = cv2.warpAffine(
        src_patch,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Create triangle mask in destination-rect coords
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_tri_offset.astype(np.int32), 255)

    # Blend into destination
    roi = dst_img[y:y+h, x:x+w]
    mask_3ch = cv2.merge([mask, mask, mask])
    roi[:] = np.where(mask_3ch > 0, warped_patch, roi)


def warp_face(
    source_img: np.ndarray,
    source_lm: np.ndarray,
    displacement: np.ndarray
) -> tuple:
    """
    Warp source face to match the expression encoded in displacement.

    Args:
        source_img:   BGR image (H, W, 3)
        source_lm:    (N, 2) float32 landmark positions on source image
                      (N = 478 for MediaPipe, 68 for dlib — any count works)
        displacement: (N, 2) float32 displacement vectors from Phase 2

    Returns:
        warped_img: (H, W, 3) warped image
        face_mask:  (H, W) uint8 mask of the face region (for blending)
    """
    target_lm = source_lm + displacement

    # Clip target landmarks to image bounds
    h, w = source_img.shape[:2]
    target_lm[:, 0] = np.clip(target_lm[:, 0], 0, w - 1)
    target_lm[:, 1] = np.clip(target_lm[:, 1], 0, h - 1)

    # Delaunay triangulation on source landmarks
    tri = Delaunay(source_lm)

    warped_img = source_img.copy()

    for simplex in tri.simplices:
        src_tri = source_lm[simplex]   # (3, 2)
        dst_tri = target_lm[simplex]   # (3, 2)
        _apply_affine_to_triangle(source_img, warped_img, src_tri, dst_tri)

    # Face mask: convex hull of target landmarks
    hull = cv2.convexHull(target_lm.astype(np.int32))
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)

    return warped_img, face_mask


def warp_image_by_landmarks(
    image: np.ndarray,
    src_lm: np.ndarray,
    dst_lm: np.ndarray,
) -> np.ndarray:
    """
    Warp image so that src_lm landmarks move to dst_lm positions.
    Returns the warped image (same size, no mask).
    """
    displacement = (dst_lm - src_lm).astype(np.float32)
    warped, _ = warp_face(image, src_lm, displacement)
    return warped


if __name__ == "__main__":
    import sys
    # Quick sanity check with a blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    lm = np.random.rand(68, 2).astype(np.float32)
    lm[:, 0] *= 640
    lm[:, 1] *= 480
    disp = np.random.randn(68, 2).astype(np.float32) * 3
    warped, mask = warp_face(img, lm, disp)
    print(f"Warped shape: {warped.shape}, Mask shape: {mask.shape}")
    print("warp.py OK")