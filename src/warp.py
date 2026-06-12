"""
Phase 3: Face Warping via Delaunay Triangulation + Polar Eye Warp
Owner: Member B

Global warp: piecewise affine transforms over a Delaunay mesh.
Eye regions: polar-coordinate warp so eyelid open/close is expressed
as a smooth radial displacement rather than piecewise affine seams.
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


def _apply_polar_radial_warp(
    polar_img: np.ndarray,
    src_angles: np.ndarray,
    dr: np.ndarray,
    max_radius: float,
) -> np.ndarray:
    """
    Shift pixels in a polar image along the radius axis (columns) by a
    per-angle amount interpolated from sparse landmark displacements.

    polar_img   (H, W, 3)  – OpenCV warpPolar output (cols=radius, rows=angle)
    src_angles  (K,)       – landmark angles in [0, 2π)
    dr          (K,)       – radial displacement per landmark (pixels)
    max_radius             – the maxRadius used when building polar_img
    """
    H, W = polar_img.shape[:2]

    order   = np.argsort(src_angles)
    ang_s   = src_angles[order]
    dr_s    = dr[order]

    # Circular wrap-around padding so np.interp handles 0/2π seam correctly
    ang_w = np.r_[ang_s[-1] - 2 * np.pi, ang_s, ang_s[0] + 2 * np.pi]
    dr_w  = np.r_[dr_s[-1],               dr_s,  dr_s[0]]

    row_angles = np.arange(H, dtype=np.float64) / H * 2 * np.pi
    dr_rows    = np.interp(row_angles, ang_w, dr_w).astype(np.float32)

    # Convert pixel-space dr → column-index shift in the polar image
    dr_cols = (dr_rows / max_radius * W).astype(np.float32)   # (H,)

    xs    = np.tile(np.arange(W, dtype=np.float32), (H, 1))          # (H, W)
    ys    = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W)) # (H, W)
    map_x = xs - dr_cols[:, None]   # inverse map: sample from shifted col
    map_y = ys

    return cv2.remap(polar_img, map_x, map_y,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _warp_eye_polar(
    source_img: np.ndarray,
    eye_src_lm: np.ndarray,
    eye_dst_lm: np.ndarray,
    polar_size: int = 256,
    padding: float = 1.8,
):
    """
    Warp one eye region using polar coordinates.

    Eyelid open/close and squint are radial motions from the eye centre,
    so expressing them as per-angle radius shifts in polar space produces
    smoother results than the piecewise affine triangulation mesh.

    The result patch is positioned at the TARGET eye centre (not source) so
    it aligns with the already-Delaunay-warped surroundings. The blend mask
    is the convex hull of the target eyelid landmarks — tight to the eye
    opening — feathered just enough to avoid a hard seam.

    Returns (warped_crop, blend_mask, tx0, ty0) or None on degenerate input.
      warped_crop  (tgt_h, tgt_w, 3) – polar-warped eye patch
      blend_mask   (tgt_h, tgt_w)    – uint8 polygon-based soft mask
      tx0, ty0                       – top-left write position in warped_img
    """
    H_img, W_img = source_img.shape[:2]

    # Source geometry: used for reading the crop and driving the warp
    src_center = eye_src_lm.mean(axis=0)
    cx, cy     = float(src_center[0]), float(src_center[1])
    dists      = np.linalg.norm(eye_src_lm - src_center, axis=1)
    max_radius = float(dists.max()) * padding
    R          = int(np.ceil(max_radius))

    x0 = max(0, int(cx) - R);  x1 = min(W_img, int(cx) + R + 1)
    y0 = max(0, int(cy) - R);  y1 = min(H_img, int(cy) + R + 1)
    crop_w, crop_h = x1 - x0, y1 - y0
    if crop_w < 4 or crop_h < 4:
        return None

    # Target geometry: where to write the result in the Delaunay-warped image
    tgt_center = eye_dst_lm.mean(axis=0)
    tx, ty     = float(tgt_center[0]), float(tgt_center[1])
    tx0 = max(0, int(tx) - R);  tx1 = min(W_img, int(tx) + R + 1)
    ty0 = max(0, int(ty) - R);  ty1 = min(H_img, int(ty) + R + 1)
    tgt_w, tgt_h = tx1 - tx0, ty1 - ty0
    if tgt_w < 4 or tgt_h < 4:
        return None

    crop        = source_img[y0:y1, x0:x1]
    cx_c, cy_c  = cx - x0, cy - y0
    center_crop = (float(cx_c), float(cy_c))

    # Polar decomposition of landmark displacements
    rel_src    = eye_src_lm - src_center
    rel_dst    = eye_dst_lm - src_center
    src_angles = np.arctan2(rel_src[:, 1], rel_src[:, 0]) % (2 * np.pi)
    dr         = np.linalg.norm(rel_dst, axis=1) - np.linalg.norm(rel_src, axis=1)

    # Cartesian → Polar → radial warp → Cartesian (written at target centre)
    polar = cv2.warpPolar(
        crop, (polar_size, polar_size), center_crop, max_radius,
        cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR,
    )
    warped_polar = _apply_polar_radial_warp(polar, src_angles, dr, max_radius)

    cx_tc, cy_tc = tx - tx0, ty - ty0
    warped_crop  = cv2.warpPolar(
        warped_polar, (tgt_w, tgt_h), (float(cx_tc), float(cy_tc)), max_radius,
        cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )

    # Tight polygon mask: convex hull of target eyelid landmarks, feathered
    # slightly to avoid a hard seam. This prevents any halo into surrounding skin.
    lm_local  = (eye_dst_lm - np.array([[tx0, ty0]])).astype(np.int32)
    hull_pts  = cv2.convexHull(lm_local)
    poly_mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    cv2.fillConvexPoly(poly_mask, hull_pts, 255)
    blend_mask = cv2.GaussianBlur(poly_mask, (11, 11), 4)

    return warped_crop, blend_mask, tx0, ty0


def _warp_mouth_rbf(
    source_img: np.ndarray,
    mouth_src_lm: np.ndarray,
    mouth_dst_lm: np.ndarray,
    padding: float = 1.6,
):
    """
    Warp the mouth region using a Gaussian RBF dense flow field.

    Each pixel in the target crop receives a displacement interpolated from
    the sparse lip landmark displacements via a Gaussian kernel — no
    triangulation, so the curved lip surface warps smoothly.

    Returns (warped_patch, blend_mask, tx0, ty0) or None on degenerate input.
      warped_patch  (tgt_h, tgt_w, 3) – RBF-warped mouth patch
      blend_mask    (tgt_h, tgt_w)    – uint8 polygon-based soft mask
      tx0, ty0                        – top-left write position in warped_img
    """
    H_img, W_img = source_img.shape[:2]

    src_min  = mouth_src_lm.min(axis=0)
    src_max  = mouth_src_lm.max(axis=0)
    src_span = src_max - src_min
    pad      = src_span * (padding - 1.0) * 0.5

    # Target write region (based on displaced landmarks)
    tgt_min = mouth_dst_lm.min(axis=0)
    tgt_max = mouth_dst_lm.max(axis=0)
    tx0 = max(0, int(tgt_min[0] - pad[0]))
    ty0 = max(0, int(tgt_min[1] - pad[1]))
    tx1 = min(W_img, int(tgt_max[0] + pad[0]) + 1)
    ty1 = min(H_img, int(tgt_max[1] + pad[1]) + 1)
    tgt_w, tgt_h = tx1 - tx0, ty1 - ty0
    if tgt_w < 4 or tgt_h < 4:
        return None

    disp  = (mouth_dst_lm - mouth_src_lm).astype(np.float64)
    sigma = float(np.linalg.norm(src_span)) * 0.35

    # Dense grid in absolute image coordinates for the target crop
    yy, xx = np.mgrid[ty0:ty1, tx0:tx1].astype(np.float64)
    pts    = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)

    # Gaussian RBF: weight each control point by distance from target landmark
    diffs    = pts[:, None, :] - mouth_dst_lm[None, :, :]   # (N, K, 2)
    dists_sq = (diffs ** 2).sum(axis=2)                      # (N, K)
    w        = np.exp(-dists_sq / (2.0 * sigma ** 2))        # (N, K)
    w       /= w.sum(axis=1, keepdims=True) + 1e-8

    # Inverse-map: where to sample in source_img for each target pixel
    map_x = (xx.ravel() - w @ disp[:, 0]).reshape(tgt_h, tgt_w).astype(np.float32)
    map_y = (yy.ravel() - w @ disp[:, 1]).reshape(tgt_h, tgt_w).astype(np.float32)

    warped_patch = cv2.remap(source_img, map_x, map_y,
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Blend mask: erode inward before feathering so the transition zone
    # stays strictly inside the lip boundary and never bleeds into stubble.
    lm_local  = (mouth_dst_lm - np.array([[tx0, ty0]])).astype(np.int32)
    hull_pts  = cv2.convexHull(lm_local)
    poly_mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    cv2.fillConvexPoly(poly_mask, hull_pts, 255)
    k_erode    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    poly_mask  = cv2.erode(poly_mask, k_erode, iterations=2)
    blend_mask = cv2.GaussianBlur(poly_mask, (9, 9), 3)

    return warped_patch, blend_mask, tx0, ty0


def warp_face(
    source_img: np.ndarray,
    source_lm: np.ndarray,
    displacement: np.ndarray,
    lm_cfg: dict | None = None,
) -> tuple:
    """
    Warp source face to match the expression encoded in displacement.

    Args:
        source_img:   BGR image (H, W, 3)
        source_lm:    (N, 2) float32 landmark positions on source image
                      (N = 478 for MediaPipe, 68 for dlib — any count works)
        displacement: (N, 2) float32 displacement vectors from Phase 2
        lm_cfg:       optional landmark config dict (from landmark_config).
                      When provided, the eye regions are re-warped with a
                      polar-coordinate transform and the mouth region with a
                      Gaussian RBF dense flow field, both applied after the
                      global Delaunay pass.

    Returns:
        warped_img: (H, W, 3) warped image
        face_mask:  (H, W) uint8 mask of the face region (for blending)
    """
    target_lm = source_lm + displacement

    # Clip target landmarks to image bounds
    h, w = source_img.shape[:2]
    target_lm[:, 0] = np.clip(target_lm[:, 0], 0, w - 1)
    target_lm[:, 1] = np.clip(target_lm[:, 1], 0, h - 1)

    # Delaunay triangulation on source landmarks (global warp)
    tri = Delaunay(source_lm)

    warped_img = source_img.copy()

    for simplex in tri.simplices:
        src_tri = source_lm[simplex]   # (3, 2)
        dst_tri = target_lm[simplex]   # (3, 2)
        _apply_affine_to_triangle(source_img, warped_img, src_tri, dst_tri)

    # Polar eye warp overlay — replaces the Delaunay result inside each eye
    # region with a smoother radial warp so eyelid motion doesn't show
    # triangulation seams.
    if lm_cfg is not None:
        for eye_key in ("left_eye_full", "right_eye_full"):
            idx = lm_cfg.get(eye_key)
            if idx is None:
                continue
            result = _warp_eye_polar(
                source_img,
                source_lm[idx],
                target_lm[idx],
            )
            if result is None:
                continue
            warped_eye, blend_mask, x0, y0 = result
            crop_h, crop_w = warped_eye.shape[:2]
            x1 = min(x0 + crop_w, w)
            y1 = min(y0 + crop_h, h)
            aw, ah = x1 - x0, y1 - y0

            alpha  = blend_mask[:ah, :aw].astype(np.float32) / 255.0
            alpha3 = alpha[:, :, None]
            roi    = warped_img[y0:y1, x0:x1]
            roi[:] = (warped_eye[:ah, :aw] * alpha3
                      + roi * (1.0 - alpha3)).astype(np.uint8)

        # Mouth RBF warp overlay — replaces the Delaunay result inside the
        # lip boundary with a smooth dense-flow warp, removing triangulation
        # seams on the curved lip surface.
        mouth_idx = lm_cfg.get("mouth_idx")
        if mouth_idx is not None:
            result = _warp_mouth_rbf(
                source_img,
                source_lm[mouth_idx],
                target_lm[mouth_idx],
            )
            if result is not None:
                warped_mouth, blend_mask, x0, y0 = result
                crop_h, crop_w = warped_mouth.shape[:2]
                x1 = min(x0 + crop_w, w)
                y1 = min(y0 + crop_h, h)
                aw, ah = x1 - x0, y1 - y0

                alpha  = blend_mask[:ah, :aw].astype(np.float32) / 255.0
                alpha3 = alpha[:, :, None]
                roi    = warped_img[y0:y1, x0:x1]
                roi[:] = (warped_mouth[:ah, :aw] * alpha3
                          + roi * (1.0 - alpha3)).astype(np.uint8)

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