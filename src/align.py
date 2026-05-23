from __future__ import annotations
"""
Face alignment utilities.

Provides a simple similarity alignment that rotates the face so that the
inter-ocular line is horizontal. This reduces geometric variance and
improves warping stability.

We intentionally keep the canvas size the same as the input image and only
apply a rotation (no scaling) to preserve FOV and avoid cropping.

Supports both MediaPipe 478-point and dlib 68-point landmark layouts via
an optional ``lm_cfg`` dict (see ``src.landmark_config``).  When omitted,
MediaPipe defaults are used so all existing call-sites remain unchanged.
"""
import numpy as np
import cv2

# Default outer-eye-corner indices (MediaPipe 478)
_DEFAULT_LEFT_EYE_OUTER  = 33
_DEFAULT_RIGHT_EYE_OUTER = 263


def _eye_angle(landmarks: np.ndarray,
               left_outer: int,
               right_outer: int) -> float:
    p1 = landmarks[left_outer]
    p2 = landmarks[right_outer]
    dx, dy = (p2 - p1).astype(float)
    return float(np.arctan2(dy, dx))


def align_face(image: np.ndarray,
               landmarks: np.ndarray,
               lm_cfg: dict | None = None):
    """
    Rotate image so that the line between outer eye corners is horizontal.

    Args:
        image:     BGR image (H, W, 3)
        landmarks: (N, 2) float32 landmark array
                   (N = 478 for MediaPipe, 68 for dlib)
        lm_cfg:    Optional landmark-mode config dict from
                   ``src.landmark_config.get_config()``.
                   Keys used: ``left_eye_outer``, ``right_eye_outer``.
                   When None (default), MediaPipe 478 indices are used.

    Returns:
        aligned_img:       rotated image (same H, W)
        aligned_landmarks: rotated (N, 2) landmark array
        M:                 2×3 affine transform applied to the image
        M_inv:             inverse 2×3 transform
    """
    cfg = lm_cfg or {}
    left_outer  = cfg.get("left_eye_outer",  _DEFAULT_LEFT_EYE_OUTER)
    right_outer = cfg.get("right_eye_outer", _DEFAULT_RIGHT_EYE_OUTER)

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    theta = _eye_angle(landmarks, left_outer, right_outer)
    # Rotate by -theta to make eye line horizontal
    angle_deg = -theta * 180.0 / np.pi
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    aligned_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

    # Transform landmarks (augment with ones column)
    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    pts  = np.hstack([landmarks.astype(np.float32), ones])
    aligned_lm = (pts @ M.T).astype(np.float32)

    M_inv = cv2.invertAffineTransform(M)
    return aligned_img, aligned_lm, M, M_inv
