"""
Facial landmark detection via MediaPipe Face Mesh.
Returns 68 dlib-compatible landmark points mapped from the 468-point mesh.
"""
from __future__ import annotations
import numpy as np

from .landmark_mp import detect_landmarks_mp, mp_to_dlib68


def detect_landmarks(image) -> np.ndarray | None:
    lm = detect_landmarks_mp(image, refine=True)
    return None if lm is None else mp_to_dlib68(lm)
