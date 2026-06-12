"""
landmark_mp.py — MediaPipe Face Landmarker using the Tasks API (mediapipe >= 0.10)

Provides:
- detect_landmarks_mp(image, refine=True) -> np.ndarray of shape (478, 2)
- mp_to_dlib68(mp_lm) -> np.ndarray of shape (68, 2) approximated subset
"""
from __future__ import annotations
import os
import urllib.request
import numpy as np
import cv2

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(os.path.abspath(_MODEL_PATH)), exist_ok=True)
        print("[landmark_mp] Downloading face_landmarker.task model (~30 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[landmark_mp] Model saved to {_MODEL_PATH}")


try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _mp_available = True
    _import_error = None
except Exception as e:
    _mp_available = False
    _import_error = e


# MediaPipe → dlib 68 index mapping.
# Fixes vs original: jawline no longer has duplicate MP indices (0 and 16 both
# mapped to 152); outer upper lip (dlib 49-53) now maps to actual upper-lip MP
# indices instead of lower-lip ones; nose tip uses MP 4 (canonical tip point).
_MP_TO_68 = {
    # Jawline (0-16): left ear → chin → right ear
    0: 234, 1: 93,  2: 132, 3: 58,  4: 172,
    5: 136, 6: 150, 7: 149, 8: 152,
    9: 377, 10: 400, 11: 378, 12: 379,
    13: 365, 14: 397, 15: 288, 16: 454,
    # Left eyebrow (17-21)
    17: 70,  18: 63,  19: 105, 20: 66,  21: 107,
    # Right eyebrow (22-26)
    22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
    # Nose bridge (27-30)
    27: 168, 28: 6,   29: 197, 30: 1,
    # Nose base (31-35)
    31: 98,  32: 97,  33: 4,   34: 326, 35: 327,
    # Left eye (36-41)
    36: 33,  37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
    # Right eye (42-47)
    42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
    # Outer lips: upper left→right (48-54), lower right→left (55-59)
    48: 61,  49: 40,  50: 37,  51: 0,   52: 267, 53: 270,
    54: 291, 55: 321, 56: 314, 57: 17,  58: 84,  59: 91,
    # Inner lips: upper left→right (60-63), lower right→left (64-67)
    60: 78,  61: 191, 62: 80,  63: 13,
    64: 308, 65: 402, 66: 14,  67: 88,
}


# Module-level singleton detectors (keyed by blendshapes flag).
# Creating a FaceLandmarker loads the ~30 MB model and initialises XNNPACK.
# Re-doing this for every image in batch mode would be extremely slow, so we
# cache one instance per mode and reuse it across calls.
_detector_cache: dict = {}


def _get_detector(blendshapes: bool) -> "_mp_vision.FaceLandmarker | None":
    """Return a cached FaceLandmarker, creating it on first call per mode."""
    if not _mp_available:
        return None
    if blendshapes not in _detector_cache:
        _ensure_model()
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_python.BaseOptions(
                model_asset_path=os.path.abspath(_MODEL_PATH)
            ),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=blendshapes,
        )
        _detector_cache[blendshapes] = _mp_vision.FaceLandmarker.create_from_options(options)
    return _detector_cache[blendshapes]


def _run_detector(image: np.ndarray, blendshapes: bool = False):
    """
    Run FaceLandmarker and return (landmarks_array, blendshapes_array).
    Either element may be None on detection failure.
    Centralises model setup so both public functions share one code path.

    The underlying FaceLandmarker is cached as a module-level singleton so
    the model is only loaded once per process (per blendshapes mode).
    """
    if not _mp_available:
        print(f"[landmark_mp] mediapipe not available: {_import_error}")
        return None, None

    detector = _get_detector(blendshapes)
    if detector is None:
        return None, None

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if not result.face_landmarks:
        print("[landmark_mp] Warning: no face detected.")
        return None, None

    pts = [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]]
    lm_array = np.array(pts, dtype=np.float32)

    bs_array = None
    if blendshapes and result.face_blendshapes:
        bs_array = np.array(
            [c.score for c in result.face_blendshapes[0]], dtype=np.float32
        )

    return lm_array, bs_array


def detect_landmarks_mp(image: np.ndarray, refine: bool = True) -> np.ndarray | None:
    """Return (478, 2) pixel-coordinate landmark array, or None."""
    lm, _ = _run_detector(image, blendshapes=False)
    return lm


def detect_blendshapes_mp(image: np.ndarray) -> np.ndarray | None:
    """
    Return a (52,) float32 array of MediaPipe blendshape scores, or None.
    Scores are in [0, 1]. Index 0 is _neutral; indices 1-51 cover brows,
    eyes, jaw, and mouth action units.
    """
    _, bs = _run_detector(image, blendshapes=True)
    return bs


def mp_to_dlib68(mp_lm: np.ndarray) -> np.ndarray:
    out = np.zeros((68, 2), dtype=np.float32)
    for k, v in _MP_TO_68.items():
        out[k] = mp_lm[v]
    return out
