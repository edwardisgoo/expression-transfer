"""
landmark_config.py — Landmark mode configuration registry.

Two modes are supported:
  'mp'   — MediaPipe FaceLandmarker, 478 points (default, higher quality)
  'dlib' — dlib shape_predictor_68,   68 points (legacy / lighter weight)

Each config dict exposes:
  n_points        int           total landmark count
  left_eye        list[int]     6-point left-eye cluster (for IOD / alignment)
  right_eye       list[int]     6-point right-eye cluster
  left_eye_outer  int           single outer-corner index  (align_face rotation)
  right_eye_outer int           single outer-corner index
  brow_idx        list[int]     eyebrow landmarks   (for RMSE evaluation)
  mouth_idx       list[int]     full-mouth landmarks (outer + inner)
  jaw_idx         list[int]     jawline landmarks   (identity-preservation check)
  expr_idx        list[int]     expression-relevant subset (brows + mouth)
  inner_lip_idx   list[int]     inner-lip landmarks (mouth-mask in demo.py)
"""
from __future__ import annotations

# ── MediaPipe 478-point ───────────────────────────────────────────────────────
MP478: dict = {
    "n_points"       : 478,
    # 6-point eye clusters — more stable than single outer-corner points
    "left_eye"       : [33, 160, 158, 133, 153, 144],
    "right_eye"      : [362, 385, 387, 263, 373, 380],
    # Single outer-corner index used by align_face() rotation
    "left_eye_outer" : 33,
    "right_eye_outer": 263,
    # Evaluation region indices
    "brow_idx"  : [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    "mouth_idx" : [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
                   78, 191, 80, 13, 308, 402, 14, 88],
    "jaw_idx"   : [234, 93, 132, 58, 172, 136, 150, 149, 152,
                   377, 400, 378, 379, 365, 397, 288, 454],
    # Inner-lip polygon used for mouth-region mask in demo.py
    "inner_lip_idx": [78, 191, 80, 13, 308, 402, 14, 88],
}
# Expression-relevant subset = brows + outer lips + inner lips (30 pts)
MP478["expr_idx"] = (
    MP478["brow_idx"]
    + [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91]   # outer lips
    + MP478["inner_lip_idx"]
)

# ── dlib 68-point ─────────────────────────────────────────────────────────────
DLIB68: dict = {
    "n_points"       : 68,
    # 6-point eye clusters (dlib indices 36-41 left, 42-47 right)
    "left_eye"       : list(range(36, 42)),
    "right_eye"      : list(range(42, 48)),
    # Outer corner indices for rotation
    "left_eye_outer" : 36,
    "right_eye_outer": 45,
    # Evaluation region indices
    "brow_idx"  : list(range(17, 27)),   # 17-26  (left + right brows)
    "mouth_idx" : list(range(48, 68)),   # 48-67  (outer + inner lips)
    "jaw_idx"   : list(range(0,  17)),   # 0-16   (jawline)
    # Inner-lip polygon: dlib points 60-67
    "inner_lip_idx": list(range(60, 68)),
}
# Expression-relevant subset = brows + full mouth (30 pts, same count as MP478)
DLIB68["expr_idx"] = DLIB68["brow_idx"] + DLIB68["mouth_idx"]


# ── Public accessor ───────────────────────────────────────────────────────────

def get_config(mode: str) -> dict:
    """
    Return the landmark configuration dict for the given mode.

    Args:
        mode: 'mp' (MediaPipe 478-point) or 'dlib' (dlib 68-point)

    Returns:
        Config dict — see module docstring for keys.

    Raises:
        ValueError: if mode is not 'mp' or 'dlib'.
    """
    if mode == "mp":
        return MP478
    if mode == "dlib":
        return DLIB68
    raise ValueError(
        f"Unknown landmark mode {mode!r}. "
        "Choose 'mp' (MediaPipe 478-pt) or 'dlib' (dlib 68-pt)."
    )
