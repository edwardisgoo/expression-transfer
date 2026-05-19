"""
Phase 2: Expression Parameterization
Owner: Member A

Computes displacement vectors representing a facial expression,
normalized by face scale for robustness across different image sizes.
"""

import numpy as np


def _interocular_distance(landmarks: np.ndarray) -> float:
    """
    Compute inter-ocular distance for scale normalization.
    Left eye outer corner: landmark 36
    Right eye outer corner: landmark 45
    """
    return float(np.linalg.norm(landmarks[45] - landmarks[36]))


def compute_displacement(
    source_lm: np.ndarray,
    driver_lm: np.ndarray,
    driver_neutral_lm: np.ndarray = None,
    scale: float = 1.0
) -> np.ndarray:
    """
    Compute expression displacement vectors.

    Two modes:
    - With driver_neutral_lm (full mode): delta = driver_expressive - driver_neutral,
      rescaled to match source face size. Best results, requires a neutral photo of driver.
    - Without driver_neutral_lm (direct mode): directly warp source landmarks toward
      driver landmark positions, scale-normalized. No neutral photo needed.

    Args:
        source_lm:         (68, 2) landmarks of the source (target) face
        driver_lm:         (68, 2) landmarks of the driver face (expressive)
        driver_neutral_lm: (68, 2) optional — driver neutral baseline
        scale:             multiplier (0.7–1.0 to reduce exaggeration, try 0.7 first)

    Returns:
        displacement: (68, 2) float32 array of (dx, dy) vectors
    """
    source_scale = _interocular_distance(source_lm)

    if driver_neutral_lm is not None:
        # Full mode: delta from neutral to expressive
        raw_delta = driver_lm - driver_neutral_lm
        driver_scale = _interocular_distance(driver_neutral_lm)
        if driver_scale < 1e-6:
            raise ValueError("Driver neutral landmarks degenerate — interocular distance near zero.")
        displacement = raw_delta * (source_scale / driver_scale) * scale
    else:
        # Direct mode: align driver landmarks onto source face space,
        # then displace source landmarks toward the aligned driver positions
        driver_scale = _interocular_distance(driver_lm)
        if driver_scale < 1e-6:
            raise ValueError("Driver landmarks degenerate — interocular distance near zero.")

        # Align driver landmarks to source: match scale and center
        source_center = source_lm.mean(axis=0)
        driver_center = driver_lm.mean(axis=0)
        ratio = source_scale / driver_scale
        driver_aligned = (driver_lm - driver_center) * ratio + source_center

        # Displacement = how far each source landmark needs to move
        displacement = (driver_aligned - source_lm) * scale

    return displacement.astype(np.float32)


def apply_displacement(landmarks: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """
    Apply displacement vectors to a set of landmarks.

    Args:
        landmarks:    (68, 2) source landmark positions
        displacement: (68, 2) displacement vectors

    Returns:
        new_landmarks: (68, 2) displaced landmark positions
    """
    return (landmarks + displacement).astype(np.float32)


if __name__ == "__main__":
    # Mock test with random data
    src = np.random.rand(68, 2).astype(np.float32) * 200 + 100
    drv = np.random.rand(68, 2).astype(np.float32) * 200 + 100
    drv_n = drv + np.random.rand(68, 2).astype(np.float32) * 5

    disp = compute_displacement(src, drv, drv_n)
    print(f"Displacement shape: {disp.shape}")
    print(f"Max displacement: {np.abs(disp).max():.2f}px")
    new_lm = apply_displacement(src, disp)
    print(f"New landmarks shape: {new_lm.shape}")
