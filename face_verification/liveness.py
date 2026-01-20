import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def check_liveness(image_path):
    """
    Static liveness detection using texture analysis (LBP).

    Returns:
        is_live (bool): Whether face is considered live
        texture_score (float): Texture variance score
    """

    img = cv2.imread(image_path)
    if img is None:
        return False, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LBP parameters
    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(
        gray, n_points, radius, method="uniform"
    )

    # Texture variance
    texture_score = np.var(lbp)

    # Empirical threshold
    LIVE_THRESHOLD = 50

    is_live = texture_score > LIVE_THRESHOLD

    return is_live, texture_score