import os

import cv2


def load_images(filepaths: list) -> dict:
    """Load images by filepath

    Args:
        filepaths (list)

    Returns:
        dict: {basename: image as np.ndarray}
    """
    images = {}
    for path in filepaths:
        name = os.path.basename(path)
        images[name] = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

    return images
