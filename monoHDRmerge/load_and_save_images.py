import os

import cv2


def load_images(filepaths: list) -> dict:
    images = {}
    for path in filepaths:
        name = os.path.basename(path)
        images[name] = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

    return images
