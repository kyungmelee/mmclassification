import os.path

from tqdm import tqdm
from typing import List, Dict
import cv2
import numpy as np
from anomaly_detection_classifier.utils.coco import contour_area
from pipeline.shapes import Contour


def build_images_lut(images_paths: List[str]) -> Dict:
    images_lut = {}
    print("Start building images LUT ...")
    for image_path in tqdm(set(images_paths)):
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
        else:
            raise ValueError(f"Image at {image_path} does not exist.")

        images_lut[image_path] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Images LUT built.")
    return images_lut


def generate_contours(image: np.array, min_contour_area: int = 0) -> List[Contour]:
    assert is_binary_image(image)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = list(map(lambda cnt: np.array(list(map(lambda x: x[0], cnt))), contours))

    filtered = list(
        filter(
            lambda cnt: contour_area(cnt) >= min_contour_area,
            contours,
        )
    )

    return filtered


def is_binary_image(image: np.array) -> bool:
    return len(np.unique(image)) <= 2
