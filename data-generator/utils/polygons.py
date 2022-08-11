from typing import List, Tuple
from shapely.geometry import Polygon, Point, LineString
import numpy as np


def overlaps(shape1, shape2) -> bool:
    return (
        shape1 == shape2
        or shape1.equals(shape2)
        or shape1.intersects(shape2)
        or shape1.contains(shape2)
        or shape2.contains(shape1)
    )


def bbox2contour(bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    x, y, w, h = bbox
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def polygon2contour(polygon: Polygon) -> List[Tuple[float, float]]:
    return np.array(polygon.boundary.coords.xy).T[:-1].tolist()


def _get_shape(cnt: List[Tuple[float, float]]):
    if len(cnt) == 1:
        return Point(cnt[0])
    elif len(cnt) == 2:
        return LineString(cnt)
    return Polygon(cnt)


def overlaps_with_any(
    contour1: List[Tuple[float, float]], contours_list: List[List[Tuple[float, float]]]
) -> bool:
    shape1 = _get_shape(contour1)

    for contour2 in contours_list:
        shape2 = _get_shape(contour2)
        if overlaps(shape1, shape2):
            return True

    return False


def get_overlapping_contours(
    contour1: List[Tuple[float, float]], contours_list: List[List[Tuple[float, float]]]
) -> List[Tuple[float, float]]:

    shape1 = _get_shape(contour1)

    overlapping_contours = []

    for contour2 in contours_list:
        shape2 = _get_shape(contour2)
        if overlaps(shape1, shape2):
            overlapping_contours.append(contour2)

    return overlapping_contours
