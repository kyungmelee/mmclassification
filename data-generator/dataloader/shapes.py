from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import cv2
from pycocotools.coco import COCO

from utils.bbox import scale_square


@dataclass(init=False)
class Contour:
    def __init__(self, points: np.array, anomaly_type: str = "anomaly", anomaly_id : int = 1):
        self.points = points
        self.anomaly_type = anomaly_type
        self.anomaly_id = anomaly_id

        # x, y, w, h
        self.bbox: Tuple[int, int, int, int] = cv2.boundingRect(
            self.points.astype(np.int)
        )

        self.local_points = self._move_points_by(
            self.points, -self.bbox[0], -self.bbox[1]
        )
        self.local_bbox = cv2.boundingRect(self.local_points.astype(np.int))
        self.mask = self._build_mask()
        self.area = self._build_mask().sum()

    @classmethod
    def from_bbox(cls, bbox: Tuple[int, int, int, int], anomaly_type: str = "anomaly"):
        return Contour(
            np.array(cls._points_from_bbox(bbox)),
            anomaly_type
        )

    @staticmethod
    def _move_points_by(points: np.array, x: int, y: int) -> np.array:
        moved_points = points.copy()
        moved_points[:, 0] += x
        moved_points[:, 1] += y
        return moved_points

    def _build_mask(self) -> np.array:
        return cv2.fillPoly(
            np.zeros(
                (self.local_bbox[3], self.local_bbox[2]),
            ),  # dimensions are in the correct order first h, then w
            [self.local_points.astype(np.int)],
            True,
        ).astype(np.bool)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        return self.points[item]

    def get_area(self):
        return self.area

    def get_bbox(self):
        return self.bbox

    def get_center_point(self) -> (float, float):
        x, y, w, h = self.bbox
        return (x + w / 2), (y + h / 2)

    def is_point_in_bbox(self, x: float, y: float) -> bool:
        cx, cy, cw, ch = self.bbox
        return cx <= x <= cx + cw and cy <= y <= cy + ch

    @staticmethod
    def _points_from_bbox(bbox) -> List:
        x, y, w, h = bbox
        if x < 0 or y < 0 or w < 0 or h < 0:
            raise ValueError("Values below zero.")
        return [
            (x, y),
            (max(0, x + w), y),
            (max(0, x + w), max(0, y + h)),
            (x, max(0, y + h))
        ]

    def to_patch_contour(self, img_shape: Tuple[int, int], patch_size: int):
        return Contour(
            points=np.array(self._points_from_bbox(
                scale_square(self.get_bbox(), img_shape, patch_size)
            )),
            anomaly_type=self.anomaly_type
        )

    def to_dict(self) -> Dict:
        return {"points": self.points.tolist(), "anomaly_type": self.anomaly_type}

    @staticmethod
    def from_dict(vals: Dict):
        assert "points" in vals
        assert 'anomaly_type' in vals

        return Contour(np.array(vals["points"]), vals["anomaly_type"])

    def __hash__(self):
        return self.points.tostring().__hash__() + self.anomaly_type.__hash__()

    def __eq__(self, other):
        return (
                np.array_equal(self.points, other.points)
                and
                self.anomaly_type == other.anomaly_type
        )

    def dilate(self, kernel: np.array):
        assert max(kernel.shape) <= 100

        points = self._move_points_by(self.points, 100, 100)

        mask = cv2.fillPoly(
            np.zeros((int(points.max() + 100), int(points.max() + 100))),
            [points.astype(np.int)],
            1,
        ).astype(np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        contours_points = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0][0]
        contours_points = np.array([cp[0] for cp in contours_points])
        contours_points = self._move_points_by(contours_points, -100, -100)

        return Contour(contours_points, self.anomaly_type, self.anomaly_id)

    @staticmethod
    def _bbox_overlaps(
            a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
    ) -> bool:
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y
        return w > 0 and h > 0

    def overlaps_with_any(self, contours_list: List) -> bool:
        for contour2 in contours_list:
            if self.overlaps(contour2):
                return True

        return False

    def overlaps(self, cnt) -> bool:
        if self._bbox_overlaps(self.bbox, cnt.bbox):
            mask1, mask2 = self._build_related_masks(self, cnt)

            return np.any(np.bitwise_and(mask1, mask2).sum())

        return False

    @staticmethod
    def _build_related_masks(c1, c2) -> Tuple:
        """
        Function is building related masks using smalles possible region to fit both
        contours.
        """
        x, y, w, h = c1.bbox
        ox, oy, ow, oh = c2.bbox

        new_min_x = min(x, ox)
        new_min_y = min(y, oy)

        diff_x_from_min_x = x - new_min_x
        diff_y_from_min_y = y - new_min_y

        diff_ox_from_min_x = ox - new_min_x
        diff_oy_from_min_y = oy - new_min_y

        new_w = max(diff_x_from_min_x + x + w, diff_ox_from_min_x + ox + ow) - new_min_x
        new_h = max(diff_y_from_min_y + y + h, diff_oy_from_min_y + oy + oh) - new_min_y

        mask1 = np.zeros((new_h, new_w), dtype=bool)
        mask1[
        diff_y_from_min_y: diff_y_from_min_y + h,
        diff_x_from_min_x: diff_x_from_min_x + w,
        ] = c1.mask

        mask2 = np.zeros((new_h, new_w), dtype=bool)
        mask2[
        diff_oy_from_min_y: diff_oy_from_min_y + oh,
        diff_ox_from_min_x: diff_ox_from_min_x + ow,
        ] = c2.mask

        return mask1, mask2

    def contains(self, cnt):
        if self._bbox_overlaps(self.bbox, cnt.bbox):
            mask1, mask2 = self._build_related_masks(self, cnt)

            # calculate the overlapping part of two masks
            common_part = np.bitwise_and(mask1, mask2)

            # checking if the second mask has anything more than the common part
            return not np.any(np.bitwise_xor(mask2, common_part))

        return False

    def contained_by_any(self, contours_list: List) -> bool:
        for contour2 in contours_list:
            if contour2.contains(self):
                return True

        return False

    def get_overlapping_contours(self, contours_list: List) -> List:
        """
        The function returns the list of overlapping contours to self.
        @param contours_list: list of contours to search for overlapping ones.
        """
        overlapping_contours = []
        for contour2 in contours_list:
            if self.overlaps(contour2):
                overlapping_contours.append(contour2)

        return overlapping_contours


@dataclass(frozen=True)
class PredictedContour:
    contour: Contour
    probability: float


def generate_mask(contours: List[Contour], mask_shape: Tuple[int, int]) -> np.array:
    # TODO maybe revise this rounding points to int because this is only CASTING
    # I added this in that form to have consistent metrics because we used this
    # in that form before
    contour_points = list(map(lambda c: c.points.astype(np.int), contours))
    mask = cv2.fillPoly(np.zeros(mask_shape), contour_points, 1)

    return mask.astype(np.uint8)


def filter_by_size(contours: List[Contour], min_size) -> List[Contour]:
    return list(filter(lambda cnt: cnt.get_area() >= min_size, contours))


def generate_contours(image: np.array, min_contour_area: int = 0) -> List[Contour]:
    def is_binary_image(_image: np.array) -> bool:
        return len(np.unique(_image)) <= 2

    assert is_binary_image(image)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_points = list(
        map(lambda cnt: np.array(list(map(lambda x: x[0], cnt))), contours)
    )

    return filter_by_size(list(map(Contour, contours_points)), min_contour_area)


def generate_contours_preserving_anomaly_types(
        image: np.array, old_contours: List[Contour], min_contour_area: int = 0
) -> List[Contour]:
    def is_binary_image(_image: np.array) -> bool:
        return len(np.unique(_image)) <= 2

    assert is_binary_image(image)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_points = list(
        map(lambda cnt: np.array(list(map(lambda x: x[0], cnt))), contours)
    )

    filtered_by_size = filter_by_size(list(map(Contour, contours_points)), min_contour_area)

    fixed_contours = []
    for contour in filtered_by_size:
        overlapping_contours = contour.get_overlapping_contours(old_contours)
        overlapping_contours_anomaly_types = list(
            set(map(lambda c: c.anomaly_type, overlapping_contours))
        )

        if len(overlapping_contours_anomaly_types) == 1:
            fixed_contours.append(
                Contour(contour.points, overlapping_contours_anomaly_types[0])
            )
        else:
            biggest_mask_contour = overlapping_contours[0]
            for overlapping_c in overlapping_contours:
                if overlapping_c.mask.sum() > biggest_mask_contour.mask.sum():
                    biggest_mask_contour = overlapping_c

            fixed_contours.append(
                Contour(contour.points, biggest_mask_contour.anomaly_type)
            )
    return fixed_contours


def filter_overlapping(
        source: List[Contour], to_filter: List[Contour]
) -> List[Contour]:
    return list(
        filter(
            lambda c: not c.overlaps_with_any(to_filter),
            source,
        )
    )


def filter_contained(source: List[Contour], to_filter: List[Contour]) -> List[Contour]:
    return list(
        filter(
            lambda c: not c.contained_by_any(to_filter),
            source,
        )
    )


def _segmentation2contour(segmentation: List[float]) -> List[Tuple[float, float]]:
    return [
        (segmentation[i], segmentation[i + 1])
        for i in range(0, len(segmentation) - 1, 2)
    ]


def get_contours(image_filename: str, dataset: COCO) -> List[Contour]:
    if dataset is None:
        return []

    image_index = -1
    for index, val in dataset.imgs.items():
        if val["file_name"] == image_filename:
            image_index = index
            break
    if image_index == -1:
        return []

    gt_contours = []
    for anomaly_id, anomaly_meta in dataset.anns.items():
        if anomaly_meta["image_id"] == image_index:
            anomaly_category = dataset.cats[anomaly_meta["category_id"]]["name"]

            # if the anomaly contains multiple segmentations it will be connected into
            # one list of points
            points = []
            for segmentation in anomaly_meta["segmentation"]:
                points.extend(_segmentation2contour(segmentation))

            # in case we dont have segmentation (when the annotator is not using polygon but
            # the box for annotation) we will use bounding box of annotation as contour
            if not points:
                points.extend(Contour._points_from_bbox(anomaly_meta["bbox"]))

            # if there was no points in segmentations, we don't add the contour
            if points:
                gt_contours.append(
                    Contour(points=np.array(points), anomaly_type=anomaly_category, anomaly_id=anomaly_meta["category_id"])
                )

    return gt_contours
