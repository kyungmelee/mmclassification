from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
from pycocotools.coco import COCO
import json
import numpy as np
import cv2

from anomaly_detection_classifier.utils import polygons
from pipeline.shapes import Contour


def contour_area(contour: List[Tuple[float, float]]) -> int:
    local_contour = np.array(contour).astype(np.float32).copy()

    if len(local_contour) == 0:
        return 0
    x, y, w, h = cv2.boundingRect(local_contour)

    local_contour[:, 0] = local_contour[:, 0] - x
    local_contour[:, 1] = local_contour[:, 1] - y

    temp_img_part = np.zeros((h, w))
    cv2.fillPoly(temp_img_part, pts=[local_contour.astype(int)], color=1)

    return temp_img_part.sum()


def segmentation_area(segmentation: List[float]) -> int:
    return contour_area(Contour(np.array(segmentation)))


def has_correct_contrast_and_size(
        segmentation: List[float],
        contrast: float,
        contrasts_map: List[Tuple[int, int]],
):
    """
    Function is checking if the annotation should be filtered or not.
    Anomalies with small size and small contrast should be filtered, but the
    values are related.
    @param: segmentation - the contours of the annomalies in COCO format
    @param: contrast - contrast value for the specific anomaly
    @param: contrasts_map - the map with defined minimal contrast for minimal anomaly area
    """
    # TODO probably we could round here values using round function, not simple casting,
    # for small anomalies it will do huge difference
    area = segmentation_area(segmentation)

    # alternative area calculation method using contour only,
    # and not approximating to pixels
    # area = cv2.contourArea(contour)

    for min_area, min_contrast in contrasts_map:
        if area < min_area and contrast < min_contrast:
            return False

    return True


def load_components_annotations(
        annotations_dir: str, components_filename: str = "components.json"
) -> Optional[Dict]:
    def is_component_or_bg_annotation_name(_name: str) -> bool:
        return "comp" in _name.lower() or "background" in _name.lower()

    components_annotations_filepath = os.path.join(annotations_dir, components_filename)
    dataset_subdir = os.path.dirname(annotations_dir)
    dataset_images_dir = os.path.join(dataset_subdir, "images")

    if not os.path.exists(components_annotations_filepath):
        return None

    dataset = COCO(components_annotations_filepath)

    # finding the ids of annotations category which are components or background
    component_and_bg_annotations_ids = set(
        map(
            lambda v: v["id"],
            filter(
                lambda val: is_component_or_bg_annotation_name(val["name"]),
                dataset.cats.values(),
            ),
        )
    )

    if len(component_and_bg_annotations_ids) == 0:
        return None

    components_annotations = {}
    for annotation_meta in dataset.anns.values():
        image_abs_path = os.path.join(
            dataset_images_dir, dataset.imgs[annotation_meta["image_id"]]["file_name"]
        )

        if annotation_meta["category_id"] in component_and_bg_annotations_ids:
            current_image_annotations = components_annotations.get(image_abs_path, [])
            contours = list(map(lambda s: Contour(np.array(s)), annotation_meta["segmentation"]))
            current_image_annotations.extend(contours)
            components_annotations[image_abs_path] = current_image_annotations

    return components_annotations


def load_contrasts_mapping(
        annotations_dir: str, contrasts_filename: str = "contrasts.json"
) -> Dict:
    """
    The annotation could contain contrast mapping for contrast filtering, this
    function is returning those mapping if exists, if not, it return None.
    """
    annotations_contrast = None
    annotations_contrast_filepath = os.path.join(annotations_dir, contrasts_filename)
    if os.path.exists(annotations_contrast_filepath):
        with open(annotations_contrast_filepath) as json_file:
            annotations_contrast = json.load(json_file)

    return annotations_contrast


def load_gt_bboxes(
        annotations_filepaths: List[str],
        contrasts_map: List[Tuple[int, int]],
        components_annotations: Dict,
        component_labels_names: List[str] = ("FootPrint", "ComponentMask"),
        load_filtered_by_contrast_and_size: bool = False,
) -> Dict:
    """
    Function is extracting all annotations which are anomalies and return them in the specific format of a dict
    where the key of the dict is the absolute filename and the value is a list of anomalies bboxes

    @param annotations_filepaths: absolute paths to extract COCO annotations files
    @param component_labels_names: list of the components categories in COCO annotations files
    @contrasts_map: map with minimal contrast and area of the anomaly to consider as correct
    """
    datasets = {}
    for annotations_filepath in tqdm(annotations_filepaths):
        datasets[annotations_filepath] = COCO(annotations_filepath)

    gt_bboxes = {}

    # iterate over all loaded datasets
    for annotations_filepath, dataset in tqdm(datasets.items()):
        # extracting images directory out of absolute COCO filepath name
        dataset_images_dir = os.path.join(
            os.path.dirname(os.path.dirname(annotations_filepath)), "images"
        )
        annotations_contrast = load_contrasts_mapping(annotations_filepath)

        for img_id, img_filename in list(
                map(lambda v: (v["id"], v["file_name"]), dataset.imgs.values())
        ):
            image_filepath = os.path.join(dataset_images_dir, img_filename)

            if image_filepath not in gt_bboxes:
                gt_bboxes[image_filepath] = []

            current_image_gt_bboxes = []
            for ia in dataset.anns.values():
                if (
                        ia["image_id"] == img_id
                        and dataset.cats[ia["category_id"]]["name"]
                        not in component_labels_names
                ):
                    correct_anomaly = True
                    annotation_id = str(ia["id"])
                    if annotations_contrast:
                        if annotation_id in annotations_contrast:
                            contrast_value = annotations_contrast[annotation_id]

                            # since the annotation could be build with a few segments,
                            # we are checking each of this segment separately
                            for sp in ia["segmentation"]:
                                correct_anomaly = has_correct_contrast_and_size(
                                    sp, contrast_value, contrasts_map
                                )
                        else:
                            raise ValueError(
                                "Missing contrast value for the annotation!"
                            )

                    # filtering annotations by components overlapping
                    if components_annotations:
                        for component_annotation in components_annotations.get(
                                image_filepath, []
                        ):
                            # CAUTION: anomaly_meta["segmentation"] contains list of segmentations, not a single one
                            if polygons.overlaps_with_any(
                                    component_annotation,
                                    list(map(lambda s: Contour(np.array(s)), ia["segmentation"])),
                            ):
                                correct_anomaly = False
                                break

                    if not correct_anomaly and not load_filtered_by_contrast_and_size:
                        continue

                    rounded_bbox = list(map(round, ia["bbox"]))
                    current_image_gt_bboxes.append(
                        (
                            rounded_bbox,
                            dataset.cats[ia["category_id"]]["name"],
                            correct_anomaly,
                        )
                    )

            gt_bboxes[image_filepath].extend(current_image_gt_bboxes)
    return gt_bboxes
