from dataclasses import dataclass
from timeit import default_timer as timer

import numpy as np
from typing import List, Optional, Dict
import os
from glob import glob
from cv2 import cv2
from pycocotools.coco import COCO
from abc import ABC, abstractmethod

from utils.bbox import scale_square
from dataloader.params import GroundTruthContourParams, PipelineParams

from dataloader.params import DataParams

from utils.utils import align_images
from dataloader.shapes import (
    Contour,
    generate_contours,
    get_contours,
    filter_contained,
    generate_mask,
    filter_by_size,
    generate_contours_preserving_anomaly_types,
)
from utils.utils import crop_resize


def get_dataset(path: str) -> Optional[COCO]:
    if os.path.exists(path):
        return COCO(path)

    return None


def read_image(image_path: str, grayscale=False) -> Optional[np.array]:
    if os.path.exists(image_path):
        if grayscale:
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    else:
        return None


@dataclass(frozen=True)
class DetectorDataLoaderItem:
    id: str  # image_name

    query_image: np.array
    query_image_path: str

    query_image_bottom: Optional[np.array]
    query_image_path_bottom: Optional[str]

    reference_image: Optional[np.array]
    reference_image_path: Optional[str]

    reference_image_bottom: Optional[np.array]
    reference_image_path_bottom: Optional[str]

    gt_image_contours: List[Contour]  # may be empty
    filtering_components_mask: Optional[np.array]
    components_contours: List[Contour]

    def to_dict(self) -> Dict:
        result_dict = {
            "id": self.id,
            "query_image_path": self.query_image_path,
            "reference_image_path": self.reference_image_path,
            "gt_image_contours": list(map(lambda c: c.to_dict(), self.gt_image_contours)),
            "components_contours": list(map(lambda c: c.to_dict(), self.gt_image_contours))
        }

        if self.query_image_bottom is not None:
            result_dict["query_image_path_bottom"] = self.query_image_path_bottom
            result_dict["reference_image_path_bottom"] = self.reference_image_path_bottom

        return result_dict


@dataclass(frozen=True)
class BaseClassifierDataLoaderItem:
    # combination of image_name + center_of the crop points + patch size
    id: str

    image_name: str
    image_path: str

    # cutted patch from the original image
    patch_image: np.array

    # if any gt contours overlaps bounding box
    overlapping_gt_contours: List[Contour]

    # patch bounding box in form of a contour
    patch_contour: Contour

    @abstractmethod
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "image_name": self.image_name,
            "image_path": self.image_path,
            "patch_contour": self.patch_contour.to_dict(),
            "overlapping_gt_contours": list(map(lambda c: c.to_dict(), self.overlapping_gt_contours)),
        }


@dataclass(frozen=True)
class FeatureClassifierDataLoaderItem(BaseClassifierDataLoaderItem):
    # list of cutted query patches
    query_images: List[np.array]

    # list of cutted reference patches
    reference_images: List[np.array]

    # component mask
    components_mask: np.array


@dataclass(frozen=True)
class ClassifierDataLoaderItem(BaseClassifierDataLoaderItem):

    # cutted patch from the original image
    patch_image: np.array

    # contour which detector predicts as an anomaly in this patch
    detector_predicted_contour: Contour

    def to_dict(self) -> Dict:
        result = super().to_dict()
        result["predicted_contour"] = self.detector_predicted_contour.to_dict()

        return result


class DetectorDataLoader:
    def __init__(
            self,
            data_params: DataParams,
            detector_params: GroundTruthContourParams,
            verbose: bool = False,
            use_bottom_lighting: bool = True
    ):
        loading_start_time = timer()
        self.data_params = data_params
        self.detector_params = detector_params
        self.use_bottom_lighting = use_bottom_lighting

        self.anomalies_dataset = get_dataset(
            os.path.join(data_params.coco_dir, "annotations", "instances_default.json")
        )
        self.components_dataset = get_dataset(
            os.path.join(data_params.coco_dir, "annotations", "components.json")
        )

        self.query_images_dir = os.path.join(data_params.coco_dir, "images", "Middle")
        self.query_bottom_images_dir = os.path.join(data_params.coco_dir, "images", "Bottom")
        self.reference_images_dir = os.path.join(data_params.img_reference_dir, "Middle")
        self.reference_bottom_images_dir = os.path.join(data_params.img_reference_dir, "Bottom")

        self.images_names = sorted(
            [
                os.path.basename(path)
                for path in glob(os.path.join(self.query_images_dir, "*"))
            ]
        )

        # filter FOVs if required
        if self.data_params.fovs_to_load:
            self.images_names = list(
                filter(
                    lambda img_name: int(os.path.splitext(img_name)[0].split("-")[1])
                    in self.data_params.fovs_to_load,
                    self.images_names,
                )
            )

        if verbose:
            processing_time = round(timer() - loading_start_time, 3)
            print(f"Detector data loader initialization took: {processing_time}s.")

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index) -> DetectorDataLoaderItem:
        image_filename = self.images_names[index]
        query_image_path = os.path.join(self.query_images_dir, image_filename)
        query_image = read_image(query_image_path)

        query_bottom_image_path = os.path.join(self.query_bottom_images_dir, image_filename)
        query_image_bottom = read_image(query_bottom_image_path)

        anomalies_contours = get_contours(image_filename, self.anomalies_dataset)

        # filtering ground truth anomalies by anomalies labels
        for label in self.data_params.filter_anomalies_labels:
            anomalies_contours = list(
                filter(lambda c: c.anomaly_type != label, anomalies_contours)
            )

        # filtering anomalies using contrast and size algorithm
        if os.path.exists(self.reference_images_dir):
            reference_image_path = os.path.join(self.reference_images_dir, image_filename)
            reference_image = read_image(reference_image_path)
            filtered_anomalies_contours = generate_contours_preserving_anomaly_types(
                (
                    self._generate_target(
                        query_image=query_image,
                        reference_image=reference_image,
                        target_image=generate_mask(anomalies_contours, query_image.shape[:2]),
                        contrast_threshold=self.detector_params.min_contrast_value,
                        kernel_size=self.detector_params.kernel_size,
                    )
                    > 0
                ).astype(np.uint8),
                anomalies_contours,
                self.detector_params.min_anomaly_area,
            )
        else:
            # if we dont have reference images to apply contrast filtering, we are filtering
            # only by area size
            reference_image_path = None
            reference_image = None
            filtered_anomalies_contours = filter_by_size(
                anomalies_contours, self.detector_params.min_anomaly_area
            )

        if self.components_dataset:
            components_contours = get_contours(image_filename, self.components_dataset)
            components_mask = generate_mask(components_contours, query_image.shape[:2])

            # filter generated anomalies by components
            filtered_anomalies_contours = (
                filter_contained(filtered_anomalies_contours, components_contours)
            )
        else:
            components_contours = []
            components_mask = None

        reference_bottom_image_path = os.path.join(self.reference_bottom_images_dir, image_filename)
        reference_bottom_image = read_image(reference_bottom_image_path)

        return DetectorDataLoaderItem(
            id=image_filename,
            query_image=query_image,
            query_image_bottom=query_image_bottom,
            query_image_path=query_image_path,
            query_image_path_bottom=query_bottom_image_path,
            reference_image=reference_image,
            reference_image_bottom=reference_bottom_image,
            reference_image_path=reference_image_path,
            gt_image_contours=filtered_anomalies_contours,
            reference_image_path_bottom=reference_bottom_image_path,
            filtering_components_mask=components_mask,
            components_contours=components_contours
        )


def build_bottom_image_path(path: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(path)),
        "Bottom",
        os.path.basename(path)
    )


class BaseClassifierDataLoader(ABC):
    def __init__(
            self,
            data_params: DataParams,
            pipeline_params: PipelineParams,
            source_contours: Dict[str, List[Contour]],  # Dict in format {image_id: List[Contour]}
            use_images_cache: bool = True,
            verbose: bool = False,
    ):
        start_time = timer()
        print("Starting initialization of classifier data loader...")

        self.data_params = data_params

        if pipeline_params.classifier_type == "classifier":
            if os.path.exists(data_params.img_reference_dir):
                self.reference_images_dir = os.path.join(data_params.img_reference_dir, "Middle")
            else:
                self.reference_images_dir = None
                print("Reference directory does not exist. Contrast filtering will be disabled.")

            self.query_images_dir = os.path.join(data_params.coco_dir, "images")

        elif pipeline_params.classifier_type == "feature_classifier":
            if os.path.exists(data_params.img_reference_dir):
                self.reference_images_dir = os.path.join(data_params.img_reference_dir, "Bottom")
            else:
                self.reference_images_dir = None
                print("Reference directory does not exist. Contrast filtering will be disabled.")

            self.query_images_dir = os.path.join(data_params.coco_dir, "images", "Bottom")

        else:
            raise Exception("Incorrect classifier type")

        self.pipeline_params = pipeline_params
        self.classifier_params = pipeline_params.classifier_params
        self.detector_params = pipeline_params.gt_contour_params

        self.detectorDataLoader = DetectorDataLoader(
            data_params=data_params,
            detector_params=pipeline_params.gt_contour_params,
            verbose=verbose
        )

        print("Starting building query images LUT...")
        self.query_images_paths = {}
        self.query_images_lut = {}
        self.gt_contours = {}
        for item in self.detectorDataLoader:

            if pipeline_params.use_bottom_images_for_classifier and os.path.exists(item.query_image_path_bottom):
                query_image_path = item.query_image_path_bottom
            else:
                query_image_path = item.query_image_path

            self.query_images_paths[query_image_path] = query_image_path

            if use_images_cache:
                self.query_images_lut[query_image_path] = read_image(query_image_path)

            self.gt_contours[query_image_path] = item.gt_image_contours

        print("Query images LUT initialized. Filtering contours ...")
        self.source_contours_list = []
        for image_path, contours in source_contours.items():
            new_image_path = build_bottom_image_path(image_path)
            if (
                    self.pipeline_params.use_bottom_images_for_classifier
                    and
                    os.path.exists(new_image_path)
            ):
                image_path = new_image_path

            if image_path not in self.query_images_paths:
                continue

            anomalies_contours = self.gt_contours.get(image_path, [])

            image_shape = self._get_query_image(image_path).shape

            for contour in contours:
                # searching for overlapping gt contours to a patch
                patch_contour = contour.to_patch_contour(
                    image_shape, self.classifier_params.patch_size
                )
                overlapping_gt_contours = patch_contour.get_overlapping_contours(
                    anomalies_contours
                )
                self.source_contours_list.append(
                    (image_path, contour, patch_contour, overlapping_gt_contours)
                )

        print("Contours filtered. Classifier dataloader initialized.")
        finish_time = timer()
        if verbose:
            print(
                f"Classifier data loader initialization took:",
                round(finish_time - start_time, 3),
                "s.",
            )

    def _get_query_image(self, image_name: str) -> np.array:
        return self.query_images_lut.get(
            image_name,
            read_image(self.query_images_paths[image_name])
        )

    def __len__(self):
        return len(self.source_contours_list)

    def __getitem__(self, index) -> ClassifierDataLoaderItem:
        image_path, contour, patch_contour, overlapping_gt_contours = self.source_contours_list[index]
        query_image = self._get_query_image(image_path)
        x, y, w, h = patch_contour.get_bbox()

        image_name = os.path.basename(image_path)
        return ClassifierDataLoaderItem(
            id=image_name + str(contour.get_bbox()),
            image_name=image_name,
            image_path=self.query_images_paths[image_path],
            patch_image=query_image[y: y + h, x: x + w].copy(),
            patch_contour=patch_contour,
            detector_predicted_contour=contour,
            overlapping_gt_contours=overlapping_gt_contours,
        )


class SingleImageClassifierDataLoader(BaseClassifierDataLoader):

    def __getitem__(self, index) -> ClassifierDataLoaderItem:
        image_name, contour, patch_contour, overlapping_gt_contours = self.source_contours_list[index]
        query_image = self._get_query_image(image_name)
        x, y, w, h = patch_contour.get_bbox()

        return ClassifierDataLoaderItem(
            id=image_name + str(contour.get_bbox()),
            image_name=image_name,
            image_path=os.path.join(self.query_images_dir, image_name),
            patch_image=query_image[y: y + h, x: x + w].copy(),
            patch_contour=patch_contour,
            detector_predicted_contour=contour,
            overlapping_gt_contours=overlapping_gt_contours,
        )


class FeatureClassifierDataLoader(BaseClassifierDataLoader):
    def __init__(
            self,
            data_params: DataParams,
            pipeline_params: PipelineParams,
            source_contours: Dict[str, List[Contour]],
            use_images_cache: bool = True,
    ):

        super().__init__(
            data_params=data_params,
            pipeline_params=pipeline_params,
            source_contours=source_contours,
            use_images_cache=use_images_cache
        )

        self.crop_offset = 20
        self.crop_sizes = pipeline_params.feature_classifier_params.crop_sizes

        self.reference_images_lut = {}
        if use_images_cache:
            for image_path in self.query_images_lut:
                self.reference_images_lut[image_path] = self._get_reference_image(image_path)

        self.components_dataset = get_dataset(
            os.path.join(data_params.coco_dir, "annotations", "components.json")
        )

        self.component_mask_images_lut = {}
        for image_path in self.query_images_lut:
            image_filename = os.path.basename(image_path)
            components_contours = get_contours(image_filename, self.components_dataset)
            components_mask = generate_mask(components_contours, self.query_images_lut[image_path].shape[:2])

            inverted_components_mask = cv2.threshold(components_mask, 0, 1, cv2.THRESH_BINARY_INV)[1]
            self.component_mask_images_lut[image_path] = inverted_components_mask

    def _get_reference_image(self, image_name: str) -> np.array:

        result = self.reference_images_lut.get(image_name, read_image(
            os.path.join(self.reference_images_dir, os.path.basename(image_name))))
        return result

    def __getitem__(self, index) -> FeatureClassifierDataLoaderItem:

        image_path, contour, patch_contour, overlapping_gt_contours = self.source_contours_list[index]

        query_image = self._get_query_image(image_path)
        ref_image = self._get_reference_image(image_path)
        component_mask = self.component_mask_images_lut[image_path]

        x, y, w, h = scale_square(contour.get_bbox(), query_image.shape[:2], self.classifier_params.patch_size)
        w = self.crop_sizes[0]
        h = self.crop_sizes[0]

        height_min = max(y - self.crop_offset, 0)
        height_max = min(y + h + self.crop_offset, query_image.shape[0])
        width_min = max(x - self.crop_offset, 0)
        width_max = min(x + w + self.crop_offset, query_image.shape[1])

        query_patch = query_image[height_min:height_max, width_min:width_max].copy()
        reference_patch = ref_image[height_min:height_max, width_min:width_max].copy()

        height_min = max(y, 0)
        height_max = min(y + h, query_image.shape[0])
        width_min = max(x, 0)
        width_max = min(x + w, query_image.shape[1])

        component_mask_patch = component_mask[height_min:height_max,
                                              width_min:width_max].copy()

        reference_patch_aligned = align_images(query_patch.copy(), reference_patch.copy())

        reference_patch = reference_patch_aligned[self.crop_offset:-self.crop_offset,
                                                  self.crop_offset:-self.crop_offset].copy()
        query_patch = query_patch[self.crop_offset:-self.crop_offset, self.crop_offset:-self.crop_offset].copy()

        query_images = []
        reference_images = []
        for crop_size in self.crop_sizes:
            query_images.append(crop_resize(query_patch, crop_size))
            reference_images.append(crop_resize(reference_patch, crop_size))

        dataloaderitem = FeatureClassifierDataLoaderItem(
            id=image_path + str(contour.get_bbox()),
            image_name=image_path,
            image_path=os.path.join(self.query_images_dir, image_path),
            patch_contour=patch_contour,
            overlapping_gt_contours=overlapping_gt_contours,
            query_images=query_images,
            reference_images=reference_images,
            patch_image=query_patch,
            components_mask=component_mask_patch
        )

        return dataloaderitem
