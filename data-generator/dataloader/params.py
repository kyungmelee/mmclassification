from typing import List, Tuple, Optional
from pickle import NONE
from typing import Tuple, Optional

from dataclasses import dataclass
import os
import json
import numpy as np

class Params:
    @staticmethod
    def load(json_path: str):
        raise NotImplemented()

    def save(self, json_path: str):
        with open(json_path, "w") as f:
            json.dump(self, f, default=lambda o: o.__dict__, sort_keys=True, indent=4)

@dataclass(frozen=True)
class ContrastImageParams:
    current_image : np.array
    reference_image : np.array
    valid_area_mask : np.array
    mask_kernel_size : int
    # need to set defualt
    threads_per_block : Tuple[int, int]

@dataclass()
class HysterisisThresholdParam:
    threshold_low : int
    threshold_high : int

class DetectorParams:  
    current_image : np.array
    reference_image : np.array
    thres_mode : str # "hyst" 
    hyst_thres_param : HysterisisThresholdParam
    threshold : int # for generate contour .. 
    min_anomaly_area : int # for generate contour ..

class ContrastDetectorParams(DetectorParams):
    valid_area_mask : np.array
    mask_kernel_size : int
    threads_per_block : Tuple[int, int]

    def set_option(self, _thres_mode : str, _hyst_thres_param : HysterisisThresholdParam, 
                    _threshold : int , _mask_kernel_size : int , _threads_per_block : Tuple[int, int] ):
        self.thres_mode = _thres_mode
        self.hyst_thres_param = _hyst_thres_param
        self.threshold = _threshold
        self.mask_kernel_size = _mask_kernel_size
        self.threads_per_block = _threads_per_block

class ColoredDiffDetectorParams(DetectorParams):
    valid_area_mask : np.array
    mask_kernel_size : int # use with generate contour, gt_image_contours
    threads_per_block : Tuple[int, int]
    #difference param
    diff_threshold_h : int
    diff_threshold_s : int
    #stability param
    weight : float
    stability_threshold_h : int
    stability_threshold_s : int
    #white region param
    use_white_region_mask : bool
    threshold_white_region : int
    dilation_k_size : int
    erosion_k_size : int

    def set_option(self, _thres_mode : str, _mask_kernel_size : int , _threads_per_block : Tuple[int, int],
                    _diff_threshold_h : int , _diff_threshold_s : int, 
                    _weight : float, _stability_thres_h : int, _stability_thres_s : int,
                    _use_white_region_mask : bool, _threshold_white_region : int, _dilate_k_size : int, _erosion_k_size : int ):
        self.thres_mode = _thres_mode
        self.mask_kernel_size = _mask_kernel_size
        self.threads_per_block = _threads_per_block
        self.diff_threshold_h = _diff_threshold_h
        self.diff_threshold_s = _diff_threshold_s
        self.weight = _weight
        self.stability_threshold_h = _stability_thres_h
        self.stability_threshold_s = _stability_thres_s
        self.use_white_region_mask = _use_white_region_mask
        self.threshold_white_region = _threshold_white_region
        self.dilation_k_size = _dilate_k_size
        self.erosion_k_size = _erosion_k_size

class RGBColoredDiffDetectorParams(DetectorParams):
    valid_area_mask : np.array
    mask_kernel_size : int # use with generate contour, gt_image_contours
    threads_per_block : Tuple[int, int]
    #difference param
    diff_threshold_r : int
    diff_threshold_g : int
    diff_threshold_b : int
    #white region param
    use_white_region_mask : bool
    threshold_white_region : int
    dilation_k_size : int
    opening_k_size : int

    def set_option(self, _thres_mode : str, _mask_kernel_size : int , _threads_per_block : Tuple[int, int],
                    _diff_threshold_r : int , _diff_threshold_g : int, _diff_threshold_b : int, 
                    _use_white_region_mask : bool, _threshold_white_region : int, _dilation_k_size : int,
                    _opening_k_size : int):
        self.thres_mode = _thres_mode
        self.mask_kernel_size = _mask_kernel_size
        self.threads_per_block = _threads_per_block
        self.diff_threshold_r = _diff_threshold_r
        self.diff_threshold_g = _diff_threshold_g
        self.diff_threshold_b = _diff_threshold_b
        self.use_white_region_mask = _use_white_region_mask
        self.threshold_white_region = _threshold_white_region
        self.dilation_k_size = _dilation_k_size
        self.opening_k_size = _opening_k_size

@dataclass(frozen=True)
class AnomalyDetectorParams:
    kernel_size : int # 1channel 
    min_anomaly_area : int 
    min_contrast_value : int
    # need to set defualt
    threads_per_block : Tuple[int, int]

@dataclass(frozen=True)
class DifferenceImageParams:
    curr_image : np.array # 1channel 
    reference_image : np.array # 1channel 
    valid_area_mask : np.array
    mask_kernel_size : int
    # need to set defualt
    threads_per_block : Tuple[int, int]
    isAngle : bool

@dataclass(frozen=True)
class WhiteRegionEdgeMaskParams:
    gray_image : np.array
    threshold : int
    dilation_k_size : int
    erosion_k_kize : int

@dataclass(frozen=True)
class StabilityParams:
    rgb_image : np.array
    weight : float
    threshold_h : int
    threshold_s : int
    
@dataclass(frozen=True)
class EdgeReflectionCandidatesParams:
    intersection_rate: float
    
@dataclass(frozen=True)
class ContrastParams:
    nc_dilation_k_size : int
    near_contrast_thr : int

@dataclass(frozen=True)
class SizeParams:
    valid_blob_TH: int

@dataclass(frozen=True)
class SimilarityParams:
    connections_dilation_k_size: int
    similarity_diff_threshold : int

@dataclass(frozen=True)
class LuminanceParams:
    luminance_thr_m: float
    luminance_thr_b: float

@dataclass(frozen=True)
class ClassifierParams:
    model_path: str
    model_architecture: str = "efficientnet-b0"
    num_classes: int = 2
    patch_size: int = 128
    probability_threshold: float = 0.5


@dataclass(frozen=True)
class FeatureClassifierParams:
    model_architecture: str = "ResNet"
    threshold: float = 3.0
    crop_sizes: Tuple[int, int, int] = (128, 64, 32)
    holes_mask_threshold: float = 40
    visualize_result: bool = True

@dataclass(frozen=True)
class GroundTruthContourParams:
    min_anomaly_area : int
    min_contrast_value : float
    kernel_size : int

@dataclass(frozen=True)
class LuminanceImageParams:
    threads_per_block : Tuple[int, int]

@dataclass
class PipelineParams(Params):
    #data generator 
    anomaly_detector_params : AnomalyDetectorParams
    #detector
    contrast_detector_params: ContrastDetectorParams
    #contrast_use_hystThreshold: bool  # use connector    
    colored_detector_params: Optional[ColoredDiffDetectorParams]
    rgb_detector_params: Optional[RGBColoredDiffDetectorParams]
    #classify 
    classifier_params: ClassifierParams
    feature_classifier_params: FeatureClassifierParams
    classifier_type: str

    #filter
    use_bottom_images_for_classifier: bool
    edgereflections_candidates_params: Optional[EdgeReflectionCandidatesParams]
    filter_contrast_params : ContrastParams
    filter_luminance_params : LuminanceParams
    filter_size_params : SizeParams
    filter_Similarity_params : SimilarityParams

    #ground truth contour 
    gt_contour_params : GroundTruthContourParams
    luminance_image_params : LuminanceImageParams

    calculate_balanced_classifier_kpis: bool

    return_detailed_detector_results: bool
    detailed_detector_results_file_path: str
    return_detailed_classifier_results: bool
    detailed_classifier_results_file_path: str
    return_detailed_classifier_balanced_results: bool
    detailed_classifier_balanced_results_file_path: str
    save_result_image_tensorboard : bool
    save_detailed_result_image_tensorboard : bool
    
    @staticmethod
    def load(json_path: str):
        if not os.path.exists(json_path):
            raise ValueError(f"Config file in path: {json_path} does not exists.")

        with open(json_path, "r") as f:
            data = json.load(f)

        colored_detector_data = data["colored_detector_params"]
        white_region_data = colored_detector_data['white_region_masking_params']
        stability_data = colored_detector_data['stability_masking_params']
        colored_detector_params = ColoredDiffDetectorParams()
        colored_detector_params.set_option(
            _thres_mode="hyst",
            _mask_kernel_size=colored_detector_data['kernel_size'],
            _threads_per_block=colored_detector_data['threads_per_block'],
            _diff_threshold_h=colored_detector_data['diff_h_threshold'],
            _diff_threshold_s=colored_detector_data['diff_s_threshold'],
            
            _weight = stability_data['diff_coeff']
            if stability_data else NONE,
            _stability_thres_h = stability_data['thr_h_local']
            if stability_data else NONE,
            _stability_thres_s = stability_data['thr_s_local']
            if stability_data else NONE,
            
            _use_white_region_mask = True 
            if white_region_data else False,
            _threshold_white_region = white_region_data['white_thr'] 
            if white_region_data else None,
            _dilate_k_size = white_region_data['dilation_k_size'] 
            if white_region_data else None,
            _erosion_k_size = white_region_data['erosion_k_size'] 
            if white_region_data else None  
        )

        contrast_detector_data = data["contrast_detector_params"]
        contrast_image_data = contrast_detector_data['contrast_image_params']
        contrast_detector_params = ContrastDetectorParams()
        contrast_detector_params.set_option(
            _thres_mode=contrast_detector_data['thres_mode'],
            _hyst_thres_param = HysterisisThresholdParam(contrast_detector_data['hyst_low_thr'], contrast_detector_data['hyst_high_thr']),
            _threshold=contrast_detector_data['min_contrast_value'],
            _mask_kernel_size=contrast_image_data['kernel_size']
            if contrast_image_data else NONE,
            _threads_per_block=contrast_image_data['threads_per_block']
            if contrast_image_data else NONE
        )

        rgb_detector_data = data["rgb_detector_params"]
        rgb_detector_params = RGBColoredDiffDetectorParams()
        rgb_detector_params.set_option(
            _thres_mode = 'threshold',
            _mask_kernel_size = rgb_detector_data['kernel_size'],
            _threads_per_block = rgb_detector_data['threads_per_block'],
            _diff_threshold_r = rgb_detector_data['r_diff_thr'],
            _diff_threshold_g = rgb_detector_data['g_diff_thr'],
            _diff_threshold_b = rgb_detector_data['b_diff_thr'],
            _use_white_region_mask = True,
            _threshold_white_region = rgb_detector_data['white_thr'],
            _dilation_k_size = rgb_detector_data['white_dilation_k_size'],
            _opening_k_size = rgb_detector_data['opening_k_size']
        )

        
        return PipelineParams(
            contrast_detector_params=contrast_detector_params,
            colored_detector_params=colored_detector_params,
            rgb_detector_params=rgb_detector_params,
            anomaly_detector_params = AnomalyDetectorParams(**data["anomaly_detector_params"]),
            classifier_params=ClassifierParams(**data["classifier_params"]),
            feature_classifier_params=FeatureClassifierParams(**data["feature_classifier_params"]),
            classifier_type=data["classifier_type"],

            use_bottom_images_for_classifier=data["use_bottom_images_for_classifier"],
            edgereflections_candidates_params=EdgeReflectionCandidatesParams(**data["edgereflections_candidates_params"])
            if data["edgereflections_candidates_params"] else None,
            
            filter_size_params = SizeParams(**data["filter_size_params"]),
            filter_Similarity_params= SimilarityParams(**data["filter_Similarity_params"]),
            filter_contrast_params = ContrastParams(**data["filter_contrast_params"]),
            filter_luminance_params = LuminanceParams(**data["filter_luminance_params"]),
            
            gt_contour_params=GroundTruthContourParams(**data["ground_truth_contour_params"]),
            luminance_image_params=LuminanceImageParams(**data["luminance_image_params"]),
            calculate_balanced_classifier_kpis=data["calculate_balanced_classifier_kpis"],
            
            return_detailed_detector_results=data["return_detailed_detector_results"],
            detailed_detector_results_file_path=data["detailed_detector_results_file_path"],
            return_detailed_classifier_results=data["return_detailed_classifier_results"],
            detailed_classifier_results_file_path=data["detailed_classifier_results_file_path"],
            return_detailed_classifier_balanced_results=data["return_detailed_classifier_balanced_results"],
            detailed_classifier_balanced_results_file_path=data["detailed_classifier_balanced_results_file_path"],
            save_result_image_tensorboard=data["save_result_image_tensorboard"],
            save_detailed_result_image_tensorboard=data["save_detailed_result_image_tensorboard"]
        )


@dataclass(frozen=True)
class DataParams(Params):
    # TODO: take all coco datasaet, replace images with bmp from KJ format, and fix annotations
    # path do dir with coco format data, store components to filter in coco format also
    coco_dir: str

    # path to dir with only reference images
    img_reference_dir: str

    filter_anomalies_labels: List[str]

    # fovs ids to use in the data loader
    fovs_to_load: List[int] = ()

    @staticmethod
    def load(json_path: str):
        if not os.path.exists(json_path):
            raise ValueError(f"Config file in path: {json_path} does not exists.")

        with open(json_path, "r") as f:
            data = json.load(f)

        return DataParams(
            coco_dir=data["coco_dir"],
            img_reference_dir=data["img_reference_dir"],
            filter_anomalies_labels=data["filter_anomalies_labels"],
            fovs_to_load=data["fovs_to_load"]
        )


@dataclass(frozen=True)
class DatasetsMetadata(Params):
    datasets: List[DataParams]

    @staticmethod
    def load(json_path: str):
        if not os.path.exists(json_path):
            raise ValueError(f"Config file in path: {json_path} does not exists.")

        with open(json_path, "r") as f:
            data = json.load(f)

        datasets = [
            DataParams(
                coco_dir=dataset["coco_dir"],
                img_reference_dir=dataset["img_reference_dir"],
                filter_anomalies_labels=dataset["filter_anomalies_labels"],
                fovs_to_load=dataset["fovs_to_load"]
            )
            for dataset in data["datasets"]]

        return DatasetsMetadata(datasets=datasets)
