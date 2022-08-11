from calendar import c
import os
from pickle import NONE
import cv2

os.environ['KMP_DUPLICATE_LIB_OK']='True' # or numpy version missmatch 

from dataloader.dataloaders import DetectorDataLoader, DetectorDataLoaderItem
from dataloader.params import DataParams, PipelineParams
from dataloader.shapes import Contour

import pandas as pd
import numpy as np
from typing import Tuple, List

from dataclasses import dataclass

from glob import glob
from tqdm import tqdm

import random 

from sklearn.model_selection import train_test_split

main_data_dir = r"Z:\CommonModule\classifierTraning\Database\training datasets\data\data\new_dataset\anomaly_detection"
main_save_dir = r".\data\AnomalyDatasetTest"
json_param_path = r".\data-generator\default_pipeline_params.json"
label_str = ["Background","Anomaly"]

g_patch_size = 128 # need to connect json param 

@dataclass
class TrainingPatch:
    contour: Contour
    label: int
    image_path: str
    image_shape: Tuple[int, int]
    patch_size: int
    
    image : np.array # need to check 

    new_patch_bbox : Tuple[int, int, int, int]
    
    @staticmethod
    def _extract_dataset_name(image_path: str) -> str:
        return os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(image_path))))
        )

    def __init__(
        self, contour: Contour, label: int, image_path: str, image_shape: Tuple[int, int], patch_size: int
    ) -> None:
        self.contour = contour
        self.label = label
        self.image_path = image_path
        self.image_shape = image_shape
        self.patch_size = patch_size

    def _build_id(self):
        return (
                f"{self.image_path}_{self.contour.get_bbox()}"
                + f"_{self.contour.points}_{self.label}"
        )

    def _load_image(self):    
        self.image = cv2.imread(self.image_path)
        self.image = self.image[self.contour.bbox[1]:self.contour.bbox[1]+self.contour.bbox[3],self.contour.bbox[0]:self.contour.bbox[0]+self.contour.bbox[2], :]
    
    def _update_patch_size(self):
        self.new_patch_bbox = self.contour.bbox 
        if(self.new_patch_bbox[2] < g_patch_size):
            tmpbboxLeft = self.new_patch_bbox[0]+(self.new_patch_bbox[2]/2) - g_patch_size/2 
            tmpbboxRight =self.new_patch_bbox[0]+(self.new_patch_bbox[2]/2) + g_patch_size/2 
            if(tmpbboxLeft < 0):
                tmpbboxLeft = 0
            if(tmpbboxRight > self.image_shape[0]):
                tmpbboxRight = self.image_shape[0]
            temp = list(self.new_patch_bbox)
            temp[0] = int(tmpbboxLeft)
            temp[2] = int(tmpbboxRight - tmpbboxLeft) 
            self.new_patch_bbox = tuple(temp) # tuple 은 변경이 안된다. list로 변환 후 변경 가능 
        
        if(self.new_patch_bbox[3] < g_patch_size):
            tmpbboxTop = self.new_patch_bbox[1]+(self.new_patch_bbox[3]/2) - g_patch_size/2 
            tmpbboxBottom =self.new_patch_bbox[1]+(self.new_patch_bbox[3]/2) + g_patch_size/2 
            if(tmpbboxTop < 0):
                tmpbboxTop = 0
            if(tmpbboxBottom > self.image_shape[1]):
                tmpbboxBottom = self.image_shape[1]
            temp = list(self.new_patch_bbox)
            temp[1] = int(tmpbboxTop)
            temp[3] = int(tmpbboxBottom - tmpbboxTop) 
            self.new_patch_bbox = tuple(temp) 
           
        
    def save_patch_image(self, save_path:str, prefixid = ""):
        self._load_image()
        full_path = os.path.join(save_path,str(self.label).zfill(3)+"_"+label_str[self.label])
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        image_name = prefixid+"_patch"+str(self.contour.anomaly_id).zfill(3)+"_"+ self.contour.anomaly_type+".bmp"
        cv2.imwrite(full_path+"/"+image_name, self.image)
        self.image = np.empty(0) # release for memory issue 

    def to_dict(self):
        self._update_patch_size()
        return {
            "id": self._build_id(),
            "image_path": self.image_path,
            "dataset": self._extract_dataset_name(self.image_path),
            "bbox": self.contour.to_patch_contour(
                self.image_shape, self.patch_size
            ).get_bbox(),
            "anomaly_bbox": self.contour.get_bbox(),
            "segmentation": self.contour.points.tolist(),
            "anomaly_category": self.contour.anomaly_type,
            "label": self.label,
            "anomaly_id": self.contour.anomaly_id,
            "new_bbox" : self.new_patch_bbox
        }

def generate_empty_patches(
        image_shape: Tuple[int, int],
        patch_size: int,
        stride_size: int,
        image_path: str,
        contours_to_omit: List[Contour],

) -> List[TrainingPatch]:
    empty_patches = []
    image_h, image_w = image_shape[:2]
    for x in np.arange(start=0, stop=image_w - patch_size, step=stride_size):
        for y in np.arange(start=0, stop=image_h - patch_size, step=stride_size):
            current_contour = Contour.from_bbox(
                (x, y, patch_size, patch_size), "normal"
            )
            if current_contour.overlaps_with_any(contours_to_omit):
                continue

            empty_patches.append(
                TrainingPatch(
                    contour=current_contour,
                    label=0,
                    image_path=image_path,
                    image_shape=image_shape[:2],
                    patch_size=patch_size
                )
            )

    return empty_patches

def save_image_from_df(df: pd.DataFrame, folder:str, prefix:str ):
    image = cv2.imread(df['image_path'])
    bbox = df.bbox #['new_bbox'] # df['anomaly_bbox']
    image = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2], :]
    full_path = os.path.join(folder,str(df.label).zfill(3)+"_"+label_str[df.label])
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    image_name = prefix+"_patch"+str(df['anomaly_id']).zfill(3)+"_"+ df['anomaly_category']+".bmp"
    cv2.imwrite(full_path+"/"+image_name, image)
    image = np.empty(0) # release for memory issue 


def get_not_empty_patches(item: DetectorDataLoaderItem, patch_size: int) -> (List, List):
    anomaly_patches = []
    components_patches = []

    for gt_contour in item.gt_image_contours:
        # If we found component that fits in the standard patch size we get it
        if (
                gt_contour.anomaly_type == "ComponentMask"
                and
                gt_contour.get_bbox()[2] <= patch_size
                and
                gt_contour.get_bbox()[3] <= patch_size
        ):

            components_patches.append(
                TrainingPatch(
                    contour=gt_contour,
                    label=0,  
                    image_path=item.query_image_path,
                    image_shape=item.query_image.shape,
                    patch_size=patch_size
                )
            )
        elif gt_contour.anomaly_type not in ["FootPrint", "ComponentMask"]:
            anomaly_patches.append(
                TrainingPatch(
                    contour=gt_contour,
                    label=1,  
                    image_path=item.query_image_path,
                    image_shape=item.query_image.shape,
                    patch_size=patch_size
                )
            )

    for component_contour in item.components_contours:
        if component_contour.get_bbox()[2] <= patch_size and component_contour.get_bbox()[3] <= patch_size:
            components_patches.append(
                TrainingPatch(
                    contour=component_contour,
                    label=0, # NEED TO CHANGE 
                    image_path=item.query_image_path,
                    image_shape=item.query_image.shape,
                    patch_size=patch_size
                )
            )

    return anomaly_patches, components_patches


def process_dataset(params: Tuple) -> (List[TrainingPatch], List[TrainingPatch], List[TrainingPatch]):
    datapath, patch_size, filter_anomalies_labels = params

    print(f"Start processing dataset: {datapath}")

    detector_dataloader = DetectorDataLoader(
        DataParams(
            coco_dir=datapath,
            img_reference_dir=os.path.join(datapath, "reference_images"),
            filter_anomalies_labels=filter_anomalies_labels
        ),
        PipelineParams.load(json_param_path).anomaly_detector_params,
    )
    all_anomaly_patches = []
    all_empty_patches = []
    all_small_component_patches = []

    # we are generating empty patches until we have them >= of anomaly patches
    # the first loop of generation we are using the same stride of the moving windw
    # as we have patch size, if we are not able to generate >= amount of empty patches
    # with that setting we are shrinking the size of the stride 2 times, then 4 times,
    # then 8 times and so on until we will have number of empty patches >= number
    # of anomaly patches
    for stride_power in range(5):
        for item in tqdm(detector_dataloader):
            anomaly_patches, components_patches = get_not_empty_patches(item, patch_size)
            contours_to_omit = item.gt_image_contours + item.components_contours

            stride_size = int(patch_size / pow(2, stride_power))

            empty_patches = generate_empty_patches(
                image_shape=item.query_image.shape,
                patch_size=patch_size,
                stride_size=stride_size,
                image_path=item.query_image_path,
                contours_to_omit=contours_to_omit
            )

            all_anomaly_patches.extend(anomaly_patches)
            all_small_component_patches.extend(components_patches)
            all_empty_patches.extend(empty_patches)

        # we finish generation if the empty patches while the number of empty patches
        # is bigger than anomaly patches, then we could select the same number of empty
        # patches as we have anomalies
        if len(all_empty_patches) >= len(all_anomaly_patches):
            break

        all_anomaly_patches = []
        all_empty_patches = []
        all_small_component_patches = []

    return (
        all_anomaly_patches + random.sample(all_empty_patches, len(all_anomaly_patches)),
        all_empty_patches,
        all_small_component_patches
    )


def flatten(t: List[List]) -> List:
    return [item for sublist in t for item in sublist]


def _parse_fov_number(path: str) -> int:
    return int(os.path.basename(path).split("-0")[1][:2])


def split_dataset(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    #     df = pd.read_csv("../data/test.csv")
    # filtering anomalies with anomaly type count < 10
    df = df[
        ~df["anomaly_category"].isin(
            df["anomaly_category"].value_counts()[df["anomaly_category"].value_counts() < 10].keys()
        )
    ]

    # TODO fix that, temporarly changing the label for edge reflection to 0 to be threaded as not anomaly
    df.at[df["anomaly_category"] == "edge reflection", "label"] = 0

    # ju2 and hs01 datasets required split on image level, not anomaly level to train and test datasets
    ju2_part = df[df["dataset"].apply(lambda dataset: "JU2_query" in dataset)]
    ju2_train_part = ju2_part[
        ju2_part["image_path"].apply(lambda path: _parse_fov_number(path) in [1, 3, 5, 6, 8, 11])
    ]
    ju2_test_part = ju2_part[
        ju2_part["image_path"].apply(lambda path: _parse_fov_number(path) not in [1, 3, 5, 6, 8, 11])
    ]

    hs01_part = df[df["dataset"].apply(lambda dataset: "HS01_Query" in dataset)]
    hs01_train_part = hs01_part[
        hs01_part["image_path"].apply(lambda path: _parse_fov_number(path) in [1, 3, 5, 6, 8, 11])
    ]
    hs01_test_part = hs01_part[
        hs01_part["image_path"].apply(lambda path: _parse_fov_number(path) not in [1, 3, 5, 6, 8, 11])
    ]

    # data part that is not in Blue normal JU2 or HS01
    other_df = df[
        df["dataset"].apply(
            lambda d: "Blue Normal" not in d and "JU2_query" not in d and "HS01_Query" not in d
        )
    ]

    blue_normal_df = df[df["dataset"] == "Blue Normal"]

    temp_df = other_df

    if blue_normal_df.shape[0] > 0:
        # selecting blue normal anomalies to use in training process,
        # we have to reduce them because of overrepresentation in training dataset
        filtered_blue_normal, _ = train_test_split(
            blue_normal_df,
            train_size=other_df.shape[0],
            stratify=blue_normal_df["label"].apply(str) + blue_normal_df["anomaly_category"],
            random_state=123,
            shuffle=True
        )

        temp_df = pd.concat([temp_df, filtered_blue_normal])

    # aggregation of the temp_df on the same columns on which we want to do stratification
    temp_df_aggregated = temp_df.groupby(["dataset", "anomaly_category"]).aggregate(list)["id"]

    # finding an ids of the groups with single object (those cannot be split into train and test parts and should
    # be removed)
    ids_to_omit = temp_df_aggregated[temp_df_aggregated.apply(lambda ids: len(ids) == 1)].explode().to_list()

    # filtering single groups ids
    temp_df = temp_df[~temp_df["id"].isin(ids_to_omit)]

    # making train test split
    temp_train_df, temp_test_df = train_test_split(
        temp_df,
        stratify=temp_df["dataset"] + temp_df["anomaly_category"],
        train_size=0.8,
        random_state=123,
        shuffle=True
    )

    # generating final sets
    global_test_df = pd.concat([ju2_test_part, hs01_test_part, temp_test_df])
    global_train_df = pd.concat([ju2_train_part, hs01_train_part, temp_train_df])

    return global_train_df, global_test_df

def get_datapaths(root_dir):
    
    return list(map(lambda path: os.path.dirname(path), glob(rf"{root_dir}\*\*\images")))


def main():
    training_patches = []
    empty_patches = []
    small_components_patches = []

    filter_anomalies_labels = ["Board cutting"]

    for datapath in get_datapaths(main_data_dir):
        training_patches_sample, empty_patches_sample, components_patches_sample = process_dataset(
            (datapath, 128, filter_anomalies_labels)
        )
        training_patches.extend(training_patches_sample)
        empty_patches.extend(empty_patches_sample)
        small_components_patches.extend(components_patches_sample)

    # data_files_suffix = "edge_reflection_included_hs01_included_bottom_images_fixed_paths_additional_components"
    data_files_suffix = "new_dataset_generate"

    all_empty_df = pd.DataFrame(list(map(lambda p: p.to_dict(), empty_patches)))
    all_empty_df.to_csv(f"all_empty_patches_{data_files_suffix}.csv", index=False)

    all_small_components_df = pd.DataFrame(list(map(lambda p: p.to_dict(), small_components_patches)))
    all_small_components_df.to_csv(f"all_small_components_patches_{data_files_suffix}.csv", index=False)

    df = pd.DataFrame(list(map(lambda p: p.to_dict(), training_patches)))
    df.to_csv(f"all_training_and_testing_dataset_{data_files_suffix}.csv", index=False)

    train_df, test_df = split_dataset(df)
    train_df.to_csv(f"train_df_new_dataformat_{data_files_suffix}.csv", index=False)
    test_df.to_csv(f"test_df_new_dataformat_{data_files_suffix}.csv", index=False)

    anomaly_types_distribution = pd.Series(
        list(map(lambda p: p.contour.anomaly_type, training_patches))
    ).value_counts()

    label_distribution = pd.Series(
        list(map(lambda p: p.label, training_patches))
    ).value_counts()

    #nLabelID = 0
    #for data in training_patches:
    #    data.save_patch_image(main_save_dir+"/training", str(nLabelID).zfill(3))
    #    nLabelID = nLabelID+1 

    #nLabelID = 0
    #for data in empty_patches:
    #    data.save_patch_image(main_save_dir+"/valid", str(nLabelID).zfill(3))
    #    nLabelID = nLabelID+1 

    #nLabelID = 0
    #for data in small_components_patches:
    #    data.save_patch_image(main_save_dir+"/small", str(nLabelID).zfill(3))
    #    nLabelID = nLabelID+1 

    nLabelID = 0
    for i in train_df.index:
        save_image_from_df(train_df.loc[i],main_save_dir+"/training", str(nLabelID).zfill(3))
        nLabelID = nLabelID+1

    nLabelID = 0
    for i in test_df.index:
        save_image_from_df(test_df.loc[i],main_save_dir+"/valid", str(nLabelID).zfill(3))
        nLabelID = nLabelID+1
    

    print(f"Loaded {len(training_patches)} patches.")
    print(f"Anomalies types in training patches:\n{anomaly_types_distribution}")
    print(f"Anomalies labels in training patches:\n{label_distribution}")


if __name__ == "__main__":
    main()
