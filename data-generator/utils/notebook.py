import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from typing import List, Tuple, Dict
from python.anomaly_detection_classifier.utils.bbox import overlaps, calculate_box_size
import ast


def print_metrics(
    gts: pd.Series, predictions: pd.Series, debug: bool = False, silent: bool = False
) -> Tuple[float, float, float, float, int, int, int, int, int]:
    def _assert_unique_values(s: pd.Series) -> bool:
        return (
            # assert that s has only two values and they are 0 and 1
            (s.shape[0] == 2 and (0 in s and 1 in s))
            or
            # assert that s has single value and it is 0 or 1
            (s.shape[0] == 1 and (0 in s or 1 in s))
        )

    gts_unique = gts.unique()
    predictions_unique = predictions.unique()

    assert _assert_unique_values(gts_unique)
    assert _assert_unique_values(predictions_unique)

    total = len(gts)

    TP = sum((gts == 1) & (predictions == 1))
    TN = sum((gts == 0) & (predictions == 0))
    FP = sum((gts == 0) & (predictions == 1))
    FN = sum((gts == 1) & (predictions == 0))

    if debug:
        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, sum: {TP + TN + FP + FN}")
        print(f"{TP}\t{TN}\t{FP}\t{FN}\t{TP + TN + FP + FN}")

    accuracy = (TP + TN) / total + 0.00000001
    precision = TP / (TP + FP + 0.00000001)
    recall = TP / (TP + FN + 0.00000001)
    f1score = 2 * (precision * recall) / (precision + recall + 0.00000001)

    if not silent:
        print(
            f"Recall: {np.round(recall * 100, 2)}",
            f"Precision: {np.round(precision * 100, 2)}",
            f"F1Score: {np.round(f1score * 100, 2)}",
            f"Accuracy: {np.round(accuracy * 100, 2)}",
        )

        print(
            f"{recall}\t{precision}\t{f1score}\t{accuracy}\t{TP}\t{TN}\t{FP}\t{FN}\t{total}"
        )

    return (
        round(recall, 6),
        round(precision, 6),
        round(f1score, 6),
        round(accuracy, 6),
        TP,
        TN,
        FP,
        FN,
        total,
    )


def print_metrics_image_level(df: pd.DataFrame, threshold: float):
    assert "label" in df.columns
    assert "image_name" in df.columns
    assert "patch_label" in df.columns
    assert "probability" in df.columns
    assert "detector_label" in df.columns

    def calculate_global_metrics(label_col: str) -> List:
        m = []
        for image_name in df["image_name"].unique():
            temp_df = df[df["image_name"] == image_name]
            m.append(
                print_metrics(
                    temp_df[label_col],
                    (temp_df["detector_label"] == 1)
                    & (temp_df["probability"] > threshold),
                    silent=True,
                )
            )
        return m

    def calculate_detector_metrics(label_col: str) -> List:
        m = []
        for image_name in df["image_name"].unique():
            temp_df = df[df["image_name"] == image_name]
            m.append(
                print_metrics(
                    temp_df[label_col], temp_df["detector_label"], silent=True
                )
            )
        return m

    def calculate_classifier_metrics(label_col: str) -> List:
        m = []
        for image_name in df["image_name"].unique():
            temp_df = df[df["image_name"] == image_name]
            m.append(
                print_metrics(
                    temp_df[label_col], temp_df["probability"] > threshold, silent=True
                )
            )
        return m

    metrics_df_cols = [
        "recall",
        "precision",
        "f1score",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "SUM",
    ]

    def print_partial_results(metrics: List, title: str):
        metrics_df = pd.DataFrame(metrics, columns=metrics_df_cols).mean()
        recall, precision, f1score, accuracy, TP, TN, FP, FN, SUM = list(
            map(lambda v: round(v, 3), metrics_df.values)
        )
        print(title)
        print(
            f"Avg TP: {TP}, Avg TN: {TN}, Avg FP: {FP}, Avg FN: {FN}, Avg sum: {TP + TN + FP + FN}"
        )
        print(f"{TP}\t{TN}\t{FP}\t{FN}\t{TP + TN + FP + FN}")
        print(
            f"Avg Recall: {np.round(recall * 100, 2)}",
            f"Avg Precision: {np.round(precision * 100, 2)}",
            f"Avg F1Score: {f1score}",
            f"Avg Accuracy: {np.round(accuracy * 100, 2)}",
        )

        print(
            f"{recall}\t{precision}\t{f1score}\t{accuracy}\t{TP}\t{TN}\t{FP}\t{FN}\t{SUM}"
        )
        print()

    print_partial_results(calculate_global_metrics("label"), "Global metrics 'label':")
    print_partial_results(
        calculate_global_metrics("patch_label"), "Global metrics 'patch_label':"
    )
    print_partial_results(
        calculate_detector_metrics("label"), "Detector metrics 'label':"
    )
    print_partial_results(
        calculate_detector_metrics("patch_label"), "Detector metrics 'patch_label':"
    )
    print_partial_results(
        calculate_classifier_metrics("label"), "Classifier metrics 'label':"
    )
    print_partial_results(
        calculate_classifier_metrics("patch_label"), "Classifier metrics 'patch_label':"
    )


def plot_size_analysis(bboxes: List, max: int):
    plt.figure()
    plt.rcParams.update({"font.size": 20})

    boxes_distribution_sizes = (
        pd.Series(list(map(lambda bbox: calculate_box_size(bbox, max), bboxes)))
        .value_counts()
        .sort_index()
    )

    # modify index to contain ranges description
    boxes_distribution_sizes.index = np.concatenate(
        [
            boxes_distribution_sizes.index[:-1].astype(str)
            + "-"
            + boxes_distribution_sizes.index[1:].astype(str),
            np.array([f">{boxes_distribution_sizes.index[-1]}"]),
        ]
    )

    boxes_distribution_sizes.plot(
        kind="bar",
        figsize=(25, 6),
        title="Number of anomalies with specific pixel size",
    )


def plot_anomalies_distribution(
    df: pd.DataFrame,
    label_column: str,
    calc_for_FN: bool,
    plot_normalized: bool,
    threshold: float,
):
    assert "gt_bboxes" in df.columns
    assert "probability" in df.columns

    plt.figure()
    plt.rcParams.update({"font.size": 20})

    if calc_for_FN:
        title_label = "FN"
        filtered_data = df[(df[label_column] == 1) & (df["probability"] <= threshold)]
    else:
        title_label = "FP"
        filtered_data = df[(df[label_column] == 0) & (df["probability"] > threshold)]

    # calculate errors boxes sizes distribution
    error_boxes_distribution = (
        filtered_data["gt_bboxes"]
        .explode()
        .dropna()
        .apply(lambda bbox: calculate_box_size(bbox, 200))
        .value_counts()
        .sort_index()
    )

    error_boxes_distribution.index = np.concatenate(
        [
            error_boxes_distribution.index[:-1],
            np.array([f">{error_boxes_distribution.index[-1]}"]),
        ]
    )

    if plot_normalized:
        # calculate all bboxes sizes distribution
        all_boxes_distribution = pd.DataFrame(
            df["gt_bboxes"]
            .explode()
            .dropna()
            .apply(lambda bbox: calculate_box_size(bbox, 200))
            .value_counts()
            .sort_index()
        )
        all_boxes_distribution.index = np.concatenate(
            [
                all_boxes_distribution.index[:-1],
                np.array([f">{all_boxes_distribution.index[-1]}"]),
            ]
        )

        # concat both and left only values that are in value
        temp_dist = pd.concat(
            [all_boxes_distribution, error_boxes_distribution], axis=1
        ).dropna()
        temp_dist.columns = ["all", "val"]

        # normalize value number by all boxes number
        final_distribution = temp_dist["val"] / temp_dist["all"]
    else:
        final_distribution = error_boxes_distribution

    norm_title_part = " normalized" if plot_normalized else ""
    final_distribution.plot(
        kind="bar",
        figsize=(28, 10),
        title=f"{title_label} anomaly{norm_title_part} size distribution: '{label_column}' method",
        ylabel=f"{'Percent' if plot_normalized else 'Number'} of values from group",
    )


def print_incorrect_predictions(
    incorrect_predictions_df: pd.DataFrame,
    all_gt_bboxes: pd.DataFrame,
    print_additional_clean_patch: bool,
    images_lut: Dict,
):
    counter = 0

    for i, image_index in enumerate(incorrect_predictions_df.index):
        bbox = ast.literal_eval(incorrect_predictions_df["bbox"].loc[image_index])
        original_bbox = ast.literal_eval(
            incorrect_predictions_df["original_bbox"].loc[image_index]
        )
        image_path = incorrect_predictions_df["image_path"].loc[image_index]

        print_cam = False
        if "cam" in incorrect_predictions_df.columns:
            print_cam = True
            cam_img = incorrect_predictions_df["cam"].loc[image_index]

        x, y, w, h = bbox

        imgg = images_lut[image_path].copy()

        current_img_gt_bboxes = all_gt_bboxes[all_gt_bboxes["image_path"] == image_path]
        print(f"BBOXES for {image_path}: {current_img_gt_bboxes.shape}")
        for i, row in current_img_gt_bboxes.iterrows():
            if overlaps(bbox, row["original_bbox"]):
                print(
                    f"GT bbox: {row['original_bbox']}",
                    f", patch bbox: {bbox}",
                    f", anomaly_category: {row['anomaly_category']}",
                )
                gt_x, gt_y, gt_w, gt_h = list(map(round, row["original_bbox"]))
                # RED GT BOXES
                cv2.rectangle(
                    imgg, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (255, 0, 0), 1
                )

        ogt_x, ogt_y, ogt_w, ogt_h = list(
            map(
                round,
                ast.literal_eval(
                    incorrect_predictions_df["original_bbox"].loc[image_index]
                ),
            )
        )
        # GREEN GT BOX FOR GT PATCH
        if incorrect_predictions_df["label"].loc[image_index] == 1:
            cv2.rectangle(
                imgg, (ogt_x, ogt_y), (ogt_x + ogt_w, ogt_y + ogt_h), (0, 255, 0), 1
            )

        img_patch = imgg[y : y + h, x : x + w].copy()

        #     if print_patch:
        counter += 1
        print(
            f"I: {i}\n",
            f"Image index: {image_index}\n",
            f"Image path: {image_path}\n",
            f"Anomaly category: {incorrect_predictions_df['anomaly_category'].loc[image_index]}\n",
            f"Image label: {incorrect_predictions_df['label'].loc[image_index]}\n",
            f"Probability: {incorrect_predictions_df['probability'].loc[image_index]}\n",
            f"Bbox: {bbox}\n",
            f"Original bbox: {original_bbox}\n",
        )

        # for FP printing only single image but for FN print also reference patch
        # without bboxes
        if print_additional_clean_patch:
            plt.rcParams["figure.figsize"] = [20, 10]
            fig, axs = plt.subplots(1, 2 + int(print_cam))
            axs[0].imshow(img_patch)
            axs[1].imshow(images_lut[image_path][y : y + h, x : x + w])

            if print_cam:
                axs[2].imshow(cam_img)

            plt.show()
        else:
            plt.rcParams["figure.figsize"] = [20, 10]
            fig, axs = plt.subplots(1, 1)
            axs.imshow(img_patch)
            plt.show()

    print(counter)


def draw_incorrect_predictions(
    test_df: pd.DataFrame,
    images_lut: Dict,
    gt_bboxes: Dict,
    show_fp: bool,
    threshold: float,
):
    assert "bbox" in test_df.columns
    assert "label" in test_df.columns
    assert "image_path" in test_df.columns
    assert "probability" in test_df.columns

    # FP
    if show_fp:
        incorrect_predictions_df = test_df[
            (test_df["label"] == 0) & (test_df["probability"] >= threshold)
        ]
    else:

        # FN
        incorrect_predictions_df = test_df[
            (test_df["label"] == 1) & (test_df["probability"] < threshold)
        ]

    counter = 0

    for i, image_index in enumerate(incorrect_predictions_df.index):
        if "original_bbox" in incorrect_predictions_df.columns:
            original_bbox = ast.literal_eval(
                incorrect_predictions_df["original_bbox"].loc[image_index]
            )
        else:
            original_bbox = None

        if type(incorrect_predictions_df["bbox"].loc[image_index]) == str:
            bbox = ast.literal_eval(incorrect_predictions_df["bbox"].loc[image_index])
        else:
            bbox = incorrect_predictions_df["bbox"].loc[image_index]

        image_path = incorrect_predictions_df["image_path"].loc[image_index]
        x, y, w, h = bbox

        img = images_lut[image_path].copy()

        for gt_bbox, category_id in gt_bboxes[image_path]:
            if overlaps(bbox, gt_bbox):
                msg = f"GT bbox: {gt_bbox}, patch bbox: {bbox}"
                if "anomaly_category" in incorrect_predictions_df.columns:
                    msg += f", anomaly_category: {category_id}"
                print(msg)

                gt_x, gt_y, gt_w, gt_h = gt_bbox
                cv2.rectangle(
                    img, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (255, 0, 0), 2
                )

        if "original_bbox" in incorrect_predictions_df.columns:
            ogt_x, ogt_y, ogt_w, ogt_h = list(
                map(
                    round,
                    ast.literal_eval(
                        incorrect_predictions_df["original_bbox"].loc[image_index]
                    ),
                )
            )
            cv2.rectangle(
                img, (ogt_x, ogt_y), (ogt_x + ogt_w, ogt_y + ogt_h), (0, 255, 0), 1
            )

        img_patch = img[y : y + h, x : x + w].copy()

        counter += 1
        print(
            f"I: {i}\n",
            f"Image index: {image_index}\n",
            f"Image path: {image_path}\n",
            f"Image label: {incorrect_predictions_df['label'].loc[image_index]}\n",
            f"Probability: {incorrect_predictions_df['probability'].loc[image_index]}\n",
            f"Bbox: {bbox}\n",
        )
        if "original_bbox" in incorrect_predictions_df.columns:
            print(f"Original bbox: {original_bbox}\n")
        if "anomaly_category" in incorrect_predictions_df.columns:
            print(
                f"Anomaly category: {incorrect_predictions_df['anomaly_category'].loc[image_index]}\n"
            )

        plt.rcParams["figure.figsize"] = [16, 6]

        # for FP printing only single image but for FN print also reference patch
        # without bboxes
        if show_fp:
            fig, axs = plt.subplots(1, 1)
            axs.imshow(img_patch)
        else:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(img_patch)
            axs[1].imshow(images_lut[image_path][y : y + h, x : x + w])

        plt.show()

    print(counter)


def draw_fp_fn(df: pd.DataFrame, threshold: float, draw_on_patches: bool = False):
    assert "image_path" in df.columns
    assert "patch_bbox" in df.columns
    assert "anomaly_label" in df.columns
    assert "patch_label" in df.columns
    assert "probability" in df.columns
    assert "predicted_anomaly_bbox" in df.columns

    plt.figure()
    plt.rcParams.update({"font.size": 10})

    groupped_df = df.groupby("image_path").aggregate(list)
    for image_path, row in groupped_df.iterrows():
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        print(f"Img path: {image_path}")

        max_in_col = 4
        cols_num = min(len(row["patch_bbox"]), max_in_col)
        rows_num = int(len(row["patch_bbox"]) / cols_num) + int(
            len(row["patch_bbox"]) % cols_num > 0
        )

        # draw image
        plt.rcParams["figure.figsize"] = [24, 8]
        fig1, axs1 = plt.subplots(1, 3)

        if draw_on_patches:
            # draw gt bboxes on the image
            for bbox_list in row["overlapping_gt_bboxes"]:
                for bbox in bbox_list:
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 1)

        image_copy = image.copy()
        for i in range(len(row["patch_bbox"])):
            x, y, w, h = row["patch_bbox"][i]

            if (row["probability"][i] > threshold) == row["anomaly_label"][i]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            image_copy = cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 1)

        axs1[0].imshow(image_copy)
        axs1[0].set_title("Full patch")
        axs1[1].imshow(cv2.imread(row["prediction_map_path"][0], cv2.IMREAD_GRAYSCALE))
        axs1[1].set_title("Prediction")
        axs1[2].imshow(cv2.imread(row["anomaly_map_path"][0], cv2.IMREAD_GRAYSCALE))
        axs1[2].set_title("Label")
        plt.show()

        # draw patches
        small_patch_size = 6
        plt.rcParams["figure.figsize"] = [
            small_patch_size * cols_num,
            small_patch_size * rows_num,
        ]
        fig2, axs2 = plt.subplots(rows_num, cols_num)

        for i in range(len(row["patch_bbox"])):
            if draw_on_patches:
                # drawing the anomaly predicted by detector
                ox, oy, ow, oh = row["predicted_anomaly_bbox"][i]
                cv2.rectangle(image, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 1)

            x, y, w, h = row["patch_bbox"][i]
            img_patch = image[y : y + h, x : x + w].copy()

            col_index = i % cols_num
            row_index = int(i / cols_num)
            title_gt_bboxes = "\n".join(map(str, row["overlapping_gt_bboxes"][i]))
            title = (
                f"Label: {row['anomaly_label'][i]},"
                + f" Patch label: {row['patch_label'][i]},"
                + f" Probability: {round(row['probability'][i], 3)}"
                + f"\nDetector predicted bbox: {row['predicted_anomaly_bbox'][i]}"
                + f"\nGT overlapping bboxes: {title_gt_bboxes}"
            )

            if rows_num == 1 and cols_num == 1:
                axs2.imshow(img_patch)
                axs2.set_title(title)
            elif rows_num == 1:
                axs2[col_index].imshow(img_patch)
                axs2[col_index].set_title(title)
            else:
                axs2[row_index, col_index].imshow(img_patch)
                axs2[row_index, col_index].set_title(title)

        plt.tight_layout()
        plt.show()


def print_FP_and_FN_numbers(df: pd.DataFrame, threshold: float):
    errors_df = pd.concat(
        [
            df[(df["label"] == 0) & (df["probability"] >= threshold)][
                "dataset_name"
            ].value_counts(),
            df[(df["label"] == 1) & (df["probability"] < threshold)][
                "dataset_name"
            ].value_counts(),
        ],
        axis=1,
    )
    errors_df.columns = ["FP", "FN"]
    errors_df.plot(
        kind="bar", title="FP and FN number groupped by source dataset", figsize=(25, 6)
    )
