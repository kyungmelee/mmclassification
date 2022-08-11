from typing import List, Iterable
from dataclasses import dataclass

from python.pipeline.shapes import Contour


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1score: float
    accuracy: float
    TP: int
    TN: int
    FP: int
    FN: int

    def print(self):
        return f"{self.precision}\t{self.recall}\t{self.f1score}\t{self.accuracy}\t{self.TP}\t{self.TN}\t{self.FP}\t{self.FN}\t"


@dataclass(frozen=True)
class Indicators:
    TP: int = 0
    TN: int = 0
    FP: int = 0
    FN: int = 0

    def __add__(self, other):
        return Indicators(
            self.TP + other.TP,
            self.TN + other.TN,
            self.FP + other.FP,
            self.FN + other.FN,
        )

    def __radd__(self, other):
        return self.__add__(other)

    @staticmethod
    def sum(indicators: Iterable):
        temp_indicators = Indicators()
        for indicator in indicators:
            temp_indicators += indicator
        return temp_indicators

    def calculate_metrics(self) -> Metrics:
        if self.TP + self.FP == 0:
            precision = 0
        else:
            precision = self.TP / (self.TP + self.FP)

        if self.TP + self.FN == 0:
            recall = 0
        else:
            recall = self.TP / (self.TP + self.FN)

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)

        if self.TP + self.TN + self.FP + self.FN == 0:
            accuracy = 0
        else:
            accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

        return Metrics(
            round(precision, 3),
            round(recall, 3),
            round(f1_score, 3),
            round(accuracy, 3),
            self.TP,
            self.TN,
            self.FP,
            self.FN,
        )


def calculate_single_indicator_prediction_perspective(
        predicted_contour: Contour,
        prediction_probability: float,
        target_contours: List[Contour],
        probability_threshold: float = 1.0,
) -> Indicators:
    if predicted_contour.overlaps_with_any(target_contours):
        if prediction_probability >= probability_threshold:
            return Indicators(TP=1)
        else:
            return Indicators(FN=1)
    else:
        if prediction_probability >= probability_threshold:
            return Indicators(FP=1)
        else:
            return Indicators(TN=1)


def calculate_indicators_prediction_perspective(
        prediction_contours: List[Contour],
        target_contours: List[Contour],
        prediction_probabilities: List[float] = None,
        target_probabilities: List[float] = None,
        probability_threshold: float = 1.0,
) -> Indicators:
    if target_probabilities is None:
        target_probabilities = [1.0] * len(target_contours)

    if prediction_probabilities is None:
        prediction_probabilities = [1.0] * len(prediction_contours)

    assert len(prediction_probabilities) == len(prediction_contours)
    assert len(target_probabilities) == len(target_contours)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for prediction_index, prediction_patch in enumerate(prediction_contours):
        for target_index, target_contour in enumerate(target_contours):
            if prediction_patch.overlaps(target_contour):
                prediction_probability = prediction_probabilities[prediction_index]
                target_probability = target_probabilities[target_index]

                if (
                        target_probability >= probability_threshold
                        and prediction_probability >= probability_threshold
                ):
                    TP += 1
                    break
                elif (
                        target_probability < probability_threshold
                        and prediction_probability < probability_threshold
                ):
                    TN += 1
                    break
                elif target_probability >= probability_threshold > prediction_probability:
                    FN += 1
                    break
                elif target_probability < probability_threshold <= prediction_probability:
                    FP += 1
                    break
                else:
                    raise ValueError("Unsupported probabilities pair.")

    for pc_i, pc in enumerate(prediction_contours):
        if not pc.overlaps_with_any(target_contours):
            if prediction_probabilities[pc_i] >= probability_threshold:
                FP += 1
            elif prediction_probabilities[pc_i] < probability_threshold:
                TN += 1

    for tc_i, tc in enumerate(target_contours):
        if (
                not tc.overlaps_with_any(prediction_contours)
                and target_probabilities[tc_i] >= probability_threshold
        ):
            FN += 1

    return Indicators(TP, TN, FP, FN)


def calculate_indicators_target_perspective(
        prediction_contours: List[Contour],
        target_contours: List[Contour],
        prediction_probabilities: List[float] = None,
        target_probabilities: List[float] = None,
        probability_threshold: float = 1.0,
) -> Indicators:
    if target_probabilities is None:
        target_probabilities = [1.0] * len(target_contours)

    if prediction_probabilities is None:
        prediction_probabilities = [1.0] * len(prediction_contours)

    assert len(prediction_probabilities) == len(prediction_contours)
    assert len(target_probabilities) == len(target_contours)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for ts_i, tc in enumerate(target_contours):
        target_shape_found = False

        for ps_i, pc in enumerate(prediction_contours):
            if tc.overlaps(pc):
                t_probability = target_probabilities[ts_i]
                p_probability = prediction_probabilities[ps_i]

                if (
                        t_probability >= probability_threshold
                        and p_probability >= probability_threshold
                ):
                    TP += 1
                elif (
                        t_probability < probability_threshold
                        and p_probability < probability_threshold
                ):
                    TN += 1
                elif t_probability >= probability_threshold > p_probability:
                    FN += 1
                elif t_probability < probability_threshold <= p_probability:
                    FP += 1
                else:
                    raise ValueError("Unsupported probabilities pair.")

                target_shape_found = True
                break

        if not target_shape_found:
            FN += 1

    for pc_i, pc in enumerate(prediction_contours):
        prediction_shape_found = False
        for ts_i, tc in enumerate(target_contours):
            if pc.overlaps(tc):
                prediction_shape_found = True
                break
        if (
                not prediction_shape_found
                and prediction_probabilities[pc_i] >= probability_threshold
        ):
            FP += 1

    return Indicators(TP, TN, FP, FN)
