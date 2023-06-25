from typing import List, Dict, Tuple, Generator, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from numpy import ndarray
from matplotlib import pyplot as plt

import pdb


class Metrics:
    """
    - ROC (ROC curve) - sklearn
    - AUC (Area Under the Curve) - skelearn
    - F-score (F1-score) - sklearn
    - Accuracy - sklean
    - EER (Equal Error Rate)
    - FAR (False Acceptance Rate)
    - FRR (False Rejection Rate)
    - CMC curve (Cumulative Match Characteristic curve
    """

    __metrics: List[str] = []
    __enable_plot: bool = False

    def __init__(self, enable_plot: bool = False, metrics: Optional[List[str]] = None):
        self.__enable_plot = enable_plot
        self.__metrics = metrics if metrics else []

    def evaluate(
        self,
        y_true: Tuple[ndarray, ndarray, Any, Any],
        y_pred: List[int],
        requested_metrics: List[Tuple[str, Optional[Dict[str, Any]]]] | None = None,
    ) -> Generator[Tuple[str, float], None, None]:
        if not requested_metrics:
            raise ValueError("You must specify at least one metric to evaluate")

        metrics_methods = self.__get_metrics_methods(requested_metrics)

        for method, params in metrics_methods:
            if method in dir(self):
                yield (method, getattr(self, method)(y_true, y_pred, **params))

    @staticmethod
    def accuracy_score(y_true, y_pred, **kwargs):
        return accuracy_score(y_true, y_pred, **kwargs)

    @staticmethod
    def precision_score(y_true, y_pred, **kwargs):
        return precision_score(y_true, y_pred, average="weighted", **kwargs)

    @staticmethod
    def recall_score(y_true, y_pred, **kwargs):
        return recall_score(y_true, y_pred, average="weighted", **kwargs)

    @staticmethod
    def f1_score(y_true, y_pred, **kwargs):
        return f1_score(y_true, y_pred, average="weighted", **kwargs)

    @staticmethod
    def roc_curve(y_true, y_pred, **kwargs):
        return roc_curve(y_true, y_pred, **kwargs)

    @staticmethod
    def roc_auc_score(y_true, y_pred, **kwargs):
        return roc_auc_score(y_true, y_pred, **kwargs)

    def __get_metrics_methods(
        self, requested_metrics: List[Tuple[str, Optional[Dict[str, Any]] | None]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get all metrics names and parameters from requested_metrics
        """
        metrics_methods = []
        for metric, params in requested_metrics:
            if metric in dir(self):
                metrics_methods.append((metric, params)) if params else metrics_methods.append((metric, {}))
        return metrics_methods
