from typing import List, Tuple, Dict, Union, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from numpy import float16
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

    def evaluate(self, y_true: List[int], y_pred: List[int], **kwargs) -> Dict[str, float16]:
        metrics_methods = self.__get_metrics_methods(**kwargs)

        for method in metrics_methods:
            if method in dir(self):
                yield method, getattr(self, method)(y_true, y_pred, **kwargs)

    @staticmethod
    def accuracy_score(y_true, y_pred, **kwargs):
        return accuracy_score(y_true, y_pred, **kwargs)

    @staticmethod
    def precision_score(y_true, y_pred, **kwargs):
        return precision_score(y_true, y_pred, **kwargs)

    @staticmethod
    def recall_score(y_true, y_pred, **kwargs):
        return recall_score(y_true, y_pred, **kwargs)

    @staticmethod
    def f1_score(y_true, y_pred, **kwargs):
        return f1_score(y_true, y_pred, **kwargs)

    @staticmethod
    def roc_curve(y_true, y_pred, **kwargs):
        return roc_curve(y_true, y_pred, **kwargs)

    @staticmethod
    def roc_auc_score(y_true, y_pred, **kwargs):
        return roc_auc_score(y_true, y_pred, **kwargs)

    def __get_metrics_methods(self, requested_metrics: List[str]) -> List[str]:
        """
        Get all metrics methods from class
        """
        metrics_methods = []
        for metric in requested_metrics:
            if metric in dir(self):
                metrics_methods.append(metric)
        return metrics_methods
