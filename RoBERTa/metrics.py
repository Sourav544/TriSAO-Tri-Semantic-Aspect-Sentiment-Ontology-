import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List

class Metrics:
    @staticmethod
    def calc(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
        accuracy = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        subset_acc = np.mean(np.all(y_true == y_pred, axis=1))
        return {
            'accuracy': accuracy,
            'subset_accuracy': subset_acc,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }