"""Evaluation metrics for binary T-stage classification (T2 vs T3)."""

import numpy as np
from sklearn.metrics import accuracy_score


def _to_binary(y_true, y_pred, threshold=0.5):
    """Convert probabilities to binary predictions."""
    y_pred_bin = (y_pred > threshold).astype(np.float32).reshape(-1)
    y_true_bin = (y_true > threshold).astype(np.float32).reshape(-1)
    return y_true_bin, y_pred_bin


def precision(y_true, y_pred):
    """Binary precision."""
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)


def recall(y_true, y_pred):
    """Binary recall (sensitivity)."""
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)


def specificity(y_true, y_pred):
    """Average specificity across thresholds [0, 1]."""
    thresholds = np.linspace(0, 1, 100)
    specs = []
    for t in thresholds:
        pred = (y_pred > t).astype(np.float32)
        tn = ((pred < 0.5) & (y_true < 0.5)).sum()
        fp = ((pred >= 0.5) & (y_true < 0.5)).sum()
        specs.append(tn / (tn + fp + 1e-15))
    return np.mean(specs)


def f_score(y_true, y_pred, beta=1):
    """F-beta score. beta=1 gives F1, beta=0.5 gives F0.5, beta=2 gives F2."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-15)


def dice_score(y_true, y_pred):
    """Dice coefficient."""
    return (2 * (y_true * y_pred).sum() + 1e-15) / (
        y_true.sum() + y_pred.sum() + 1e-15
    )


def jac_score(y_true, y_pred):
    """Jaccard (IoU) coefficient."""
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Compute comprehensive classification metrics.

    Returns:
        [recall, precision, accuracy, F0.5, F1, F2, dice, jaccard]
    """
    y_true_bin, y_pred_bin = _to_binary(y_true, y_pred, threshold)

    return [
        recall(y_true_bin, y_pred_bin),
        precision(y_true_bin, y_pred_bin),
        accuracy_score(y_true_bin, y_pred_bin),
        f_score(y_true_bin, y_pred_bin, beta=0.5),
        f_score(y_true_bin, y_pred_bin, beta=1),
        f_score(y_true_bin, y_pred_bin, beta=2),
        dice_score(y_true_bin, y_pred_bin),
        jac_score(y_true_bin, y_pred_bin),
    ]


def calculate_acc_pre_rec(y_true, y_pred, threshold=0.5):
    """Compute accuracy, precision, recall.

    Returns:
        [accuracy, precision, recall]
    """
    y_true_bin, y_pred_bin = _to_binary(y_true, y_pred, threshold)

    return [
        accuracy_score(y_true_bin, y_pred_bin),
        precision(y_true_bin, y_pred_bin),
        recall(y_true_bin, y_pred_bin),
    ]
