import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def np_binary_cross_entropy_loss(probability, truth):

    probability = probability.astype(np.float64)
    probability = np.nan_to_num(probability, nan=1, posinf=1, neginf=0)

    p = np.clip(probability, 1e-5, 1-1e-5)
    y = truth

    loss = -y * np.log(p) - (1 - y) * np.log(1 - p)
    loss = loss.mean()
    return loss


def get_cls_metric(probability, truth):
    f1score     = []
    threshold = np.linspace(0, 1, 50)
    
    # pdb.set_trace()

    for t in threshold:
        predict = (probability > t).astype(np.float32)

        tp = ((predict >= 0.5) & (truth >= 0.5)).sum()
        fp = ((predict >= 0.5) & (truth <  0.5)).sum()
        fn = ((predict <  0.5) & (truth >= 0.5)).sum()
        # tn = ((predict <  0.5) & (truth <  0.5)).sum()

        # pdb.set_trace()

        recall    = tp / (tp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)

        f1 = 2 * recall * precision / (recall + precision + 1e-7)
        f1score.append(f1)

    # pdb.set_trace()

    f1score = np.array(f1score)


    return f1score, threshold

