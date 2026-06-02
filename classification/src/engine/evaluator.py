"""Unified evaluation for 2D and 3D T-stage classification."""

import time
import numpy as np
import torch
import torch.cuda.amp as amp
from sklearn import metrics as sklearn_metrics

from ..utils.utils import time_to_str
from ..metrics.metrics import specificity, calculate_acc_pre_rec


@torch.no_grad()
def evaluate(model, dataloader, config, label_key="label"):
    """Run evaluation on a dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        config: config dict
        label_key: key for labels ('label' for 2D, 'T_stage' for 3D)

    Returns:
        numpy array: [loss, AUC, accuracy, precision, recall, specificity]
    """
    loss_cfg = config.get("loss", {})
    bce_weight = loss_cfg.get("bce_weight", 0.5)
    l1s_weight = loss_cfg.get("smooth_l1_weight", 0.5)
    dice_weight = loss_cfg.get("dice_weight", 0.0)

    mixed_precision = config.get("training", {}).get("mixed_precision", True)

    # Determine output key
    output_key = "label" if label_key == "label" else "T_stage"

    model = model.eval()
    valid_num = 0
    valid_loss = 0.0
    valid_truth = []
    valid_probability = []
    start_time = time.time()

    for t, batch in enumerate(dataloader):
        model.output_type = ["loss", "inference"]

        with torch.no_grad():
            with amp.autocast(enabled=mixed_precision):
                batch_size = len(batch["index"])
                for k in ["image", label_key]:
                    batch[k] = batch[k].cuda()

                output = model(batch)
                loss1 = output["bce_loss"].mean()
                loss2 = output["l1s_loss"].mean()
                loss3 = output["dice_loss"].mean()

        valid_num += batch_size
        valid_loss += batch_size * (
            bce_weight * loss1 + l1s_weight * loss2 + dice_weight * loss3
        ).item()

        valid_truth.append(batch[label_key].data.cpu().numpy())
        valid_probability.append(output[output_key].data.cpu().numpy())

        print(
            f"\r {valid_num:8d} / {len(dataloader.dataset)}  "
            f"{time_to_str(time.time() - start_time, 'sec')}",
            end="", flush=True,
        )

    print()

    assert valid_num == len(dataloader.dataset)

    truth = np.concatenate(valid_truth)
    probability = np.concatenate(valid_probability)

    loss = valid_loss / valid_num
    auc = sklearn_metrics.roc_auc_score(truth, probability)
    spec = specificity(truth, probability)
    metric = calculate_acc_pre_rec(truth, probability)

    acc = metric[0]
    prec = metric[1]
    rec = metric[2]

    return np.array([loss, auc, acc, prec, rec, spec], dtype=np.float32)
