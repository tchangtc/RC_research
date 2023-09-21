import os
import random
import torch
import numpy as np
from collections import defaultdict
from torch.optim import Optimizer
# from Config import CFG
from sklearn.metrics import accuracy_score
# cfg = CFG()

# def scheduler(epoch):
#         #return start_lr
#         # num_epoch = cfg.epochs / 2
#         num_epoch = cfg.decay_epochs
#         start_lr = cfg.start_lr
#         min_lr = cfg.min_lr

#         lr = (num_epoch-epoch)/num_epoch * (start_lr-min_lr) + min_lr
#         lr = max(min_lr,lr)
#         return lr

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']
    # return optimizer.param_groups[0]['lr']


def time_to_str(t, mode='min'):
    if mode == 'min':
        t  = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode=='sec':
        t   = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)

    else:
        raise NotImplementedError


""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

# def dice_score(y_true, y_pred):
#     return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

# def jac_score(y_true, y_pred):
#     intersection = (y_true * y_pred).sum()
#     union = y_true.sum() + y_pred.sum() - intersection
#     return (intersection + 1e-15) / (union + 1e-15)


def calculate_metrics(y_true, y_pred):
    # y_true = y_true.detach().cpu().numpy()
    # y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.float32)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.float32)

    ## Score
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_recall, score_precision, score_acc, score_fbeta]

class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

def Specifity(y_true, y_pred):
    specifity = []
    # threshold = np.linspace(0.45, 0.55, 100)
    threshold = np.linspace(0, 1, 100)
    # pdb.set_trace()
    for t in threshold:
        predict = (y_pred > t).astype(np.float32)

        tp = ((predict>=0.5) & (y_true>=0.5)).sum()
        tn = ((predict<0.5) & (y_true<0.5)).sum()
        fp = ((predict>=0.5) & (y_true< 0.5)).sum()
        fn = ((predict< 0.5) & (y_true>=0.5)).sum()

        # acc2 = (tp+tn)/(tp+tn+fp+fn+1e-15)
        # p = tp/(tp+fp+1e-15)
        # r = tp/(tp+fn+1e-15)
        spe = tn/(tn+fp+1e-15)
        
        # accuracy1.append(acc1)
        # accuracy2.append(acc2)
        # precision.append(p)
        # recall.append(r)
        specifity.append(spe)

    specifity=np.array(specifity)

    return specifity.mean()



def calculate_acc_pre_rec(y_true, y_pred):

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.float32)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.float32)

    ## Score
    score_acc = accuracy_score(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_recall = recall(y_true, y_pred)

    return [score_acc, score_precision, score_recall]
