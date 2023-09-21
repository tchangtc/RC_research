import pdb
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler
from Model.model_3D import Classification3D
from MyDataset.dataset import CRCDataset, make_train_test_df_filling_well, make_train_test_df, null_collate
                                        #   make_train_test_df_filling_not_well, 

from libs.tools import time_to_str, Specifity, calculate_acc_pre_rec
from arg_parser import get_args_parser

parser = argparse.ArgumentParser('Colorectal Cancer Segmentation & T-Stage', parents=[get_args_parser()])
args = parser.parse_args()

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


tensor_key = ['image', 'T_stage']


# ------------------- dataset -----------------------


# ------------------- net -----------------------
class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  #
    IMAGE_RGB_STD  = [0.229, 0.224, 0.225]  #
    # IMAGE_RGB_MEAN = [0.5, 0.5, 0.5] #[0.485, 0.456, 0.406]  #
    # IMAGE_RGB_STD  = [0.5, 0.5, 0.5] #[0.229, 0.224, 0.225]  #

    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class YNetCls3D(nn.Module):

    def load_pretrain(self,):
        pass

    def __init__(self, args):
        super(YNetCls3D, self).__init__()
        self.args = args
        self.output_type = ['inference', 'loss']
        self.backbone = Classification3D()


    def forward(self, batch):
        x = batch['image']

        # pdb.set_trace()

        e = self.backbone
        x = e(x)
        # x = F.adaptive_avg_pool3d(x, 1)
        # ---------------
        feature = x.reshape(-1)
        # clsout = self.clsout(feature)
        clsout = feature
        # --------------
        clsout = torch.sigmoid(clsout)
        clsout = torch.nan_to_num(clsout)
        return clsout

# ------------------- run_valid ---------------------

def run_valid():

    model = [
        [YNetCls3D, 128, f'/home/research/AMP_mysef_3D_Cls/result_0505-3D/res-0505-3D-experiment0/0.5-0.5-0-0.4/YNetCls_3D/128/fold-all/checkpoint/297_YNetCls_3D.pth'],
    ]

    num_net = len(model)
    print(f'num_net = {num_net}')

    net = []
    for i in range(num_net):
        Net, size, checkpoint = model[i]
        n = Net(size)
        f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        n.load_state_dict(f['state_dict'], strict=True)  # True
        n.cuda()
        n.eval()
        net.append(n)


    train_df, test_df = make_train_test_df(args)
    # _, df_filling      = make_train_test_df_filling_well(args)
    # _, df_unfilling    = make_train_test_df_filling_not_well()

    # test_dataset = CRCDataset(test_df, 'test')
    # test_dataset = CRCDataset(df_unfilling, 'test')
    # valid_dataset = CRCDataset(test_df, args, mode='test')
    test_dataset = CRCDataset(train_df, args, mode='train')


    test_loader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size = 8,
        drop_last =False,
        num_workers = 8,
        pin_memory = False,
        collate_fn = null_collate,
    )


    if 1:
        result = {
            'i' : [],
            'probability': [],
        }

        test_num = 0
    
        test_probability = []
        test_truth = []
    
        start_timer = time.time()
        # pdb.set_trace()
        for t, batch in enumerate(test_loader):
            # net.output_type = ['inference']
            # net.output_type = ['loss', 'inference']
            batch_size = len(batch['index'])
            
            # pdb.set_trace()

            for k in tensor_key: batch[k] = batch[k].cuda()
            # batch['image'] = batch['image'].cuda()

            # p = 0
            count = 0
            with torch.no_grad():
                with amp.autocast(enabled=True):
                    for i in range(num_net):
                        output = net[i](batch)
                        count += 1

                        # TTA
                        # batch['image'] = torch.flip(batch['image'], dims=[3, ])
                        # p += net[i](batch)
                        # count += 1

            # p = p / count

            # pdb.set_trace()
            
            test_num += batch_size
            test_truth.append(batch['T_stage'].data.cpu().numpy())
            test_probability.append(output.data.cpu().numpy())

            # print('\r %8d / %d  %s' % (test_num, len(test_loader.dataset), time_to_str(time.time() - start_timer, 'sec')), end='', flush=True)
            print('\r %8d / %d  %s' % (test_num, len(test_loader.dataset), time_to_str(time.time() - start_timer, 'sec')), end='', flush=True)

        print('')
    
    assert(test_num == len(test_loader.dataset))
    
    truth = np.concatenate(test_truth)
    probability = np.concatenate(test_probability)

    Auc = metrics.roc_auc_score(truth, probability)
    score = Specifity(truth, probability)
    metric = calculate_acc_pre_rec(truth, probability)

    truth = truth.reshape(-1)
    probability = probability.reshape(-1)

    # Acc            =     accuracy(probability, truth)
    Acc            =     metric[0]

    # Precision      =     precision(probability, truth)
    Precision      =     metric[1]

    # Recall         =     recall(probability, truth)
    Recall         =     metric[2]

    # Specificity    =     specificity(probability, truth)
    Specificity    =     score

    print('%4.4f, %4.4f, %4.4f, %4.4f, %4.4f, ' % (Auc, Acc, Precision, Recall, Specificity))

    # submid_df = test_df.loc[:,['img_name', 'pred_prob']]
    # submid_df.to_csv(f"./submission.csv", header=False, index=False, sep=' ')


            
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    run_valid()