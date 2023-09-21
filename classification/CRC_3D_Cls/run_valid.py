import os
import pdb
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
from arg_parser import get_args_parser
from timeit import default_timer as timer
from timm.models.efficientnet import efficientnet_b0
from Model.ynet import Classification3D

from MyDataset.dataset import null_collate, CRCDataset, make_train_test_df_filling_well
from torch.utils.data import DataLoader, SequentialSampler
from libs.tools import calculate_acc_pre_rec


is_amp = True 
tensor_key = ['image', 'T_stage']

parser = argparse.ArgumentParser('Colorectal Cancer Segmentation & T-Stage', parents=[get_args_parser()])
args = parser.parse_args()


#########################################
#########        helper         #########
#########################################
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

class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

#########################################
#########         metrics       #########
#########################################
def np_binary_cross_entropy_loss(probability, truth):
	probability = probability.astype(np.float64)
	probability = np.nan_to_num(probability, nan=1, posinf=1, neginf=0)

	p = np.clip(probability,1e-5,1-1e-5)
	y = truth

	loss = -y * np.log(p) - (1-y)*np.log(1-p)
	loss = loss.mean()
	return loss

def get_f1score(probability, truth):
    f1score = []
    threshold = np.linspace(0, 1, 50)
    for t in threshold:
        predict = (probability > t).astype(np.float32)

        tp = ((predict>=0.5) & (truth>=0.5)).sum()
        fp = ((predict>=0.5) & (truth< 0.5)).sum()
        fn = ((predict< 0.5) & (truth>=0.5)).sum()

        r = tp/(tp+fn+1e-3)
        p = tp/(tp+fp+1e-3)
        f1 = 2*r*p/(r+p+1e-3)
        f1score.append(f1)
    f1score=np.array(f1score)
    return f1score, threshold


#########################################
##########         model       ##########
#########################################
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

class EffNetB0(nn.Module):
    def load_pretrain(self, ):
        pass

    def __init__(self, ):

        # self.size = size
        super(EffNetB0, self).__init__()

        self.output_type = ['inference', 'loss']

        self.rgb = RGB()
        self.encoder = efficientnet_b0(pretrained=True)
        #encoder_dim = [64, 256, 512, 1024, 2048]

        # self.cancer = nn.Linear(1408,1)         # b2
        self.cancer = nn.Linear(1280,1)         # b0

    def forward(self, batch):
        # pdb.set_trace()
        x = batch['image']
        # x = x.expand(-1,3,-1,-1)
        x = self.rgb(x) #17, 3, 256, 256
        
        #------
        e = self.encoder
        x = e.forward_features(x)

        # x = self.mlp(x)
        x = F.adaptive_avg_pool2d(x,1)
        x = torch.flatten(x,1,3)
        #------

        feature = x
        cancer = self.cancer(feature)
        cancer = cancer.reshape(-1)

        # if 'inference' in self.output_type:
        cancer = torch.sigmoid(cancer)
        output = torch.nan_to_num(cancer.float()) 
        return output


class YNetCls(nn.Module):

    def load_pretrain(self,):
        pass

    def __init__(self,):
        super(YNetCls, self).__init__()

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


        # pdb.set_trace()
        # --------------
        # output = {}
        # if  'loss' in self.output_type:
        #     # # loss = F.binary_cross_entropy_with_logits(clsout, batch['T_stage'])
        #     # loss = nn.BCEWithLogitsLoss(reduction='mean')
        #     # output['bce_loss'] = loss(clsout, batch['T_stage'])

        #     loss1 = nn.BCEWithLogitsLoss(reduction='mean')
        #     # loss2 = nn.L1Loss(reduction='mean')
        #     loss2 = nn.SmoothL1Loss(reduction='mean')
        #     # nn.CrossEntropyLoss()
        #     loss3 = DiceLoss()

        #     # pdb.set_trace()

        #     output['bce_loss'] = loss1(clsout, batch['T_stage'])  
        #     output['l1s_loss'] = loss2(clsout, batch['T_stage'].long())
        #     output['dice_loss'] = loss3(clsout, batch['T_stage'].long())

        # if 'inference' in self.output_type:
        clsout = torch.sigmoid(clsout)
        clsout = torch.nan_to_num(clsout)
        output = clsout

        return output
#######################################################################

# fold = 0
root_dir = '/home/workspace/research/AMP_mysef_3D_Cls'
out_dir  = root_dir + '/res/3D'
fold_dir = out_dir  + f'/fold-all'
initial_checkpoint =\
    '/home/workspace/research/AMP_mysef_3D_Cls/result_0526-3D-fw-mask-Bspline3/res-0526-3D-fw-mask-Bspline3-experiment0/0.5-0.5-0-0/YNetCls_3D/128/fold-all/checkpoint/298_YNetCls_3D.pth'
    # '/home/workspace/research/AMP_mysef_3D_Cls/result_0525-3D-265-mask-Bspline3/res-0525-3D-265-mask-Bspline3-experiment0/0.5-0.5-0-0/YNetCls_3D/128/fold-all/checkpoint/297_YNetCls_3D.pth'
    # '/home/workspace/research/AMP_mysef_3D_Cls/result_0526-3D-fnw-mask-Bspline3/res-0526-3D-fnw-mask-Bspline3-experiment0/0.5-0.5-0-0/YNetCls_3D/128/fold-all/checkpoint/298_YNetCls_3D.pth'

os.makedirs(fold_dir, exist_ok=True)

def run_valid():
    global fold_dir

    # log = Logger()
    log = open(fold_dir+'/log_submit.txt',mode='a')

    train_df, test_df = make_train_test_df_filling_well(args)
    valid_dataset = CRCDataset(test_df, args, 'test')

    valid_loader = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = 1,
        drop_last   = False,
        num_workers = 8,
        pin_memory  = False,
        collate_fn = null_collate,
    )
    log.write(f'valid_dataset : \n{valid_dataset}\n')

    #-----
    net = YNetCls().cuda()
    f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
    state_dict = f['state_dict']
    net.load_state_dict(state_dict, strict=True)  # True

    #-----

    valid_num = 0
    valid = dotdict(
        cancer = dotdict(
            truth=[],
            predict=[],
        ),
        
    )

    net = net.eval()
    start_timer = timer()
    for t, batch in enumerate(valid_loader):

        batch_size = len(batch['index'])
        for k in tensor_key: batch[k] = batch[k].cuda()

        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with amp.autocast(enabled = is_amp):
                output = net(batch)#data_parallel(net, batch) #
                #loss0  = output['vindr_abnormality_loss'].mean()

        valid_num += batch_size
        # pdb.set_trace()
        valid.cancer.truth.append(batch['T_stage'].data.cpu().numpy())
        valid.cancer.predict.append(output.data.cpu().numpy())
        #show_result(batch, output, wait=0)

        #---

        # batch['patient_id']
        print(batch['patient_id'], '======>', output)

        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)
        #if valid_num==200*4: break

    assert(valid_num == len(valid_loader.dataset))
    cancer_t   = np.concatenate(valid.cancer.truth)
    cancer_p   = np.concatenate(valid.cancer.predict)

    test_df.loc[:,'cancer_p'] = cancer_p
        
    if 1:
        os.makedirs(fold_dir + f'/valid', exist_ok=True)
        valid_loader.dataset.df.to_csv(f'{fold_dir}/valid/valid_df.csv',index=False)
        np.save(f'{fold_dir}/valid/cancer_t.npy',cancer_t)
        np.save(f'{fold_dir}/valid/cancer_p.npy',cancer_p)

    #------
    cancer_loss  = np_binary_cross_entropy_loss(cancer_p, cancer_t)

    fpr, tpr, thresholds = metrics.roc_curve(cancer_t, cancer_p)
    Auc = metrics.auc(fpr, tpr)

    f1score, threshold = get_f1score(cancer_p, cancer_t)
    i = f1score.argmax()
    f1score, threshold = f1score[i], threshold[i]
    threshold = 0.5
    pdb.set_trace()
    specificity = ((cancer_p<threshold) &((cancer_t<=0.5))).sum() / (cancer_t<=0.5).sum()
    sensitivity = ((cancer_p>=threshold)&((cancer_t>=0.5))).sum() / (cancer_t>=0.5).sum()


    #---
    # gb = test_df[['patient_id', 'laterality', 'cancer', 'cancer_p']].groupby(['patient_id', 'laterality']).mean()
    gb = test_df
    pdb.set_trace()
    # f1score_mean, threshold_mean = get_f1score(gb.cancer_p, gb.T_Stage)
    # i = f1score_mean.argmax()
    # f1score_mean, threshold_mean = f1score_mean[i], threshold_mean[i]

    # Auc = metrics.roc_auc_score(gb.cancer_p, gb.T_Stage)

    metric = calculate_acc_pre_rec(cancer_t, cancer_p)

    # Acc            =     accuracy(probability, truth)
    Acc            =     metric[0]

    # Precision      =     precision(probability, truth)
    Precision      =     metric[1]

    # Recall         =     recall(probability, truth)
    Recall         =     metric[2]

    pdb.set_trace()
    # [ cancer_loss, auc, f1score, threshold, sensitivity, specificity, f1score_mean]
    
    # print(
    #     # [ cancer_loss, auc, f1score, threshold, sensitivity, specificity, f1score_mean]
    #     # [ cancer_loss, auc, f1score, threshold, sensitivity, specificity, f1score_mean]
    # )

    print([
        cancer_loss, 
        round(Auc, 4),
        round(Acc, 4),
        round(Precision, 4),
        round(Recall, 4),
        round(specificity, 4),
        round(f1score, 4)
    ])
        # Auc, 
        # Acc, 
        # Precision, 
        # Recall, 
        # specificity
    # [0.543689310144337, 0.8824, 0.8077, 0.9643, 0.75, 0.6667]
    # return [ loss,        Auc,    Acc, Precision, Recall, Specificity] 

def plot_auc(cancer_p, cancer_t):
	cancer_t = cancer_t.astype(int)
	pos, bin = np.histogram(cancer_p[cancer_t == 1], np.linspace(0, 1, 20))
	neg, bin = np.histogram(cancer_p[cancer_t == 0], np.linspace(0, 1, 20))
	pos = pos / (cancer_t == 1).sum()
	neg = neg / (cancer_t == 0).sum()
	print(pos)
	print(neg)
	# plt.plot(bin[1:],neg, alpha=1)
	# plt.plot(bin[1:],pos, alpha=1)
	bin = (bin[1:] + bin[:-1]) / 2
	plt.bar(bin, neg, width=0.05, label='neg',alpha=0.5)
	plt.bar(bin, pos, width=0.05, label='pos',alpha=0.5)
	plt.legend()
	plt.show()

def run_more():
	valid_df = pd.read_csv(f'{fold_dir}/valid/valid_df.csv')

	cancer_t  = np.load(f'{fold_dir}/valid/cancer_t.npy', )
	cancer_p  = np.load(f'{fold_dir}/valid/cancer_p.npy', )

	valid_df.loc[:,'cancer_p'] = cancer_p
	valid_df.loc[:,'cancer_t'] = valid_df.T_Stage
	# valid_df = valid_df[valid_df.site_id==1].reset_index(drop=True)

	#---

	f1score, threshold = get_f1score(valid_df.cancer_p, valid_df.T_Stage)
	i = f1score.argmax()
	f1score, threshold = f1score[i], threshold[i]

	fpr, tpr, thresholds = metrics.roc_curve(valid_df.T_Stage, valid_df.cancer_p)
	auc = metrics.auc(fpr, tpr)


	print('single image')
	print('f1score', f1score)
	print('threshold', threshold)
	print('auc', auc)
	print('')

	#---
	# gb = valid_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).mean()
	gb = valid_df
	f1score_mean, threshold_mean = get_f1score(gb.cancer_p, gb.T_Stage)
	i = f1score_mean.argmax()
	f1score_mean, threshold_mean = f1score_mean[i], threshold_mean[i]

	fpr, tpr, thresholds = metrics.roc_curve(gb.T_Stage, gb.cancer_p)
	auc_mean = metrics.auc(fpr, tpr)


	print('groupby mean')
	print('f1score_mean', f1score_mean)
	print('threshold_mean', threshold_mean)
	print('auc_mean', auc_mean)
	print('')

	plot_auc(gb.cancer_p, gb.T_Stage)

	# ---
	# gb = valid_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).max()
	gb = valid_df
	f1score_max, threshold_max = get_f1score(gb.cancer_p, gb.T_Stage)
	i = f1score_max.argmax()
	f1score_max, threshold_max = f1score_max[i], threshold_max[i]

	fpr, tpr, thresholds = metrics.roc_curve(gb.T_Stage, gb.cancer_p)
	auc_max = metrics.auc(fpr, tpr)

	print('groupby max')
	print('f1score_max', f1score_max)
	print('threshold_max', threshold_max)
	print('auc_mean', auc_max)
	print('')


# main #################################################################
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    run_valid()
    # run_more()

'''

'''
