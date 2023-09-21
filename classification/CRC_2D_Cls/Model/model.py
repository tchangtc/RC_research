import pdb
import torch
import torch.nn as nn
import numpy as np

# from timm.models.resnet import *
from MyMetrics.tools import DiceBCELoss, DiceLoss
from .ynet_2D import Classification2D

import torch.nn.functional as F
from timm.models.efficientnet import efficientnet_b6, efficientnet_b5, efficientnet_b4, efficientnet_b2, efficientnet_b0
from timm.models.resnet import seresnext50_32x4d

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

class XEmbedEffB2(nn.Module):
    def __init__(self, ):
        super().__init__()

        # self.linear = nn.Sequential(
        #     nn.Linear(1408, 704, bias=True),
        #     nn.BatchNorm2d(704),
        #     nn.Dropout2d(0.5),

        #     nn.Linear(704, 352, bias=True),
        #     nn.BatchNorm2d(352),
        #     nn.Dropout2d(0.5),
        # )

        self.linear1 = nn.Linear(1408, 704)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(704, 352)
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(352, 1)
        self.dropout3 = nn.Dropout(0.1)



    
    def forward(self, x):
        # pdb.set_trace()
        x =  self.linear1(x)
        x = self.dropout1(x)
        x =  self.linear2(x)
        x = self.dropout2(x)
        x =  self.linear3(x)
        x = self.dropout3(x)
        return x

class EffNet(nn.Module):
    def load_pretrain(self, ):
        pass

    def __init__(self,):
        super(EffNet, self).__init__()
        self.output_type = ['inference', 'loss']

        self.regularization_loss = 0.0
        self.lambda_ = 0.01

        self.rgb = RGB()
        # self.encoder = efficientnet_b6(pretrained=True)
        # self.encoder = efficientnet_b5(pretrained=True)
        # self.encoder = efficientnet_b6(pretrained=True)
        # self.encoder = efficientnet_b4(pretrained=True)
        # self.encoder = efficientnet_b2(pretrained=True)
        self.encoder = efficientnet_b0(pretrained=True)
        # self.encoder = seresnext50_32x4d(pretrained=True)
        #encoder_dim = [64, 256, 512, 1024, 2048]

        # self.cancer = nn.Linear(2304 ,1)          # b6
        # self.cancer = nn.Linear(2304 ,1)          # b5
        # self.cancer = nn.Linear(1792,1)         # b4
        # self.mlp = XEmbedEffB2()

        # self.cancer = nn.Linear(1408,1)         # b2
        # self.cancer = nn.Linear(1408,1)         # b2
        self.cancer = nn.Linear(1280,1)         # b0
        # self.cancer = nn.Linear(2048, 1)         # resnet
        # self.uncertainty = nn.Conv2d(32,len(agbm_percentile), kernel_size=1)


    def forward(self, batch):
        # pdb.set_trace()
        x = batch['image']
        # x = x.expand(-1,3,-1,-1)
        x = self.rgb(x) #17, 3, 256, 256
        
        # pdb.set_trace()
        #------
        e = self.encoder
        x = e.forward_features(x)

        # x = self.mlp(x)

        # pdb.set_trace()
        # pdb.set_trace()
        x = F.adaptive_avg_pool2d(x,1)
        x = torch.flatten(x,1,3)
        #------

        feature = x
        cancer = self.cancer(feature)
        # cancer = self.mlp(feature)

        cancer = cancer.reshape(-1)

        # pdb.set_trace()
        output = {}
        if  'loss' in self.output_type:
            # if 1:
            #     for param in EffNet.parameters():
            #         self.regularization_loss += torch.sum(abs(param))
            # loss = F.binary_cross_entropy_with_logits(cancer, batch['label'])
            # loss = F.cross_entropy(cancer, batch['label'], label_smoothing=0.75)
            # loss = criterion_cross_entropy(cancer, batch['label'])
            # nn.CrossEntropyLoss()
            loss1 = nn.BCEWithLogitsLoss(reduction='mean')
            # loss2 = nn.L1Loss(reduction='mean')
            loss2 = nn.SmoothL1Loss(reduction='mean')
            # nn.CrossEntropyLoss()
            loss3 = DiceLoss()

            output['bce_loss'] = loss1(cancer, batch['label'])  
            output['l1s_loss'] = loss2(cancer, batch['label'].long())
            output['dice_loss'] = loss3(cancer, batch['label'].long())


        if 'inference' in self.output_type:
            cancer = torch.sigmoid(cancer)
            cancer = torch.nan_to_num(cancer)
            output['label'] = cancer
        return output


def criterion_cross_entropy(logit, label):
    # pdb.set_trace()
    label = label.long()
    #https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py
    log_softmax = F.log_softmax(logit, -1)
    loss = F.nll_loss(log_softmax, label)
    # pdb.set_trace()
    return loss

class YNetCls(nn.Module):

    def load_pretrain(self,):
        pass

    def __init__(self):
        super(YNetCls, self).__init__()

        self.output_type = ['inference', 'loss']

        self.backbone = Classification2D()


    def forward(self, batch):

        x = batch['image']

        e = self.backbone
        x = e(x)
        # x = F.adaptive_avg_pool3d(x, 1)

        # ---------------
        feature = x.reshape(-1)
        # clsout = self.clsout(feature)
        clsout = feature

        # --------------
        output = {}
        if  'loss' in self.output_type:
            loss = F.binary_cross_entropy_with_logits(clsout, batch['label'])
            output['bce_loss'] = loss

        if 'inference' in self.output_type:
            clsout = torch.sigmoid(clsout)
            clsout = torch.nan_to_num(clsout)
            output['label'] = clsout

        return output

# def criterion_cross_entropy(logit, mask, organ):
# 	logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
# 	batch_size, C, H, W = logit.shape
	
# 	label = mask.long() * organ.reshape(batch_size, 1, 1, 1)
	
# 	#https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py
# 	log_softmax = F.log_softmax(logit,1)
# 	loss = F.nll_loss(log_softmax,label.squeeze(1))
# 	return loss



def run_check_net():

    h,w = 256, 256
    batch_size = 4

    # ---
    batch = {
        'image': torch.from_numpy(np.random.uniform(0,1,(batch_size,1,h,w))).float().cuda() ,
        'label': torch.from_numpy(np.random.choice(2,(batch_size))).float().cuda() ,
    }
    #batch = {k: v.cuda() for k, v in batch.items()}

    # net = EffNet().cuda()
    net = YNetCls().cuda()
    # print(net)
    # torch.save({ 'state_dict': net.state_dict() },  'model.pth' )
    # exit(0)
    net.load_pretrain()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)

    print('batch')
    for k, v in batch.items():
        print('%32s :' % k, v.shape)

    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print('%32s :' % k, v.shape)
    for k, v in output.items():
        if 'loss' in k:
            print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    run_check_net()
