import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .ynet import Classification3D
from Losses import DiceLoss


# class RGB(nn.Module):
#     IMAGE_RGB_MEAN = [0.5, 0.5, 0.5] #[0.485, 0.456, 0.406]  #
#     IMAGE_RGB_STD  = [0.5, 0.5, 0.5] #[0.229, 0.224, 0.225]  #

#     def __init__(self, ):
#         super(RGB, self).__init__()
#         self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
#         self.register_buffer('std', torch.ones(1, 3, 1, 1))
#         self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
#         self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

#     def forward(self, x):
#         x = (x - self.mean) / self.std
#         return x
    

class YNetCls(nn.Module):

    def load_pretrain(self,):
        pass

    def __init__(self, args):
        super(YNetCls, self).__init__()

        self.output_type = ['inference', 'loss']
        self.args = args
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
        output = {}
        if  'loss' in self.output_type:
            # # loss = F.binary_cross_entropy_with_logits(clsout, batch['T_stage'])
            # loss = nn.BCEWithLogitsLoss(reduction='mean')
            # output['bce_loss'] = loss(clsout, batch['T_stage'])

            loss1 = nn.BCEWithLogitsLoss(reduction='mean')
            # loss2 = nn.L1Loss(reduction='mean')
            loss2 = nn.SmoothL1Loss(reduction='mean')
            # nn.CrossEntropyLoss()
            loss3 = DiceLoss()

            # pdb.set_trace()

            output['bce_loss'] = loss1(clsout, batch['T_stage'])  
            output['l1s_loss'] = loss2(clsout, batch['T_stage'].long())
            output['dice_loss'] = loss3(clsout, batch['T_stage'].long())

        if 'inference' in self.output_type:
            clsout = torch.sigmoid(clsout)
            clsout = torch.nan_to_num(clsout)
            output['T_stage'] = clsout

        return output




def build_YNetCls(args):

    net = YNetCls(args)
    return net


def run_check_net():

    d, h, w = 128, 128, 128
    batch_size = 4
    # ---
    batch = {
        'image': torch.from_numpy(np.random.uniform(0,1,(batch_size,1,d,h,w))).float().cuda() ,
        'T_stage': torch.from_numpy(np.random.choice(2,(batch_size))).float().cuda() ,
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
# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     for i in range(100):
#         run_check_net()
