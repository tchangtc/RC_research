import pdb
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from timm.models.efficientnet import efficientnet_b6, efficientnet_b5, efficientnet_b4, efficientnet_b2, efficientnet_b0
from timm.models.mobilenetv3 import mobilenetv3_small_100, tf_mobilenetv3_small_100


class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.5, 0.5, 0.5] #[0.485, 0.456, 0.406]  #
    IMAGE_RGB_STD  = [0.5, 0.5, 0.5] #[0.229, 0.224, 0.225]  #

    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class Mobile_V3(nn.Module):
    def load_pretrain(self, ):
        pass

    def __init__(self,):
        super(Mobile_V3, self).__init__()
        self.output_type = ['inference', 'loss']

        self.regularization_loss = 0.0
        self.lambda_ = 0.01

        self.rgb = RGB()
        self.encoder = tf_mobilenetv3_small_100(pretrained=True)
        #encoder_dim = [64, 256, 512, 1024, 2048]

        # self.mlp = XEmbedEffB2()

        self.cancer = nn.Linear(1024,1)         # b0
        # self.uncertainty = nn.Conv2d(32,len(agbm_percentile), kernel_size=1)


    def forward(self, batch):
        # pdb.set_trace()
        x = batch['image']
        x = x.expand(-1,3,-1,-1)
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

        cancer = cancer.reshape(-1)

        # pdb.set_trace()
        output = {}
        if  'loss' in self.output_type:
            # loss = F.binary_cross_entropy_with_logits(cancer, batch['label'])
            # loss = F.cross_entropy(cancer, batch['label'], label_smoothing=0.75)
            loss = nn.BCEWithLogitsLoss(reduction='mean')
            output['bce_loss'] = loss(cancer, batch['label'])

        if 'inference' in self.output_type:
            cancer = torch.sigmoid(cancer)
            cancer = torch.nan_to_num(cancer)
            output['label'] = cancer

        return output

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
    net = Mobile_V3().cuda()
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
