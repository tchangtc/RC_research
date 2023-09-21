import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nextvit import next_vit_base




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


class Net(nn.Module):
    def load_pretrain(self, ):
        pass

    def __init__(self,):
        super(Net, self).__init__()
        self.output_type = ['inference', 'loss']

        self.rgb = RGB()
        self.encoder = next_vit_base()
        # self.encoder = efficientnet_b2(pretrained=True)
        # self.encoder = efficientnet_b4(pretrained=True)
        # self.encoder = efficientnet_b6(pretrained=True)
        # self.encoder = seresnext50_32x4d(pretrained=True)
        #encoder_dim = [64, 256, 512, 1024, 2048]

        self.cancer = nn.Linear(1024, 1)

        # self.cancer = nn.Linear(1408, 1)
        # self.cancer = nn.Linear(1792, 1)
        # self.cancer = nn.Linear(2304, 1)
        # self.cancer = nn.Linear(2048, 1)
        # self.uncertainty = nn.Conv2d(32,len(agbm_percentile), kernel_size=1)


    def forward(self, batch):
        x = batch['image']
        x = x.expand(-1,3,-1,-1)
        x = self.rgb(x) #17, 3, 256, 256

        # pdb.set_trace()
        #------
        e = self.encoder
        # x = e.forward_features(x)
        x = e(x)
        # x = F.adaptive_avg_pool2d(x,1)
        # x = torch.flatten(x,1,3)
        #------

        feature = x
        cancer = self.cancer(feature)
        cancer = cancer.reshape(-1)

        output = {}
        if  'loss' in self.output_type:
            loss = F.binary_cross_entropy_with_logits(cancer, batch['cancer'])
            output['bce_loss'] = loss

        if 'inference' in self.output_type:
            cancer = torch.sigmoid(cancer)
            cancer = torch.nan_to_num(cancer)
            output['cancer'] = cancer

        return output



def run_check_net():

    h,w = 256, 256
    batch_size = 4

    # ---
    batch = {
        'image': torch.from_numpy(np.random.uniform(0,1,(batch_size,1,h,w))).float().cuda() ,
        'cancer': torch.from_numpy(np.random.choice(2,(batch_size))).float().cuda() ,
    }
    #batch = {k: v.cuda() for k, v in batch.items()}

    net = Net().cuda()
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
    run_check_net()