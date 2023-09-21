import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parallel import data_parallel
from monai.networks.nets.densenet import DenseNet121

class Net(nn.Module):

    def load_pretrain(self,):

        pass

    def __init__(self) -> None:
        super(Net, self).__init__()

        self.output_type = ['inference', 'loss']

        self.net = DenseNet121(
            spatial_dims = 3,
            in_channels = 1,
            out_channels = 1,
            pretrained = False
        )

    
    def forward(self, batch):

        x = batch['image']

        net = self.net

        x = net(x)

        clsout = x.reshape(-1)
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
    


def run_check_net():
    d, h, w = 32, 384, 384
    batch_size = 4

    # --------------------
    batch = {
        'image': torch.from_numpy(np.random.uniform(0,1,(batch_size,1,d,h,w))).float().cuda() ,
        'label': torch.from_numpy(np.random.choice(2,(batch_size))).float().cuda() ,
    }

    net = Net().cuda()

    net.load_pretrain()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            # output = net(batch)
            output = data_parallel(net, batch)

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
    


if __name__ == '__main__':
# if 0: # for debug
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    run_check_net()