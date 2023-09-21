import pdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np

from sklearn import metrics
from timeit import default_timer as timer
from torch.utils.data import DataLoader, SequentialSampler
from timm.models.efficientnet import efficientnet_b0
from MyDataset.dataset_crc import CRCDataset, make_train_test_df_filling_well, make_train_test_df_filling_not_well, make_train_test_df, null_collate
from libs.tools import calculate_metrics

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

class EffNet(nn.Module):
    def load_pretrain(self, ):
        pass

    def __init__(self, size):

        self.size = size
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
        # output = {}

        
        # if  'loss' in self.output_type:
        #     # if 1:
        #     #     for param in EffNet.parameters():
        #     #         self.regularization_loss += torch.sum(abs(param))
        #     # loss = F.binary_cross_entropy_with_logits(cancer, batch['label'])
        #     # loss = F.cross_entropy(cancer, batch['label'], label_smoothing=0.75)
        #     # loss = criterion_cross_entropy(cancer, batch['label'])
        #     # nn.CrossEntropyLoss()
        #     loss1 = nn.BCEWithLogitsLoss(reduction='mean')
        #     # loss2 = nn.L1Loss(reduction='mean')
        #     loss2 = nn.SmoothL1Loss(reduction='mean')
        #     # nn.CrossEntropyLoss()
        #     loss3 = DiceLoss()

        #     output['bce_loss'] = loss1(cancer, batch['label'])  
        #     output['l1s_loss'] = loss2(cancer, batch['label'].long())
        #     output['dice_loss'] = loss3(cancer, batch['label'].long())


        # if 'inference' in self.output_type:
        cancer = torch.sigmoid(cancer)
        output = torch.nan_to_num(cancer.float()) 
        return output



# ------------------- run_valid ---------------------

def run_valid():

    model = [
        [EffNet, 256, f'/home/workspace/AMP_mysef_2D_Cls/res-0417-experiment4/EffB0/256/fold-100/checkpoint/099_EffB0.pth'],
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


    train_df, test_df = make_train_test_df()
    _, df_filling      = make_train_test_df_filling_well()
    _, df_unfilling    = make_train_test_df_filling_not_well()

    # test_dataset = CRCDataset(test_df, 'test')
    # test_dataset = CRCDataset(df_unfilling, 'test')
    test_dataset = CRCDataset(df_filling, 'test')

    test_loader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size = 16,
        drop_last =False,
        num_workers = 4,
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
    
        start_timer = timer()

        for t, batch in enumerate(test_loader):
            # net.output_type = ['inference']
            # net.output_type = ['loss', 'inference']
            batch_size = len(batch['index'])
            
            for k in ['image', 'label']: batch[k] = batch[k].cuda()
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
            test_truth.append(batch['label'].data.cpu().numpy())
            test_probability.append(output.data.cpu().numpy())

            print('\r %8d / %d  %s' % (test_num, len(test_loader.dataset), time_to_str(time.time() - start_timer, 'sec')), end='', flush=True)
        print('')
    
    assert(test_num == len(test_loader.dataset))
    
    truth = np.concatenate(test_truth)
    probability = np.concatenate(test_probability)

    auc = metrics.roc_auc_score(truth, probability)
    score = calculate_metrics(truth, probability)

    recall    =     score[0]
    precision =     score[1]
    accuracy  =     score[2]
    Fhalf     =     score[3]
    F1        =     score[4]
    F2        =     score[5]
    dice      =     score[6]
    jac       =     score[7]

    print('%4.4f, %4.4f, %4.4f, \
           %4.4f, %4.4f, %4.4f, \
           %4.4f, %4.4f, %4.4f, ' % (auc, recall, precision, \
                                     accuracy, Fhalf, F1, \
                                     F2, dice, jac))

    # submid_df = test_df.loc[:,['img_name', 'pred_prob']]
    # submid_df.to_csv(f"./submission.csv", header=False, index=False, sep=' ')


            
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    run_valid()