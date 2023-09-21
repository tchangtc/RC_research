import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
# from Config import CFG
# cfg = CFG()


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('Expected 5D input (got {}D input)'.format(input.dim()))
    
    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias, True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError('Expected valid activation')

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_channel=True):
    if double_channel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), act)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act)

    else:
        layer1 = LUConv(in_channel, 32 * (2 ** depth), act)
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, act)
    
    return nn.Sequential(layer1, layer2)


class ClassificationHead(nn.Module):
    def __init__(self, in_chan=1, act='prelu', lst_act='sigmoid', cls_classes=1):
        super(ClassificationHead, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, 1024, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(1024)

        if act == 'relu':
            self.activation = nn.ReLU(1024)
        elif act == 'prelu':
            self.activation = nn.PReLU(1024)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError('Expected valid activation')

        self.avgpool1 = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Linear(1024, cls_classes)

        # if lst_act == 'soft':
        #     self.lst_act = nn.Softmax(dim=1)
        # else:
        #     self.lst_act = nn.Sigmoid()

    def forward(self, x):
        
        # x [b, 512, 4, 48, 48]
        # pdb.set_trace()

        x = self.conv1(x)           # x [b, 1024, 4, 48, 48]
        x = self.bn1(x)             # x [b, 1024, 4, 48, 48]
        x = self.activation(x)      # x [b, 1024, 4, 48, 48]
        x = self.avgpool1(x)        # x [b, 1024, 1, 1, 1]
        
        x = torch.flatten(x, 1)     # x [b, 1024]

        x = F.dropout(x, p=0.4, training=self.training)

        x = self.fc1(x)             # x [b, 1]
        # cls_out = self.lst_act(x)   
        cls_out = x                 
        return cls_out


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans + outChans // 2, depth, act, double_channel=True)
    
    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)

        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.final_conv(x)
        out = self.sigmoid(x)
        return out

class UNet3D(nn.Module):
    # the number of convolutions  in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu'):
        super(UNet3D, self).__init__()

        self.down_tr64 = DownTransition(1, 0, act)      # 32 * 2 ** (0 + 1) = 32 * 2 = 64 
        self.down_tr128 = DownTransition(64, 1, act)    # 32 * 2 ** (1 + 1) = 32 * 4 = 128 
        self.down_tr256 = DownTransition(128, 2, act)   # 32 * 2 ** (2 + 1) = 32 * 8 = 256 
        self.down_tr512 = DownTransition(256, 3, act)   # 32 * 2 ** (3 + 1) = 32 * 16 = 512

        self.up_tr256 = UpTransition(512, 512, 2, act)  # 32 * 2 ** (2 + 1) = 32 * 8 = 256
        self.up_tr128 = UpTransition(256, 256, 1, act)  # 32 * 2 ** (1 + 1) = 32 * 4 = 128
        self.up_tr64 = UpTransition(128, 128, 0, act)   # 32 * 2 ** (0 + 1) = 32 * 2 = 64
        self.out_tr = OutputTransition(64, n_class)


    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64  =  self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)

        return self.out

class YNet3D(nn.Module):
    def __init__(self, n_class=1, act='relu', lst_act='soft', cls_classes=2):
        super(YNet3D, self).__init__()

        self.down_tr64 = DownTransition(1, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)
        
        self.up_tr256 = UpTransition(512, 512, 2, act)
        self.up_tr128 = UpTransition(256, 256, 1, act)
        self.up_tr64 = UpTransition(128, 128, 0, act)
        self.out_tr = OutputTransition(64, n_class)
        self.clc_out = ClassificationHead(512, act, lst_act, cls_classes)
    
    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64  = self.up_tr64(self.out_up_128, self.skip_out64)
        self.seg_out    = self.out_tr(self.out_up_64)
        self.cls_out    = self.clc_out(self.out512)

        return [self.seg_out, self.cls_out]


class Classification3D(nn.Module):
    def __init__(self, act='relu', lst_act='soft', cls_classes=1):
        super(Classification3D, self).__init__()

        self.down_tr64  = DownTransition(2, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)
        self.clc_out = ClassificationHead(512, act, lst_act, cls_classes)
    
    def forward(self, x):

        # pdb.set_trace()        
        self.out64, _ = self.down_tr64(x)
        self.out128, _ = self.down_tr128(self.out64)
        self.out256, _ = self.down_tr256(self.out128)
        self.out512, _ = self.down_tr512(self.out256)
        self.cls_out = self.clc_out(self.out512)

        return self.cls_out


