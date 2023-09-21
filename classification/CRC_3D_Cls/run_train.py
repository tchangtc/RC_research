
import pdb
import os
import time
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.cuda.amp as amp
from torchvision import transforms
from engine import do_valid
from Model import build_model
from MyDataset import build_dataset
from arg_parser import get_args_parser
from MyDataset.dataset import CRCDataset, null_collate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from MyDataset.augmentation import GaussianNoiseTransform, GaussianBlurTransform, BrightnessTransform, GammaTransform, \
                                RandomCrop, CenterCrop, ContrastAugmentationTransform, MirrorTransform, RandomRotFlip
from libs.tools import time_to_str, get_learning_rate, set_all_random_seed
from MyMetrics.tools import get_cls_metric, np_binary_cross_entropy_loss

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import data_parallel


warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser('Colorectal Cancer Segmentation & T-Stage', parents=[get_args_parser()])
args = parser.parse_args()


IDENTIFIER = 'YNetCls_3D'
is_amp = True
tensor_key = ['image', 'T_stage']
initial_checkpoint = args.initial_checkpoint


def main_loop(args):
    name = '0526-3D-fnw-mask-Bspline3'
    root_dir = args.save_root_dir + '_' + name

    fold_dir = root_dir + f'/res-{name}-experiment0/{args.bce_weight}-{args.l1s_weight}-{args.dice_weight}-{args.focal_weight}/{IDENTIFIER}/{args.image_size}/fold-all'
    
    start_lr   = args.start_lr 
    batch_size = args.train_batch_size 
    skip_save_epoch = 3
    num_epoch = args.epochs

    ## setup  ----------------------------------------
    for f in ['checkpoint','train','valid','backup'] : os.makedirs(fold_dir +'/'+f, exist_ok=True)

    # log = Logger()
    log = open(fold_dir+'/log.train.txt',mode='a')

    log.write(f'\t__file__ = {__file__}\n')
    log.write(f'\tfold_dir = {fold_dir}\n' )
    log.write('\n')

    writer = SummaryWriter(f'{fold_dir}')
    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    # train_df, valid_df = make_fold(fold)

    train_df, test_df = build_dataset(image_set='Train_Val', args=args)
    
    # train_dataset = CRCDataset(train_df, mode='train', transforms=train_augment_v00)      # augmentation TODO

    train_dataset = CRCDataset(train_df, args, mode='train', 
                               transforms=transforms.Compose([
                                GaussianNoiseTransform(noise_variance=(0, 0.07), p_per_sample=0.25),
                                GaussianBlurTransform(blur_sigma=(1, 2.5), different_sigma_per_channel=False, p_per_channel=0.25, p_per_sample=0.25),
                                BrightnessTransform(mu=0, sigma=1, per_channel=False, p_per_channel=0.25, p_per_sample=0.25),
                                GammaTransform(gamma_range=(0.25, 1), per_channel=False, p_per_sample=0.25),
                                RandomRotFlip(p_per_sample=0.25),
                                MirrorTransform(p_per_sample=0.25),
                        ]),
                            #    transforms=transforms.Compose([
                            #         GaussianNoiseTransform(noise_variance=(0, 0.07), p_per_sample=0.25),
                                    
                            #         GaussianBlurTransform(blur_sigma=(1, 2.5), p_per_channel=0.25, p_per_sample=0.25),
                                    
                            #         BrightnessTransform(mu=0, sigma=1, p_per_channel=0.25, p_per_sample=0.25),
                                    
                            #         GammaTransform(gamma_range=(0.25, 1), p_per_sample=0.25),
                                    
                            #         RandomRotFlip(),
                                    
                            #         MirrorTransform(p_per_sample=0.5),
                            #    ])
                            )

    valid_dataset = CRCDataset(test_df, args, mode='test')

    # pdb.set_trace()

    train_loader  = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        # sampler = SequentialSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = False,
        num_workers = args.num_workers,
        pin_memory  = False,
        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn = null_collate,
    )

    valid_loader = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = args.val_batch_size,
        drop_last   = False,
        num_workers = args.num_workers,
        pin_memory  = False,
        collate_fn = null_collate,
    )

    log.write(f'train_dataset : \n{train_dataset}\n')
    log.write(f'valid_dataset : \n{valid_dataset}\n')
    log.write('\n')

    print(f'train_dataset : \n{train_dataset}\n')
    print(f'valid_dataset : \n{valid_dataset}\n')


    log.write(f'** Config Setting **\n')
    log.write(f'start_lr    : {args.start_lr}\n')
    log.write(f'min_lr      : {args.decay_epochs_steplr}\n')
    # log.write(f'min_lr      : {cfg.min_lr}\n')
    log.write(f'epochs      : {args.epochs}\n')
    log.write(f'num_workers : {args.num_workers}\n')
    log.write(f'\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    scaler = amp.GradScaler(enabled = is_amp)
    # net = YNetCls().cuda()
    net = build_model(args)
    net.cuda()

    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        # start_iteration = 0
        # start_epoch = 0
        state_dict  = f['state_dict']
        net.load_state_dict(state_dict,strict=True)  #True

        print('initial_checkpoint is not None')

    else:
        start_iteration = 0
        start_epoch = 0
        net.load_pretrain()


    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('\n')

    
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr)
    # optimizer = torch.optim.SGD(net.parameters(), start_lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(net.parameters(), start_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=1e-4)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader)*cfg.decay_epochs_steplr, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epochs_steplr, gamma=0.5)

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('scheduler\n  %s\n'%(scheduler))
    log.write('\n')

    # pdb.set_trace()
    num_iteration = num_epoch*len(train_loader)
    iter_log   = int(len(train_loader) * skip_save_epoch) #479
    iter_valid = iter_log
    iter_save  = iter_log

    ## start training here! ##############################################
    # valid
    # [cancer_loss, auc, f1score, threshold, sensitivity, specificity, f1score_mean]

    log.write('** start training here! **\n')
    log.write('   batch_size = %d\n'%(batch_size))
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                           |------------------------- VALID -------------------------|---------------- TRAIN/BATCH ----------\n')
    log.write('rate     iter        epoch | loss    Auc    Acc    Precision    Recall    Specificity|    loss              | time           \n')
    log.write('--------------------------------------------------------------------------------------------------------------------------\n')
                                #   return [ loss, auc, recall, precision, accuracy, F2] 
                                #   return [ loss, Auc, Acc, Precision, Recall, Specificity] 
                                #   return [ loss, Auc, Acc, Precision, Recall, Specificity] 



                                #  [ loss, auc, recall, precision, accuracy, F2] 
                                #  return [loss,  metric, auc, recall, 0, 0] 

    # log.write('rate     iter        epoch | loss   auc    f1    thr   sensi   specificity   f1s_mean   recall| loss                 | time         \n')

    def message(mode='print'):
        asterisk = ' '
        if mode==('print'):
            loss = batch_loss
        if mode==('log'):
            loss = train_loss
            if (iteration % iter_save == 0): asterisk = '*'
        
            # '%4.3f  %4.3f  %4.4f  %4.3f  %4.3f  %4.3f| '%(*valid_loss,) + \

        text = \
            ('%0.2e   %08d%s %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
            '%4.4f  %4.4f  %4.4f  %4.4f  %4.4f  %4.4f  | '%(*valid_loss,) + \
            '%4.4f  %4.4f  %4.4f  %4.4f  | ' % (*loss,) + \
            '%s' % (time_to_str(time.time() - start_timer,'min'))
        
        return text

    #----
    valid_loss = np.zeros(6,np.float32)
    train_loss = np.zeros(4,np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0

    start_timer = time.time()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0

# def run_train():
    
    while iteration < num_iteration:
        writer.add_scalar('Train/lr', rate, epoch + 1)

        for t, batch in enumerate(train_loader):
            
            if iteration%iter_save==0:
                if epoch < skip_save_epoch:
                    n = 0
                else:
                    n = iteration

                if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, fold_dir + '/checkpoint/%03d_%s.pth' % (int(epoch + 1), IDENTIFIER)) 
                    pass
            
            
            if (iteration > 0 and iteration%iter_valid==0): 
                valid_loss = do_valid(net, valid_loader, writer, epoch, args)  
                pass
            
            
            if (iteration%iter_log==0) or (iteration%iter_valid==0):
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')
                
                
            # learning rate schduler ------------
            # adjust_learning_rate(optimizer, scheduler(epoch))
            # adjust_learning_rate(optimizer, CosineAnnealingWarmRestarts(optimizer, 0, cfg.epochs,))

            # scheduler = CosineAnnealingLR(optimizer, cfg.epochs // 2)

            rate = get_learning_rate(optimizer) #scheduler.get_last_lr()[0] #get_learning_rate(optimizer)
            
            # one iteration update  -------------
            batch_size = len(batch['index'])
            for k in tensor_key: batch[k] = batch[k].cuda()

            net.train()
            net.output_type = ['loss', 'inference']
            if 1:
                with amp.autocast(enabled = is_amp):
                    # output = data_parallel(net,batch)
                    output = net(batch)
                    # loss0 = output['bce_loss'].mean()
                    loss1 = output['bce_loss'].mean()
                    loss2 = output['l1s_loss'].mean()
                    loss3 = output['dice_loss'].mean()

                    loss = loss1 * args.bce_weight + loss2 * args.l1s_weight + loss3 * args.dice_weight

                optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                scaler.step(optimizer)
                scaler.update()
                #
            
            # scheduler.step()

            # print statistics  --------
            batch_loss[:4] = [loss.item(), \
                                args.bce_weight*loss1.item(), \
                                    args.l1s_weight*loss2.item(), \
                                        args.dice_weight*loss3.item()]
            sum_train_loss += batch_loss
            sum_train += 1
            if t % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0
            
            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)
            epoch += 1 / len(train_loader)
            iteration += 1

        scheduler.step()
        writer.add_scalar('Train/train_loss : ', loss, epoch + 1)
        torch.cuda.empty_cache()
    log.write('\n')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    set_all_random_seed(seed=args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    main_loop(args)