import random
import pdb
import torch.nn as nn
import os
import time
import torch
import torch.cuda.amp as amp
import numpy as np
from torch.optim import RAdam

from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from Model.model import YNetCls
# from Model.model_mobileNet_for_train import Mobile_V3
from Model.model import EffNet

from MyDataset.dataset_crc import CRCDataset, \
                                  make_train_test_df, null_collate, train_augment_v00

from libs.tools import Lookahead, dotdict, \
                       time_to_str, scheduler, adjust_learning_rate, get_learning_rate, \
                       calculate_metrics, get_steplr


from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import data_parallel
from sklearn import metrics
from Config import CFG

cfg = CFG()
# IDENTIFIER = 'YNetCls_2D'
# IDENTIFIER = 'Mobile_V3_tb'
# IDENTIFIER = 'EffB0'
IDENTIFIER = 'EffB0'
# IDENTIFIER = 'ResNet'
is_amp = True


tensor_key = ['image', 'label']

# fold_valid = f'/home/workspace/tian_chi/cls_frog/valid'

initial_checkpoint = cfg.initial_checkpoint

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def main_loop(fold):
    name = '0419'
    fold_dir = f'/home/workspace/AMP_mysef_2D_Cls/res-{name}-experiment0-{cfg.weight_bce}-{cfg.weight_l1s}-{cfg.weight_dice_loss}/{IDENTIFIER}/{cfg.image_size}/fold-{fold}'
    
    start_lr   = cfg.start_lr 
    batch_size = cfg.train_batch_size 
    skip_save_epoch = 3
    num_epoch = cfg.epochs

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
    train_df, test_df = make_train_test_df()
    train_dataset = CRCDataset(train_df, mode='train', transforms=train_augment_v00)      # augmentation TODO
    # train_dataset = CRCDataset(train_df)
    valid_dataset = CRCDataset(test_df, mode='test')

    # pdb.set_trace()

    train_loader  = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        # sampler = SequentialSampler(train_dataset),
        batch_size  = cfg.train_batch_size,
        drop_last   = True,
        num_workers = cfg.num_workers,
        pin_memory  = False,
        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn = null_collate,
    )

    valid_loader = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = cfg.val_batch_size,
        drop_last   = False,
        num_workers = cfg.num_workers,
        pin_memory  = False,
        collate_fn = null_collate,
    )


    log.write(f'fold = {fold}\n')
    log.write(f'train_dataset : \n{train_dataset}\n')
    log.write(f'valid_dataset : \n{valid_dataset}\n')
    log.write('\n')

    ## config -------------------------------------
    log.write(f'** Config Setting **\n')
    
    log.write(f'start_lr    : {cfg.start_lr}\n')
    log.write(f'min_lr      : {cfg.min_lr}\n')
    log.write(f'epochs      : {cfg.epochs}\n')
    log.write(f'num_workers : {cfg.num_workers}\n')
    log.write(f'\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')

    scaler = amp.GradScaler(enabled = is_amp)
    # net = YNetCls().cuda()
    net = EffNet().cuda()
    # net = Mobile_V3().cuda()

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

    ## optimiser ----------------------------------
    if 0: ##freeze
        for p in net.stem.parameters():   p.requires_grad = False
        pass

    def freeze_bn(net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                
    #freeze_bn(net)

    num_iteration = num_epoch*len(train_loader)
    iter_log   = int(len(train_loader) * skip_save_epoch) #479
    iter_valid = iter_log
    iter_save  = iter_log

    # optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)
    optimizer = torch.optim.Adam(net.parameters(), betas=(0.9, 0.99), eps=1e-7, weight_decay=0.01, lr=start_lr)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr)

    # scheduler = get_steplr(optimizer)
    # scheduler = StepLR(optimizer, step_size=len(train_loader), gamma=0.5)
    scheduler = StepLR(optimizer, step_size=cfg.decay_epochs_steplr, gamma=0.5)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader)*50, T_mult=2, last_epoch=-1)

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('optimizer\n  %s\n'%(scheduler))
    log.write('\n')


    ## start training here! ##############################################
    # valid
    # [cancer_loss, auc, f1score, threshold, sensitivity, specificity, f1score_mean]

    log.write('** start training here! **\n')
    log.write('   batch_size = %d\n'%(batch_size))
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                          |--------------------------------------- VALID -----------------------------------|--------------------- TRAIN/BATCH -------------\n')
    log.write('rate     iter       epoch | loss    auc    recall    precision    accuracy    Fhalf    F1    F2  dice  jac  | loss    loss1    loss2   loss3 |  time        \n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------------\n')
                                #     return [ loss, auc, recall, precision, accuracy, Fhalf, F1, F2] 
                                #     return [ loss, auc, recall, precision, accuracy, Fhalf, F1, F2, dice, jac]

                                #     return [ loss, auc, recall, precision, accuracy, F2] 
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
            ('%0.2e  %07d%s  %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
            '%4.4f  %4.4f  %4.4f  %4.4f  %4.4f  %4.4f  %4.4f  %4.4f  %4.4f  %4.4f | '%(*valid_loss,) + \
            '%4.4f  %4.4f  %4.4f %4.4f | ' % (*loss,) + \
            '%s' % (time_to_str(time.time() - start_timer,'min'))
        
        return text

    #----
    valid_loss = np.zeros(10,np.float32)
    train_loss = np.zeros(4,np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0

    start_timer = time.time()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0

# def run_train():
    regularization_loss = 0.0
    
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
            
            
            # if (iteration > num_epoch*len(train_loader) / 30 and iteration%iter_valid==0): 
            if (iteration > 0 and iteration%iter_valid==0): 
                valid_loss = do_valid(net, valid_loader, writer, epoch)  
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
                    # for param in net.parameters():
                    #     regularization_loss += torch.sum(abs(param))
                    # output = data_parallel(net,batch)
                    # loss0 = output['bce_loss'].mean() + 0.01 * regularization_loss
                    output = data_parallel(net,batch)
                    loss1 = output['bce_loss'].mean()
                    loss2 = output['l1s_loss'].mean()
                    loss3 = output['dice_loss'].mean()

                    loss = loss1 * cfg.weight_bce + loss2 * cfg.weight_l1s + loss3 * cfg.weight_dice_loss

                optimizer.zero_grad()
                # scaler.scale(loss1 * 0.5 + loss2 * 0.5).backward()
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                scaler.step(optimizer)
                scaler.update()
                #
            
            # scheduler.step()

            # print statistics  --------
            batch_loss[:4] = [loss.item(), \
                                cfg.weight_bce*loss1.item(), \
                                    cfg.weight_l1s*loss2.item(), \
                                        cfg.weight_dice_loss*loss3.item()]
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


@torch.no_grad()
def do_valid(net, valid_loader, writer, epoch): 
    
    valid_num = 0
    valid_loss = 0
    valid_probability = []
    valid_truth = []

    net = net.eval()
    start_timer = time.time()

    for t, batch in enumerate(valid_loader):
        
        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with amp.autocast(enabled = is_amp):
                
                batch_size = len(batch['index'])
                for k in ['image', 'label' ]: batch[k] = batch[k].cuda()
                
                output = data_parallel(net, batch) #net(input)#
                loss1  = output['bce_loss'].mean()
                loss2  = output['l1s_loss'].mean()
                loss3  = output['dice_loss'].mean() 


        valid_num += batch_size
        valid_loss += batch_size*(cfg.weight_bce*loss1+ cfg.weight_l1s*loss2+cfg.weight_dice_loss*loss3).item()
        valid_truth.append(batch['label'].data.cpu().numpy())
        valid_probability.append(output['label'].data.cpu().numpy())

        #---
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(time.time() - start_timer, 'sec')), end='', flush=True)
		#if valid_num==200*4: break

    assert(valid_num == len(valid_loader.dataset))
    truth = np.concatenate(valid_truth)
    probability = np.concatenate(valid_probability)

    # pred_T2 = probability[truth == 0]
    # pred_T3 = probability[truth == 1]
    # pdb.set_trace()
    # thres = np.percentile(pred_T2, np.arange(90, 100, 1))
    # mPN = np.mean(np.greater(pred_T3[:, np.newaxis], thres).mean(axis=0))

    loss = valid_loss/valid_num
    # pf1 = probabilistic_f1(truth, probability, beta=1)
    auc = metrics.roc_auc_score(truth, probability)

    score = calculate_metrics(truth, probability)
    recall    =     score[0]
    precision =     score[1]
    accuracy  =     score[2]
    Fhalf     =     score[3]
    F1        =     score[4]
    F2        =     score[5]
    # dice      =     score[6]
    # jac       =     score[7]

    # writer.add_scalar('Valid/pf1_score', pf1, epoch + 1)
    # writer.add_scalar('Valid/mPN[0.9,0.99]', mPN, epoch + 1)
    
    writer.add_scalar('Valid/auc', auc, epoch + 1)
    writer.add_scalar('Valid/loss', loss, epoch + 1)
    
    writer.add_scalar('Valid/recall', recall, epoch + 1)
    writer.add_scalar('Valid/accuracy', accuracy, epoch + 1)
    writer.add_scalar('Valid/precision', precision, epoch + 1)
    writer.add_scalar('Valid/Fhalf', Fhalf, epoch + 1)
    writer.add_scalar('Valid/F1', F1, epoch + 1)
    writer.add_scalar('Valid/F2', F2, epoch + 1)

    # writer.add_scalar('Valid/dice', dice, epoch + 1)
    # writer.add_scalar('Valid/jac', jac, epoch + 1)

    return [ loss, auc, recall, precision, accuracy, Fhalf, F1, F2, 0, 0]

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    set_all_random_seed(seed=cfg.seed)
    # for fold in range(0, 5):
    # main_loop(fold = fold)
    main_loop(fold = 100)
    # main_loop(fold = 1)
    # for fold in range(4, 5):
    # fold = 1