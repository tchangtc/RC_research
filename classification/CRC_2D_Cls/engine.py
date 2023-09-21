import pdb
import torch
import numpy as np
import time
from sklearn import metrics
from timeit import default_timer as timer
from torch.nn.parallel import data_parallel
# from MyMetrics.metrics import accuracy, precision, recall, specificity
from libs.tools import time_to_str, Specifity, calculate_acc_pre_rec

is_amp = True

@torch.no_grad()
def do_valid(net, valid_loader, writer, epoch, args): 
    
    valid_num = 0
    valid_loss = 0
    valid_probability = []
    valid_truth = []

    valid_bce_loss  = 0.0
    bce_loss_value  = 0.0
    valid_l1s_loss = 0.0
    l1s_loss_value = 0.0
    valid_dice_loss  = 0.0
    dice_loss_value  = 0.0
    valid_focal_loss = 0.0
    focal_loss_value = 0.0

    net = net.eval()
    start_timer = time.time()
    # start_timer = timer

    for t, batch in enumerate(valid_loader):
        
        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled = is_amp):
                
                batch_size = len(batch['index'])
                for k in ['image', 'label' ]: batch[k] = batch[k].cuda()
                
                output = data_parallel(net, batch) #net(input)#
                if args.bce_loss:
                    bce_loss  = output['bce_loss'].mean()
                    bce_loss_value = bce_loss.item()
                else:
                    bce_loss_value = 0.0
                if args.l1s_loss:
                    l1s_loss  = output['bce_loss'].mean()
                    l1s_loss_value = l1s_loss.item()
                else:
                    l1s_loss_value = 0.0

                if args.focal_loss:
                    focal_loss = output['focal_loss'].mean()
                    focal_loss_value = focal_loss.item()
                else:
                    focal_loss_value = 0.0

                if args.dice_loss:
                    dice_loss = output['dice_loss'].mean()
                    dice_loss_value = dice_loss.item()
                else:
                    dice_loss_value = 0.0

        valid_num += batch_size
        
        valid_bce_loss   += batch_size * bce_loss_value
        valid_l1s_loss   += batch_size * l1s_loss_value
        valid_focal_loss += batch_size * focal_loss_value
        valid_dice_loss  += batch_size * dice_loss_value
        
        valid_truth.append(batch['label'].data.cpu().numpy())
        valid_probability.append(output['label'].data.cpu().numpy())

        #---
        # print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(timer() - start_timer, 'sec')), end='', flush=True)
        # pdb.set_trace()
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

    loss = (valid_bce_loss * args.bce_weight + 
            valid_l1s_loss * args.l1s_weight +  
            valid_focal_loss * args.focal_weight + 
            valid_dice_loss * args.dice_weight) / valid_num
    
    # pf1 = probabilistic_f1(truth, probability, beta=1)

    Auc = metrics.roc_auc_score(truth, probability)
    score = Specifity(truth, probability)
    metric = calculate_acc_pre_rec(truth, probability)

    truth = truth.reshape(-1)
    probability = probability.reshape(-1)

    # Acc            =     accuracy(probability, truth)
    Acc            =     metric[0]

    # Precision      =     precision(probability, truth)
    Precision      =     metric[1]

    # Recall         =     recall(probability, truth)
    Recall         =     metric[2]

    # Specificity    =     specificity(probability, truth)
    Specificity    =     score

    # F1        =     score[4]
    # F2        =     score[5]
    # dice      =     score[6]
    # jac       =     score[7]

    # writer.add_scalar('Valid/pf1_score', pf1, epoch + 1)
    # writer.add_scalar('Valid/mPN[0.9,0.99]', mPN, epoch + 1)
    
    writer.add_scalar('Valid/Loss', loss, epoch + 1)
    writer.add_scalar('Valid/Auc', Auc, epoch + 1)
    writer.add_scalar('Valid/Accuracy', Acc, epoch + 1)
    writer.add_scalar('Valid/Recall', Recall, epoch + 1)
    writer.add_scalar('Valid/Specificity', Specificity, epoch + 1)
    writer.add_scalar('Valid/Precision', Precision, epoch + 1)
    # writer.add_scalar('Valid/Fhalf', Fhalf, epoch + 1)
    # writer.add_scalar('Valid/F1', F1, epoch + 1)
    # writer.add_scalar('Valid/F2', F2, epoch + 1)
    # writer.add_scalar('Valid/dice', dice, epoch + 1)
    # writer.add_scalar('Valid/jac', jac, epoch + 1)

    return [ loss, Auc, Acc, Precision, Recall, Specificity] 

    # return [ loss    auc    acc    precision    Recall    Sensitivity  ]
