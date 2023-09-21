import pdb
import torch
import time
import numpy as np
import torch.cuda.amp as amp
from sklearn import metrics
from torch.nn.parallel import data_parallel
from libs.tools import time_to_str, Specifity, calculate_acc_pre_rec

is_amp = True

@torch.no_grad()
def do_valid(net, valid_loader, writer, epoch, args): 
    
    valid_num = 0
    valid_loss = 0
    valid_probability = []
    valid_truth = []

    net = net.eval()
    start_timer = time.time()

    # pdb.set_trace()

    for t, batch in enumerate(valid_loader):
        
        # pdb.set_trace()

        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with amp.autocast(enabled = is_amp):
                
                batch_size = len(batch['index'])
                for k in ['image', 'T_stage' ]: batch[k] = batch[k].cuda()
                
                # pdb.set_trace()
                # output = data_parallel(net, batch) #net(input)#
                output = net(batch)
                # loss0  = output['bce_loss'].mean()
                loss1  = output['bce_loss'].mean()
                loss2  = output['l1s_loss'].mean()
                loss3  = output['dice_loss'].mean() 


        valid_num += batch_size
        # valid_loss += batch_size*loss0.item()
        valid_loss += batch_size*(args.bce_weight*loss1+ args.l1s_weight*loss2+args.dice_weight*loss3).item()

        # pdb.set_trace()
        # pdb.set_trace()
        valid_truth.append(batch['T_stage'].data.cpu().numpy())
        valid_probability.append(output['T_stage'].data.cpu().numpy())

        #---
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(time.time() - start_timer, 'sec')), end='', flush=True)
		#if valid_num==200*4: break

    # pdb.set_trace()
    assert(valid_num == len(valid_loader.dataset))
    truth = np.concatenate(valid_truth)
    probability = np.concatenate(valid_probability)

    # pdb.set_trace()

    loss = valid_loss/valid_num
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

    # pdb.set_trace()
    writer.add_scalar('Valid/Loss', loss, epoch + 1)
    writer.add_scalar('Valid/Auc', Auc, epoch + 1)
    writer.add_scalar('Valid/Accuracy', Acc, epoch + 1)
    writer.add_scalar('Valid/Precision', Precision, epoch + 1)
    writer.add_scalar('Valid/Recall', Recall, epoch + 1)
    writer.add_scalar('Valid/Specificity', Specificity, epoch + 1)

    return [ loss, Auc, Acc, Precision, Recall, Specificity] 

    print([
        cancer_loss, 
        round(Auc, 4),
        round(Acc, 4),
        round(Precision, 4),
        round(Recall, 4),
        round(specificity, 4),
    ])




def metric_to_text(ink, label, mask):
	text = []

	p = ink.reshape(-1)
	t = label.reshape(-1)
	pos = np.log(np.clip(p,1e-7,1))
	neg = np.log(np.clip(1-p,1e-7,1))
	bce = -(t*pos +(1-t)*neg).mean()
	text.append(f'bce={bce:0.5f}')


	mask_sum = mask.sum()
	#print(f'{threshold:0.1f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}')
	text.append('p_sum  th   prec   recall   fpr   dice   score')
	text.append('-----------------------------------------------')
	for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
		p = ink.reshape(-1)
		t = label.reshape(-1)
		p = (p > threshold).astype(np.float32)
		t = (t > 0.5).astype(np.float32)

		tp = p * t
		precision = tp.sum() / (p.sum() + 0.0001)
		recall = tp.sum() / t.sum()

		fp = p * (1 - t)
		fpr = fp.sum() / (1 - t).sum()

		beta = 0.5
		#  0.2*1/recall + 0.8*1/prec
		score = beta * beta / (1 + beta * beta) * 1 / recall + 1 / (1 + beta * beta) * 1 / precision
		score = 1 / score

		dice = 2 * tp.sum() / (p.sum() + t.sum())
		p_sum = p.sum()/mask_sum

		# print(fold, threshold, precision, recall, fpr,  score)
		text.append( f'{p_sum:0.2f}, {threshold:0.2f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}')
	text = '\n'.join(text)
	return text