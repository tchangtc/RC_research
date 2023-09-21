import argparse

def get_args_parser():

	parser = argparse.ArgumentParser('Colorectal Cancer Segmentation & T-Stage', add_help=False)
	
	# ------------------------------- path setting -------------------------------
	parser.add_argument('--root_dir', default='/home/workspace/research/AMP_mysef_3D_Cls', type=str)
	parser.add_argument('--save_root_dir', default='result', type=str)
	
	parser.add_argument('--data_dir', default='/home/workspace/research/AMP_mysef_3D_Cls/data', type=str)

	parser.add_argument('--image_tr_dir', default='png_img_Tr_nonzero', type=str)

	parser.add_argument('--mask_tr_dir', default='png_mask_Tr_nonzero', type=str)

	parser.add_argument('--image_ts_dir', default='png_img_Ts_nonzero', type=str)

	parser.add_argument('--mask_ts_dir', default='png_mask_Ts_nonzero', type=str)
	
	
	parser.add_argument('--image_size', default='128', type=int) 


	parser.add_argument('--decay_epochs_steplr', default=48, type=int)

	# ------------------------------- model setting -------------------------------
	parser.add_argument('--model', default='YNetCls', type=str)
	parser.add_argument('--label_smoothing', default=0.75, type=float)
	parser.add_argument('--num_class', default=2, type=int)

	parser.add_argument('--aug', default='v8', type=str)
	parser.add_argument('--drop_rate', default=0.1, type=float)

	# ------------------------------- EMA related parameters -------------------------
	# parser.add_argument('--model_ema', type=str2bool, default=False)
	# parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
	# parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
	# parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')
	parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
	parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)",)
	parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",)


	# ------------------------------- BCE Loss -------------------------------
	parser.add_argument('--bce_loss', default=True, type=bool)
	parser.add_argument('--bce_weight', default=0.5, type=float)

	# ------------------------------- Dice Loss -------------------------------
	parser.add_argument('--l1s_loss', default=True, type=bool)
	parser.add_argument('--l1s_weight', default=0.5, type=float)

	# ------------------------------- Dice Loss -------------------------------
	parser.add_argument('--dice_loss', default=False, type=bool)
	parser.add_argument('--dice_weight', default=0, type=float)

	# ------------------------------- Focal Loss -------------------------------
	parser.add_argument('--focal_loss', default=False, type=bool)
	parser.add_argument('--focal_weight', default=0, type=float)
	parser.add_argument('--alpha', default=0.25, type=int)
	parser.add_argument('--gamma', default=2, type=int)

	

	parser.add_argument('--seed', default=42, type=int)

	parser.add_argument('--dataset', default='ColoRectal', type=str)
	parser.add_argument('--gpu_id', default='1', type=str)

	parser.add_argument('--start_lr', default=1e-5, type=float)
	parser.add_argument('--min_lr', default=1e-4, type=float)
	parser.add_argument('--train_batch_size', default=8, type=int)
	parser.add_argument('--val_batch_size', default=8, type=int)

	parser.add_argument('--num_workers', default=16, type=int)
	
	parser.add_argument('--initial_checkpoint', default=None, type=str)
	parser.add_argument('--epochs', default=300, type=int)
	parser.add_argument('--decay_epochs', default=40, type=int)

	parser.add_argument('--num_fold', default=5, type=int)
	parser.add_argument('--fold', default=0, type=int)
	

	return parser


def str2bool(v):
	"""
	Converts string to bool type; enables command line 
	arguments in the format of '--arg1 true --arg2 false'
	"""
	if isinstance(v, bool):
		return v
	
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'n', 'f', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected')