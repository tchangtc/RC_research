import os
import torch 
import torch.nn as nn

def do_swa(checkpoint):
	skip = ['relative_position_index', 'num_batches_tracked']
	K = len(checkpoint)
	swa = None
	for k in range(K):
		state_dict = torch.load(checkpoint[k], map_location=lambda storage, loc: storage)['state_dict']
		if swa is None:
			swa = state_dict
		else:
			for k, v in state_dict.items():
				# print(k)
				if any(s in k for s in skip): continue
				swa[k] += v
	for k, v in swa.items():
		if any(s in k for s in skip): continue
		swa[k] /= K
	return swa

def main():
    name = '0403'
    out_dir = f'/home/workspace/research/AMP_mysef_3D_Cls/result_0526-3D-fw-mask-Bspline3/res-0526-3D-fw-mask-Bspline3-experiment0/0.5-0.5-0-0/YNetCls_3D/128'
    

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    valid = {
        # all: [
        #     '297_YNetCls_3D.pth',
        #     '294_YNetCls_3D.pth',
        #     '291_YNetCls_3D.pth',
        #     '288_YNetCls_3D.pth',
        #     '285_YNetCls_3D.pth',
        #     '282_YNetCls_3D.pth',
        #     '279_YNetCls_3D.pth',
        # ],
        
        # all: [
        #     '298_YNetCls_3D.pth',
        #     '295_YNetCls_3D.pth',
        #     '292_YNetCls_3D.pth',
        #     '289_YNetCls_3D.pth',
        #     '286_YNetCls_3D.pth',
        #     '283_YNetCls_3D.pth',
        #     '280_YNetCls_3D.pth',
        # ],
        
    }

    for f, checkpoint in valid.items():
        if len(checkpoint) == 0: continue

        project_name = out_dir.split('/')[-1]
        fold_dir = out_dir + '/fold-all'

        checkpoint = [fold_dir + '/checkpoint/' + c for c in checkpoint]
        swa = do_swa(checkpoint)
        torch.save({
            'state_dict': swa,
            'swa': [c.split('/')[-1] for c in checkpoint],
        }, fold_dir + '/%s-fold-all-swa.pth' % (project_name))

        ## setup  ----------------------------------------
        submit_dir = fold_dir + '/valid/swa'
        os.makedirs(submit_dir, exist_ok=True)


if __name__ == '__main__':
    main()