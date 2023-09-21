import pdb
import copy
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torchvision import transforms
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from augmentation import GaussianBlurTransform, GaussianNoiseTransform, RandomRotFlip, GammaTransform, BrightnessTransform, MirrorTransform

############ processed ###################
image_tr_dir = f'/home/workspace/research/AMP_mysef_3D_Cls/data/imagesTr'
mask_tr_dir  = f'/home/workspace/research/AMP_mysef_3D_Cls/data/labelsTr'
image_ts_dir = f'/home/workspace/research/AMP_mysef_3D_Cls/data/imagesTs'
mask_ts_dir  = f'/home/workspace/research/AMP_mysef_3D_Cls/data/labelsTs'

info_dict = {}
target_spacing = (0.36, 0.36, 0.36)
tensor_key = ['image', 'T_stage']

class Resample(Dataset):

    def __init__(self, df, mode, img_dir_tr, mask_dir_tr, img_dir_ts, mask_dir_ts, transforms) -> None:
        self.df = df
        self.mode = mode

        self.img_dir_tr = img_dir_tr
        self.mask_dir_tr = mask_dir_tr

        self.img_dir_ts = img_dir_ts
        self.mask_dir_ts = mask_dir_ts

        self.transforms = transforms


    def __len__(self,):
        return len(self.df)


    def read_data(self, dd, mode):
        if mode == 'train':
            image_path = f'{image_tr_dir}/{dd.img_name}.nii.gz'
            label_path = f'{mask_tr_dir}/{dd.img_name}.nii.gz'
            # print('Train: ', dd.img_name)

        if mode == 'test':
            image_path = f'{image_ts_dir}/{dd.img_name}.nii.gz'
            label_path = f'{mask_ts_dir}/{dd.img_name}.nii.gz'
            # print('Test: ', dd.img_name)


        image, label = sitk.ReadImage(image_path), sitk.ReadImage(label_path)

        assert image.GetSize() == label.GetSize()
        assert image.GetSpacing() == label.GetSpacing()
        assert image.GetOrigin() == label.GetOrigin()
        assert image.GetDirection() == label.GetDirection()

        info_dict['Spacing']    = image.GetSpacing()
        info_dict['Size']       = image.GetSize()
        info_dict['Origin']     = image.GetOrigin()
        info_dict['Direction']  = image.GetDirection()

        rr = {}
        rr['Spacing']   = info_dict['Spacing']
        rr['Size']      = info_dict['Size'] 
        rr['Origin']    = info_dict['Origin'] 
        rr['Direction'] = info_dict['Direction']

        # image, label = sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(label)

        return image, label, rr

    def ResampleXYZAxis(self, imImage, space=(1., 1., 1.), interp=sitk.sitkLinear):
        identity1 = sitk.Transform(3, sitk.sitkIdentity)
        sp1 = imImage.GetSpacing()
        sz1 = imImage.GetSize()

        sz2 = (int(round(sz1[0]*sp1[0]*1.0/space[0])), int(round(sz1[1]*sp1[1]*1.0/space[1])), int(round(sz1[2]*sp1[2]*1.0/space[2])))

        imRefImage = sitk.Image(sz2, imImage.GetPixelIDValue())
        imRefImage.SetSpacing(space)
        imRefImage.SetOrigin(imImage.GetOrigin())
        imRefImage.SetDirection(imImage.GetDirection())

        imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

        return imOutImage

    def ResampleLabelToRef(self, imLabel, imRef, interp=sitk.sitkLinear):
        identity1 = sitk.Transform(3, sitk.sitkIdentity)

        imRefImage = sitk.Image(imRef.GetSize(), imLabel.GetPixelIDValue())
        imRefImage.SetSpacing(imRef.GetSpacing())
        imRefImage.SetOrigin(imRef.GetOrigin())
        imRefImage.SetDirection(imRef.GetDirection())
            
        npLabel = sitk.GetArrayFromImage(imLabel)
        labels = np.unique(npLabel)
        resampled_nplabel_list = []
        for idx in labels:
            tmp_label = (npLabel == idx).astype(np.uint8)
            tmp_imLabel = sitk.GetImageFromArray(tmp_label)
            tmp_imLabel.CopyInformation(imLabel)
            tmp_resampled_Label = sitk.Resample(tmp_imLabel, imRefImage, identity1, interp)
            resampled_nplabel_list.append(sitk.GetArrayFromImage(tmp_resampled_Label))
        
        one_hot_resampled_label = np.stack(resampled_nplabel_list, axis=0)
        resampled_label = np.argmax(one_hot_resampled_label, axis=0)
        outLabel = sitk.GetImageFromArray(resampled_label.astype(np.uint8))
        outLabel.CopyInformation(imRef)

        return outLabel

    def ResampleCRCMRImage(self, imImage, imLabel, target_spacing, info):
        # pdb.set_trace()
        assert round(imImage.GetSpacing()[0], 3) == round(imLabel.GetSpacing()[0], 3)
        assert round(imImage.GetSpacing()[1], 3) == round(imLabel.GetSpacing()[1], 3)
        assert round(imImage.GetSpacing()[2], 3) == round(imLabel.GetSpacing()[2], 3)
        assert imImage.GetSize() == imLabel.GetSize()

        # pdb.set_trace()

        spacing   = imImage.GetSpacing()
        origin    = imImage.GetOrigin()
        size      = imImage.GetSize()
        direction = imImage.GetDirection()

        npimg = sitk.GetArrayFromImage(imImage)
        nplab = sitk.GetArrayFromImage(imLabel)
        z, y, x = npimg.shape

        
        # re_img_xy = self.ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline3)
        re_img_xy = self.ResampleXYZAxis(imImage, space=target_spacing, interp=sitk.sitkBSpline3)
        re_lab_xy = self.ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)

        # re_img_xyz = self.ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkBSpline3)
        re_img_xyz = self.ResampleXYZAxis(re_img_xy, space=target_spacing, interp=sitk.sitkBSpline3)
        re_lab_xyz = self.ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)

        # pdb.set_trace()

        info['Spacing']   = re_img_xyz.GetSpacing()
        info['Size']      = re_img_xyz.GetSize()
        info['Origin']    = re_img_xyz.GetOrigin()
        info['Direction'] = re_img_xyz.GetDirection()

        return re_img_xyz, re_lab_xyz, info


    def __getitem__(self, index):


        dd = copy.deepcopy(self.df.iloc[index])

        image, label, info = self.read_data(dd, self.mode)

        image_sampled, label_sampled, info = self.ResampleCRCMRImage(imImage=image, 
                                                                     imLabel=label, 
                                                                     target_spacing=target_spacing,
                                                                     info=info)

        image_sampled, label_sampled = sitk.GetArrayFromImage(image_sampled), sitk.GetArrayFromImage(label_sampled)
        image, label = image_sampled, label_sampled

        # image = np.expand_dims(image, axis=0)
        # label = np.expand_dims(label, axis=0)

        # sample = {'image': image, 'label': label}

        # if self.transforms is not None:
        #     sample = self.transforms(sample)
        
        # image = sample['image'].squeeze(0)
        # label = sample['label'].squeeze(0)

        # pdb.set_trace()

        rr = {}
        rr['index'] = index
        rr['d'] = dd
        rr['patient_id'] = dd.img_name  #
        rr['image'] = torch.from_numpy(image).float()
        rr['label'] = torch.from_numpy(label).long()

        
        rr['image'] = rr['image'].unsqueeze(0)
        rr['label'] = rr['label'].unsqueeze(0)


        rr['T_stage'] = torch.FloatTensor([dd.T_Stage])

        itk_image = sitk.GetImageFromArray(image)
        itk_label = sitk.GetImageFromArray(label)

        itk_image.SetSpacing(info['Spacing'])
        itk_label.SetSpacing(info['Spacing'])

        itk_image.SetDirection(info['Direction'])
        itk_label.SetDirection(info['Direction'])

        itk_image.SetOrigin(info['Origin'])
        itk_label.SetOrigin(info['Origin'])


        sitk.WriteImage(itk_image, '%s/%s.nii.gz'%(fold_dir_images, dd.img_name))
        sitk.WriteImage(itk_label, '%s/%s.nii.gz'%(fold_dir_labels, dd.img_name))
        print(dd.img_name + ' Done!')
        return rr

def null_collate(batch):
    dd = {}
    key = batch[0].keys()

    # pdb.set_trace()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_key:
            v = torch.stack(v,0)
        dd[k] = v
    
    # pdb.set_trace()
    # d['image']= d['image'].unsqueeze(1)
    dd['T_stage']= dd['T_stage'].reshape(-1)
    return dd

def make_train_test_df():
    # train_df = pd.read_csv(train_dff)
    train_df = pd.read_csv(f'/home/workspace/research/AMP_mysef_3D_Cls/prepare/0505_3D_train_T.csv')
    train_patient_id = train_df.img_name.unique()
    train_patient_id = sorted(train_patient_id)
    train_id = train_patient_id
    train_df = train_df[train_df.img_name.isin(train_id)].reset_index(drop=True)
    # pdb.set_trace()
    
    test_df = pd.read_csv(f'/home/workspace/research/AMP_mysef_3D_Cls/prepare/0522_3D_test_T.csv')
    # test_df = pd.read_csv(f'{args.root_dir}/prepare/0505_3D_test_T.csv')
    test_patient_id = test_df.img_name.unique()
    test_patient_id = sorted(test_patient_id)
    test_id = test_patient_id
    test_df = test_df[test_df.img_name.isin(test_id)].reset_index(drop=True)

    return train_df, test_df

def run_check_dataset():

    train_df, valid_df = make_train_test_df()
    # dataset = CRCDataset(train_df, args, 'train',)


    dataset = Resample( 
                    train_df, 
                    mode='train', 
                    # valid_df, 
                    # mode='test', 
                    img_dir_tr=image_tr_dir,
                    mask_dir_tr=mask_tr_dir,
                    img_dir_ts=image_ts_dir,
                    mask_dir_ts=mask_ts_dir,
                    transforms=None,
                    )
    


    # pdb.set_trace()

    print(dataset)

    # for i in range(len(dataset)):
    #     # i = 0 #240*8+ i#np.random.choice(len(dataset))
    #     r = dataset[i]
    #     print(r['index'], 'id = ', r['patient_id'], '-----------')
    #     for k in tensor_key :
    #         v = r[k]
    #         print(k)
    #         print('\t', 'dtype:', v.dtype)
    #         print('\t', 'shape:', v.shape)
    #         if len(v)!=0:
    #             print('\t', 'min/max:', v.min().item(),'/', v.max().item())
    #             print('\t', 'is_contiguous:', v.is_contiguous())
    #             print('\t', 'values:')
    #             print('\t\t', v.reshape(-1)[:8].data.numpy().tolist(), '...')
    #             print('\t\t', v.reshape(-1)[-8:].data.numpy().tolist())
    #     print('')

        # if 1:
        #     image  = r['image'].data.cpu().numpy()

        #     image_show_norm('image', image)
        #     cv2.waitKey(0)


    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=16,
        drop_last=False,
        num_workers=16,
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=null_collate,
    )
    # pdb.set_trace()

    print(loader.batch_size, len(loader), len(dataset))
    print('')

    for t, batch in enumerate(loader):
        # if t > 5: break
        # pdb.set_trace()
        print('batch ', t, '===================')
        print('index', batch['index'])
        for k in tensor_key:
            v = batch[k]
            print(k)
            print('\t', 'shape:', v.shape)
            print('\t', 'dtype:', v.dtype)
            print('\t', 'is_contiguous:', v.is_contiguous())
            print('\t', 'value:')
            print('\t\t', v.reshape(-1)[:8].data.numpy().tolist())
            if k=='cancer':
                print('\t\tsum ', v.sum().item())

        print('')


if 1:
    # for f in ['checkpoint','train','valid','backup'] : 
    import os
    fold_dir_images = '/home/workspace/research/AMP_mysef_3D_Cls/data/resampled/imagesTr'
    fold_dir_labels = '/home/workspace/research/AMP_mysef_3D_Cls/data/resampled/labelsTr'
    os.makedirs(fold_dir_images +'/', exist_ok=True)
    os.makedirs(fold_dir_labels +'/', exist_ok=True)

    run_check_dataset()
    pass