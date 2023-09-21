import pdb
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import SimpleITK as sitk
from multiprocessing import Pool
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from MyDataset.augmentation import GaussianBlurTransform, GaussianNoiseTransform, RandomRotFlip, GammaTransform, BrightnessTransform, MirrorTransform


info_dict = {
    'Spacing': 0,
    'Size': 0,
    'Origin': 0,
    'Direction': 0,
}

padding_size1       = [456, 456, 456]
cropping_size1      = [456, 456, 456]
padding_size2       = [128, 128, 128]
cropping_size2      = [128, 128, 128]
distance            = [10, 25, 25]

target_spacing = (0.36, 0.36, 0.36)

############################ original ##################################
image_tr_dir = f'/home/workspace/research/AMP_mysef_3D_Cls/data/imagesTr'
mask_tr_dir  = f'/home/workspace/research/AMP_mysef_3D_Cls/data/labelsTr'
image_ts_dir = f'/home/workspace/research/AMP_mysef_3D_Cls/data/imagesTs'
mask_ts_dir  = f'/home/workspace/research/AMP_mysef_3D_Cls/data/labelsTs'

train_dff = f'/home/workspace/research/AMP_mysef_3D_Cls/prepare/0505_3D_train_T.csv'
test_dff =  f'/home/workspace/research/AMP_mysef_3D_Cls/prepare/0522_3D_test_T.csv'



def make_train_test_df():
    train_df = pd.read_csv(train_dff)
    train_patient_id = train_df.img_name.unique()
    train_patient_id = sorted(train_patient_id)
    train_id = train_patient_id
    train_df = train_df[train_df.img_name.isin(train_id)].reset_index(drop=True)
    # pdb.set_trace()
    
    test_df = pd.read_csv(test_dff)
    test_patient_id = test_df.img_name.unique()
    test_patient_id = sorted(test_patient_id)
    test_id = test_patient_id
    test_df = test_df[test_df.img_name.isin(test_id)].reset_index(drop=True)

    return train_df, test_df


class CRCDataset(Dataset):

    def __init__(self, df, 
                 mode, 
                 img_dir_tr, mask_dir_tr, img_dir_ts, mask_dir_ts, 
                 padding_size1, padding_size2, cropping_size1, cropping_size2,
                 transforms
                 ) -> None:
        
        # self.train_df = train_df
        # self.test_df = test_df

        self.df = df

        self.mode = mode
        
        self.img_dir_tr = img_dir_tr
        self.mask_dir_tr = mask_dir_tr

        self.img_dir_ts = img_dir_ts
        self.mask_dir_ts = mask_dir_ts
        
        self.padding_size1 = padding_size1
        self.padding_size2 = padding_size2

        self.cropping_size1 = cropping_size1
        self.cropping_size2 = cropping_size2
        self.info = {
            'Spacing': 0,
            'Size': 0,
            'Origin': 0,
            'Direction': 0,
        }

        self.transforms = transforms


    def read_data(self, dd, mode):
        if mode == 'train':
            image_path = f'{self.img_dir_tr}/{dd.img_name}.nii.gz'
            label_path = f'{self.mask_dir_tr}/{dd.img_name}.nii.gz'
            # print('Train: ', dd.img_name)

        if mode == 'test':
            image_path = f'{self.img_dir_ts}/{dd.img_name}.nii.gz'
            label_path = f'{self.mask_dir_ts}/{dd.img_name}.nii.gz'
            # print('Test: ', dd.img_name)


        image, label = sitk.ReadImage(image_path), sitk.ReadImage(label_path)

        # pdb.set_trace()


        assert image.GetSize() == label.GetSize()
        
        assert round(image.GetSpacing()[0], 3) == round(label.GetSpacing()[0], 3)
        assert round(image.GetSpacing()[1], 3) == round(label.GetSpacing()[1], 3)
        assert round(image.GetSpacing()[2], 3) == round(label.GetSpacing()[2], 3)

        assert image.GetOrigin() == label.GetOrigin()

        image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        label.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        assert image.GetDirection() == label.GetDirection()

        # info_dict['Spacing']    = round(image.GetSpacing()[0], 4), round(image.GetSpacing()[1], 4), round(image.GetSpacing()[2], 4)
        info_dict['Spacing']    = target_spacing

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
    
    def __str__(self):
        num_patient = len(set(self.df.img_name))
        num_image = len(self.df)

        string = ''
        string += f'\tlen = {len(self)}\n'
        string += f'\tnum_patient = {num_patient}\n'
        string += f'\tnum_image = {num_image}\n'

        count = dict(self.df.T_Stage.value_counts())
        for k in [0,1]:
            string += f'\t\T_Stage{k} = {count[k]:5d} ({count[k]/len(self.df):0.3f})\n'
        return string

    def __len__(self):
        return len(self.df)

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
    

    def padding_or_cropping(self, image, label, padding_size, cropping_size, info):

        # padding_size = [128, 128, 128]
        # cropping_size = [128, 128, 128]
        # pdb.set_trace()

        assert image.shape == label.shape
        z, y, x = image.shape
        img, lab = image, label

        ##################### padding ######################
        # pad if the image size is smaller than trainig size
        if z < padding_size[0]:
            if z % 2 == 0:
                # diff = (training_size[0]+2 - z) // 2
                diff = (padding_size[0] - z) // 2
                img = np.pad(img, ((diff, diff), (0,0), (0,0)))
                lab = np.pad(lab, ((diff, diff), (0,0), (0,0)))

            else:
                diff = (padding_size[0] - z) // 2
                img = np.pad(img, ((diff, diff+1), (0,0), (0,0)))
                lab = np.pad(lab, ((diff, diff+1), (0,0), (0,0)))
                
        if y < padding_size[1]:
            if y % 2 == 0:
                diff = (padding_size[1] - y) // 2
                img = np.pad(img, ((0,0), (diff, diff), (0,0)))
                lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))
            
            else:
                diff = (padding_size[1] - y) // 2
                img = np.pad(img, ((0,0), (diff, diff+1), (0,0)))
                lab = np.pad(lab, ((0,0), (diff, diff+1), (0,0)))
              
        if x < padding_size[2]:
            if x % 2 == 0:
                diff = (padding_size[2] - x) // 2
                img = np.pad(img, ((0,0), (0,0), (diff, diff)))
                lab = np.pad(lab, ((0,0), (0,0), (diff, diff)))
            
            else:
                diff = (padding_size[2] - x) // 2
                img = np.pad(img, ((0,0), (0,0), (diff, diff+1)))
                lab = np.pad(lab, ((0,0), (0,0), (diff, diff+1)))
                
        ##################### cropping ######################
        if z > cropping_size[0]:
            size = cropping_size[0]
            img = img[z//2-size//2 : z//2+size//2, :, :]
            lab = lab[z//2-size//2 : z//2+size//2, :, :]

        if y > cropping_size[1]:
            size = cropping_size[1]
            img = img[:, y//2-size//2 : y//2+size//2, :]
            lab = lab[:, y//2-size//2 : y//2+size//2, :]

        if x > cropping_size[2]:
            size = cropping_size[2]
            img = img[:, :, x//2-size//2 : x//2+size//2]
            lab = lab[:, :, x//2-size//2 : x//2+size//2]

        assert img.shape == lab.shape     

        info['Size'] = img.shape

        return img, lab, info


    def get_bbox_from_mask(self, mask, outside_value=0):
        mask_voxel_coords = np.where(mask != outside_value)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


    def center_cropping(self, image, label, info):
        assert image.shape == label.shape

        bbox = self.get_bbox_from_mask(label, 0)
        z, x, y = bbox
        z1, z2 = z[0], z[1]
        x1, x2 = x[0], x[1]
        y1, y2 = y[0], y[1]


        image = image[z1 - distance[0] : z2 + distance[0], x1 - distance[1] : x2 + distance[1], y1 - distance[2] : y2 + distance[2]]
        label = label[z1 - distance[0] : z2 + distance[0], x1 - distance[1] : x2 + distance[1], y1 - distance[2] : y2 + distance[2]]

        assert image.shape == label.shape

        # info['Size'] = image.shape

        return image, label, info
    

    def __getitem__(self, index):
        # dd = self.df.iloc[index]
        # pdb.set_trace()
        dd = copy.deepcopy(self.df.iloc[index])

        # step 1
        image, label, info = self.read_data(dd, self.mode)


        # pdb.set_trace()
        # step 2 
        # resample
                                        #   ResampleCRCMRImage(self, imImage, imLabel, target_spacing=(1., 1., 1.)):
        image_sampled, label_sampled, info = self.ResampleCRCMRImage(imImage=image, 
                                                                     imLabel=label, 
                                                                     target_spacing=target_spacing,
                                                                     info=info)
                                                                    #  target_spacing=(
                                                                    # round(info['Spacing'][1], 4), 
                                                                    # round(info['Spacing'][1], 4), 
                                                                    # round(info['Spacing'][1], 4)), 

        # pdb.set_trace()
        image_sampled, label_sampled = sitk.GetArrayFromImage(image_sampled), sitk.GetArrayFromImage(label_sampled)
        

        # step 3 
        # DA
        image_sampled = np.expand_dims(image_sampled, axis=0)
        label_sampled = np.expand_dims(label_sampled, axis=0)

        # pdb.set_trace()
        sample = {'image': image_sampled, 'label': label_sampled}

        if self.transforms is not None:
            sample = self.transforms(sample)

        image_sampled = sample['image'].squeeze(0)
        label_sampled = sample['label'].squeeze(0)

        # pdb.set_trace()

        # step 4
        # padding or cropping ==> [512, 512, 512]
        # pdb.set_trace()
        image_padding_cropping1, label_padding_cropping1, info = self.padding_or_cropping(image_sampled, label_sampled, self.padding_size1, self.cropping_size1, info=info)

        # step 4
        # center cropping ==> [z-5 : z+5, x-20 : x+20, y-20 : y+20]
        # pdb.set_trace()
        image_center_cropping, label_center_cropping, info = self.center_cropping(image_padding_cropping1, label_padding_cropping1, info)

        # step 5
        # padding or cropping ==> [128, 128, 128]
        image_padding_cropping2, label_padding_cropping2, info = self.padding_or_cropping(image_center_cropping, label_center_cropping, self.padding_size2, self.cropping_size2, info=info)
        assert image_padding_cropping2.shape == label_padding_cropping2.shape

        image, label = image_padding_cropping2, label_padding_cropping2

        print(dd.img_name, 'Done')
        
        rr = {}
        rr['index'] = index
        rr['d'] = dd
        rr['patient_id'] = dd.img_name  #
        rr['image'] = torch.from_numpy(image).float()
        rr['label'] = torch.from_numpy(label).long()
        rr['T_stage'] = torch.FloatTensor([dd.T_Stage])



        itk_image = sitk.GetImageFromArray(image)
        itk_label = sitk.GetImageFromArray(label)

        itk_image.SetSpacing(info['Spacing'])
        itk_label.SetSpacing(info['Spacing'])

        itk_image.SetDirection(info['Direction'])
        itk_label.SetDirection(info['Direction'])

        itk_image.SetOrigin(info['Origin'])
        itk_label.SetOrigin(info['Origin'])

        # pdb.set_trace()

        sitk.WriteImage(itk_image, '%s/%s.nii.gz'%(fold_dir_images, dd.img_name))
        sitk.WriteImage(itk_label, '%s/%s.nii.gz'%(fold_dir_labels, dd.img_name))
        # sitk.WriteImage(image, )
        # sitk.WriteImage(image, )

        return rr
        
        # pass


tensor_key = ['image', 'T_stage']
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


def run_check_dataset():
    import argparse
    from arg_parser import get_args_parser
    parser = argparse.ArgumentParser('Colorectal Cancer Segmentation & T-Stage', parents=[get_args_parser()])
    args = parser.parse_args()

    train_df, valid_df = make_train_test_df()
    # dataset = CRCDataset(train_df, args, 'train',)

    dataset = CRCDataset(
        df=train_df,
        mode='train',

        # df=valid_df,
        # mode='test',

        img_dir_tr=image_tr_dir,
        mask_dir_tr=mask_tr_dir,
        img_dir_ts=image_ts_dir,
        mask_dir_ts=mask_ts_dir,
        padding_size1=padding_size1,
        padding_size2=padding_size2,
        cropping_size1=cropping_size1,
        cropping_size2=cropping_size2,
        # transforms=None
        transforms=transforms.Compose([
                        GaussianNoiseTransform(noise_variance=(0, 0.07), p_per_sample=1),
                        GaussianBlurTransform(blur_sigma=(1, 2.5), different_sigma_per_channel=False, p_per_channel=1, p_per_sample=1),
                        BrightnessTransform(mu=0, sigma=1, per_channel=False, p_per_channel=1, p_per_sample=1),
                        GammaTransform(gamma_range=(0.25, 1), per_channel=False, p_per_sample=1),
                        RandomRotFlip(p_per_sample=1),
                        MirrorTransform(p_per_sample=1),
                    ]),
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
    fold_dir_images = '/home/workspace/research/AMP_mysef_3D_Cls/data/final_out_0525/DA/images'
    fold_dir_labels = '/home/workspace/research/AMP_mysef_3D_Cls/data/final_out_0525/DA/labels'
    os.makedirs(fold_dir_images +'/', exist_ok=True)
    os.makedirs(fold_dir_labels +'/', exist_ok=True)

    run_check_dataset()
    pass