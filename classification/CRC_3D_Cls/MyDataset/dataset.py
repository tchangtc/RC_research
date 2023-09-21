import copy
import pdb
# import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.measure import find_contours, label, regionprops
# from .augmentation_img_mask import do_random_flip, do_elastic_transform, do_random_rotate, do_random_affine, do_random_stretch, do_random_noise, do_random_contrast, do_random_cutout
from .augmentation import GaussianNoiseTransform, GaussianBlurTransform, BrightnessTransform, GammaTransform, \
                            RandomCrop, CenterCrop, ContrastAugmentationTransform, MirrorTransform
                            
from torch.utils.data import Dataset, Sampler, DataLoader, SequentialSampler, RandomSampler
# from Config import CFG
# # root_dir = './'
# cfg = CFG()

# image_size = 128
# padding_size1 = cfg.padding_size1
# cropping_size1 = cfg.cropping_size1

# padding_size2 = cfg.padding_size2
# cropping_size2 = cfg.cropping_size2
# distance = cfg.distance

padding_size1       = [456, 456, 456]
cropping_size1      = [456, 456, 456]
padding_size2       = [128, 128, 128]
cropping_size2      = [128, 128, 128]
distance            = [10, 25, 25]

target_spacing = (0.36, 0.36, 0.36)

########################################################################################
############ original ###################
# image_tr_dir = f'/home/workspace/AMP_mysef_3D_Cls/data/imagesTr'
# mask_tr_dir  = f'/home/workspace/AMP_mysef_3D_Cls/data/labelsTr'
# image_ts_dir = f'/home/workspace/AMP_mysef_3D_Cls/data/imagesTs'
# mask_ts_dir  = f'/home/workspace/AMP_mysef_3D_Cls/data/imagesTs'

############ processed ###################
# image_tr_dir = f'/home/workspace/research/AMP_mysef_3D_Cls/data/final_out_0522/imagesTr'
image_tr_dir = f'/home/workspace/research/AMP_mysef_3D_Cls/data/resampled_BS3_0525/imagesTr'

mask_tr_dir  = f'/home/workspace/research/AMP_mysef_3D_Cls/data/resampled_BS3_0525/labelsTr'

image_ts_dir = f'/home/workspace/research/AMP_mysef_3D_Cls/data/resampled_BS3_0525/imagesTs'

mask_ts_dir  = f'//home/workspace/research/AMP_mysef_3D_Cls/data/resampled_BS3_0525/labelsTs'


##########################################
train_dff = f'/home/research/AMP_mysef_3D_Cls/prepare/0505_3D_train_T.csv'
test_dff =  f'/home/research/AMP_mysef_3D_Cls/prepare/0505_3D_test_T.csv'

########################################################################################
info_dict = {
    'Spacing': 0,
    'Size': 0,
    'Origin': 0,
    'Direction': 0,
}

def make_fold(fold=0):
    df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_img_mask_nonzero_1.csv')
    patient_id = df.img_dir.unique()
    patient_id = sorted(patient_id)

    num_fold=5
    rs = np.random.RandomState(1234)
    rs.shuffle(patient_id)
    patient_id = np.array(patient_id)
    f = np.arange(len(patient_id))%num_fold
    train_id = patient_id[f!=fold]
    valid_id = patient_id[f==fold]

    train_df = df[df.img_dir.isin(train_id)].reset_index(drop=True)
    valid_df = df[df.img_dir.isin(valid_id)].reset_index(drop=True)
    return train_df, valid_df

def make_train_test_df(args):
    # train_df = pd.read_csv(train_dff)
    train_df = pd.read_csv(f'{args.root_dir}/prepare/0522_3D_train_T_fnw.csv')
    train_patient_id = train_df.img_name.unique()
    train_patient_id = sorted(train_patient_id)
    train_id = train_patient_id
    train_df = train_df[train_df.img_name.isin(train_id)].reset_index(drop=True)
    # pdb.set_trace()
    
    test_df = pd.read_csv(f'{args.root_dir}/prepare/0522_3D_test_T_fnw.csv')
    # test_df = pd.read_csv(f'{args.root_dir}/prepare/0505_3D_test_T.csv')
    test_patient_id = test_df.img_name.unique()
    test_patient_id = sorted(test_patient_id)
    test_id = test_patient_id
    test_df = test_df[test_df.img_name.isin(test_id)].reset_index(drop=True)

    return train_df, test_df

# def make_train_test_df_filling_well(args):
#     # train_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_train_img_mask_T.csv')
#                             # '/home/workspace/research/AMP_mysef_3D_Cls/prepare/0505_3D_train_T.csv'
#     train_df = pd.read_csv(f'{args.root_dir}/prepare/0505_3D_train_T.csv')

#     train_patient_id = train_df.img_name.unique()
#     train_patient_id = sorted(train_patient_id)
#     train_id = train_patient_id
#     train_df = train_df[train_df.img_name.isin(train_id)].reset_index(drop=True)

#     # test_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_test_img_mask_T.csv')
#     test_df = pd.read_csv(f'{args.root_dir}/prepare/0522_3D_test_T.csv')
#     test_patient_id = test_df.img_name.unique()
#     test_patient_id = sorted(test_patient_id)
#     test_id = test_patient_id
#     test_df = test_df[test_df.img_name.isin(test_id)].reset_index(drop=True)

#     return train_df, test_df

def make_train_test_df_filling_well(args):
    # train_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_train_img_mask_T.csv')
                            # '/home/workspace/research/AMP_mysef_3D_Cls/prepare/0505_3D_train_T.csv'
    # train_df = pd.read_csv(f'{args.root_dir}/prepare/0505_3D_train_T.csv')
    train_df = pd.read_csv(f'{args.root_dir}/prepare/0522_3D_train_T_fw.csv')
    # train_df = pd.read_csv(f'{args.root_dir}/prepare/0522_3D_train_T_fnw.csv')

    train_patient_id = train_df.img_name.unique()
    train_patient_id = sorted(train_patient_id)
    train_id = train_patient_id
    train_df = train_df[train_df.img_name.isin(train_id)].reset_index(drop=True)

    # test_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_test_img_mask_T.csv')
    
    # test_df = pd.read_csv(f'{args.root_dir}/prepare/0522_3D_test_T.csv')
    # test_df = pd.read_csv(f'{args.root_dir}/prepare/0522_3D_test_T_fnw.csv')
    test_df = pd.read_csv(f'{args.root_dir}/prepare/0522_3D_test_T_fw.csv')

    test_patient_id = test_df.img_name.unique()
    test_patient_id = sorted(test_patient_id)
    test_id = test_patient_id
    test_df = test_df[test_df.img_name.isin(test_id)].reset_index(drop=True)

    return train_df, test_df


class CRCDataset(Dataset):
    def __init__(self, df, args, mode='train', transforms=None):
        df.loc[:, 'i'] = np.arange(len(df))
        self.length = len(df)
        self.df = df
        self.args = args
        self.mode = mode
        self.transforms = transforms

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
        return self.length

    def read_data(self, dd, mode):
        # pdb.set_trace()
        if mode == 'train':
            image_path = f'{image_tr_dir}/{dd.img_name}.nii.gz'
            label_path = f'{mask_tr_dir}/{dd.img_name}.nii.gz'
            # print('Train: ', dd.img_name)

        if mode == 'test':
            image_path = f'{image_ts_dir}/{dd.img_name}.nii.gz'
            label_path = f'{mask_ts_dir}/{dd.img_name}.nii.gz'


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

    def ResampleCRCMRImage(self, imImage, imLabel, target_spacing):
        
        assert imImage.GetSpacing() == imLabel.GetSpacing()
        assert imImage.GetSize() == imLabel.GetSize()

        # pdb.set_trace()

        spacing = imImage.GetSpacing()
        origin = imImage.GetOrigin()


        npimg = sitk.GetArrayFromImage(imImage)
        nplab = sitk.GetArrayFromImage(imLabel)
        z, y, x = npimg.shape

        
        re_img_xy = self.ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
        re_lab_xy = self.ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)

        re_img_xyz = self.ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
        re_lab_xyz = self.ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)

        return re_img_xyz, re_lab_xyz
    
    def padding_or_cropping(self, image, label, padding_size, cropping_size):

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
        return img, lab

    def get_bbox_from_mask(self, mask, outside_value=0):
        mask_voxel_coords = np.where(mask != outside_value)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    def center_cropping(self, image, label):
        assert image.shape == label.shape

        bbox = self.get_bbox_from_mask(label, 0)
        z, x, y = bbox
        z1, z2 = z[0], z[1]
        x1, x2 = x[0], x[1]
        y1, y2 = y[0], y[1]


        image = image[z1 - distance[0] : z2 + distance[0], x1 - distance[1] : x2 + distance[1], y1 - distance[2] : y2 + distance[2]]
        label = label[z1 - distance[0] : z2 + distance[0], x1 - distance[1] : x2 + distance[1], y1 - distance[2] : y2 + distance[2]]

        assert image.shape == label.shape
        return image, label

    def __getitem__(self, index):
        # dd = self.df.iloc[index]

        dd = copy.deepcopy(self.df.iloc[index])

        # step 1
        image, label, info = self.read_data(dd, self.mode)

        # pdb.set_trace()
        # # step 2 
        # # resample
        #                                 #   ResampleCRCMRImage(self, imImage, imLabel, target_spacing=(1., 1., 1.)):
        # image_sampled, label_sampled = self.ResampleCRCMRImage(imImage=image, imLabel=label, target_spacing=(
        #                                                             round(info['Spacing'][1], 4), 
        #                                                             round(info['Spacing'][1], 4), 
        #                                                             round(info['Spacing'][1], 4)))

        # # pdb.set_trace()
        image, label = sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(label)

        # normalisation
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = image.astype(np.float32)


        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # pdb.set_trace()
        sample = {'image': image, 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        # pdb.set_trace()
        image = sample['image'].squeeze(0)
        label = sample['label'].squeeze(0)
        # # pdb.set_trace()



        # # step 3
        # # padding or cropping ==> [512, 512, 512]
        image_padding_cropping1, label_padding_cropping1 = self.padding_or_cropping(image, label, padding_size1, cropping_size1)

        # # step 4
        # # center cropping ==> [z-5 : z+5, x-20 : x+20, y-20 : y+20]
        image_center_cropping, label_center_cropping = self.center_cropping(image_padding_cropping1, label_padding_cropping1)

        # # step 5
        # # padding or cropping ==> [128, 128, 128]
        image_padding_cropping2, label_padding_cropping2 = self.padding_or_cropping(image_center_cropping, label_center_cropping, padding_size2, cropping_size2)
        # assert image_padding_cropping2.shape == label_padding_cropping2.shape

        # print(dd.img_name, 'Done')

        # step 6
        # normalization

        # max98 = np.percentile(image, 98)
        # image = np.clip(image, 0, max98)
        # image = image / max98

        image, label = image_padding_cropping2, label_padding_cropping2


        rr = {}
        rr['index'] = index
        rr['d'] = dd
        rr['patient_id'] = dd.img_name  #
        rr['image'] = torch.from_numpy(image).float()
        rr['label'] = torch.from_numpy(label).long()

        # pdb.set_trace()
        
        rr['image'] = rr['image'].unsqueeze(0)
        rr['label'] = rr['label'].unsqueeze(0)

        # pdb.set_trace()

        rr['image'] = torch.cat([
            rr['image'],
            rr['label'],
            # torch.zeros([1, image_size, image_size]),
        ], 0)

        # pdb.set_trace()

        rr['T_stage'] = torch.FloatTensor([dd.T_Stage])

        return rr

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

def build(image_set, args):
    if image_set == 'Train_Val':
        train_df, valid_df = make_train_test_df(args)
        # train_df, valid_df = make_train_test_df_filling_well(args)
        return train_df, valid_df 
    # elif image_set == 'All':
    #     train_df, valid_df = read_all_df(args)
    #     return train_df, valid_df
    else:
        raise ValueError('Please Train Frist')    

def train_augmentation():
    pass






