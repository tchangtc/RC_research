import pdb
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from skimage.measure import find_contours, label, regionprops
from .augmentation_img_mask import do_random_flip, do_elastic_transform, do_random_rotate, do_random_affine, do_random_stretch, do_random_noise, do_random_contrast, do_random_cutout
from torch.utils.data import Dataset, Sampler, DataLoader, SequentialSampler, RandomSampler
from Config import CFG

# # root_dir = './'
cfg = CFG()
image_size = cfg.image_size
# image_size = 128
# image_size = 1024
# image_dir = f'/home/workspace/KData/data/images_as_pngs_512/train_images_processed_512' #512x512
# image_dir = f'/home/project/AMP_mysef_2D_Cls/data/png_img_Tr_nonzero'
#clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
image_tr_dir = f'/home/workspace/AMP_mysef_2D_Cls/data/png_img_Tr_nonzero'
mask_tr_dir  = f'/home/workspace/AMP_mysef_2D_Cls/data/png_mask_Tr_nonzero'

image_ts_dir = f'/home/workspace/AMP_mysef_2D_Cls/data/png_img_Ts_nonzero'
mask_ts_dir  = f'/home/workspace/AMP_mysef_2D_Cls/data/png_mask_Ts_nonzero'


def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes


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


def make_train_test_df():
    # train_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_train_img_mask_T.csv')
    train_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0330_train_img_mask_T.csv')
    train_patient_id = train_df.img_dir.unique()
    train_patient_id = sorted(train_patient_id)
    train_id = train_patient_id
    train_df = train_df[train_df.img_dir.isin(train_id)].reset_index(drop=True)

    # test_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_test_img_mask_T.csv')
    test_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0330_test_img_mask_T.csv')
    test_patient_id = test_df.img_dir.unique()
    test_patient_id = sorted(test_patient_id)
    test_id = test_patient_id
    test_df = test_df[test_df.img_dir.isin(test_id)].reset_index(drop=True)

    return train_df, test_df


def make_train_test_df_filling_well():
    # train_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_train_img_mask_T.csv')
    train_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0330_train_img_mask_T.csv')
    train_patient_id = train_df.img_dir.unique()
    train_patient_id = sorted(train_patient_id)
    train_id = train_patient_id
    train_df = train_df[train_df.img_dir.isin(train_id)].reset_index(drop=True)

    # test_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_test_img_mask_T.csv')
    test_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0410_train_img_mask_filling_well_T.csv')
    test_patient_id = test_df.img_dir.unique()
    test_patient_id = sorted(test_patient_id)
    test_id = test_patient_id
    test_df = test_df[test_df.img_dir.isin(test_id)].reset_index(drop=True)

    return train_df, test_df


def make_train_test_df_filling_not_well():
    # train_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_train_img_mask_T.csv')
    train_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0330_train_img_mask_T.csv')
    train_patient_id = train_df.img_dir.unique()
    train_patient_id = sorted(train_patient_id)
    train_id = train_patient_id
    train_df = train_df[train_df.img_dir.isin(train_id)].reset_index(drop=True)

    # test_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0327_test_img_mask_T.csv')
    test_df = pd.read_csv(f'/home/workspace/AMP_mysef_2D_Cls/MyDataset/0410_train_img_mask_filling_not_well_T.csv')
    test_patient_id = test_df.img_dir.unique()
    test_patient_id = sorted(test_patient_id)
    test_id = test_patient_id
    test_df = test_df[test_df.img_dir.isin(test_id)].reset_index(drop=True)

    return train_df, test_df


def read_data(dd, mode):
    # image = []
    # for t, d in df.iterrows():
        # m = cv2.imread(f'{image_dir}/{d.patient_id}_{d.image_id}.png', cv2.IMREAD_GRAYSCALE)
    if mode == 'train':
        image = cv2.imread(f'{image_tr_dir}/{dd.img_dir}/{dd.img_name}', cv2.IMREAD_UNCHANGED)
        mask  = cv2.imread(f'{mask_tr_dir}/{dd.img_dir}/{dd.img_name}', cv2.IMREAD_GRAYSCALE)
    
    if mode == 'test':
        image = cv2.imread(f'{image_ts_dir}/{dd.img_dir}/{dd.img_name}', cv2.IMREAD_UNCHANGED)
        mask  = cv2.imread(f'{mask_ts_dir}/{dd.img_dir}/{dd.img_name}', cv2.IMREAD_GRAYSCALE)

    # print(f'{image_dir}/{d.img_dir}/{d.img_name}.png')
    # image.append(m)
    
    # image = np.stack(image)
    return image, mask

# def read_data(d):
#     # pdb.set_trace()
#     image0 = cv2.imread(f'{image_dir}/{d.patient_id}/{d.image_id}.png', cv2.IMREAD_GRAYSCALE)
#     # jpeg_stream = np.load(f'{image_dir}/{d.patient_id}/{d.image_id}.npz')
#     # image1 = cv2.imdecode(jpeg_stream['data'], cv2.IMREAD_GRAYSCALE)
#     #if d.laterality=='R':
#     #    image = cv2.flip(image, 1)
#     #image = clahe.apply(image)
#     return image0

class CRCDataset(Dataset):
    def __init__(self, df, mode='train', transforms=None):
        df.loc[:, 'i'] = np.arange(len(df))

        self.length = len(df)
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __str__(self):
        num_patient = len(set(self.df.img_dir))
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

    def __getitem__(self, index):
        dd = self.df.iloc[index]

        image, mask = read_data(dd, self.mode)
        # pdb.set_trace()
        # pdb.set_trace()

        bboxes = mask_to_bbox(mask)
        x1 = bboxes[0][0]
        y1 = bboxes[0][1]
        x2 = bboxes[0][2]
        y2 = bboxes[0][3]

        cX = (x2 + x1) // 2
        cY = (y2 + y1) // 2
        Center = (cX, cY)
        Center
        dis = 50

        crop_img = image[y1 - dis: y2 + dis, x1 - dis: x2 + dis]
        crop_mask = mask[y1 - dis: y2 + dis, x1 - dis: x2 + dis]

        image = cv2.resize(crop_img, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(crop_mask, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)

        if 1: # debug
            cv2.imwrite('./img.png', image)
            cv2.imwrite('./mask.png', mask)

        # pdb.set_trace()

        assert np.unique(mask).all() in [0, 255]

        if 0:
            import matplotlib.pyplot as plt
            plt.imshow(image)
            plt.imshow(mask)
            #        
        # pdb.set_trace()
        
        image = image.astype(np.float32)/255
        mask  = mask/255

        if 1: 
            cv2.imwrite('./img1.png', image * 255)
            cv2.imwrite('./mask1.png', mask * 255)
    
        assert np.unique(mask).all() in [0, 1]

        # pdb.set_trace()

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        # pdb.set_trace()

        rr = {}
        rr['index'] = index
        rr['d'] = dd
        rr['patient_id'] = dd.img_dir  #
        rr['image'] = torch.from_numpy(image).float()
        rr['mask'] = torch.from_numpy(mask).float()

        # pdb.set_trace()
        
        rr['image'] = rr['image'].unsqueeze(0)
        rr['mask'] = rr['mask'].unsqueeze(0)

        # pdb.set_trace()

        rr['image'] = torch.cat([
            rr['image'],
            rr['mask'],
            torch.zeros([1, image_size, image_size]),
            # torch.zeros([1, image_size, image_size]),
        ], 0)

        # pdb.set_trace()

        rr['label'] = torch.FloatTensor([dd.T_Stage])

        return rr

tensor_key = ['image', 'label']
def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_key:
            v = torch.stack(v,0)
        d[k] = v
    
    # d['image']= d['image'].unsqueeze(1)
    d['label']= d['label'].reshape(-1)
    return d

# class BalanceSampler(Sampler):

#     def __init__(self, dataset, ratio=cfg.ratio):
#         # pdb.set_trace()
#         self.r = ratio-1
#         self.dataset = dataset
#         self.pos_index = np.where(dataset.df.cancer>0)[0]
#         self.neg_index = np.where(dataset.df.cancer==0)[0]

#         self.length = self.r*int(np.floor(len(self.neg_index)/self.r))

#     def __iter__(self):
#         # pdb.set_trace()
#         pos_index = self.pos_index.copy()
#         neg_index = self.neg_index.copy()
#         np.random.shuffle(pos_index)
#         np.random.shuffle(neg_index)

#         neg_index = neg_index[:self.length].reshape(-1,self.r)
#         pos_index = np.random.choice(pos_index, self.length//self.r).reshape(-1,1)

#         index = np.concatenate([pos_index,neg_index],-1).reshape(-1)
#         return iter(index)

#     def __len__(self):
#         return self.length

#################################################################################

def train_augment_v00a(image):
    image = do_random_flip(image) # hflip, vflip or both
    #image, target = do_random_hflip(image, target)

    if np.random.rand() < 0.7:
        for func in np.random.choice([
            lambda image : do_random_affine( image, degree=15, translate=0.1, scale=0.2, shear=10),
            lambda image : do_random_rotate(image,  degree=15),
            lambda image : do_random_stretch(image, stretch=(0.2,0.2)),
        ], 1):
            image = func(image)

    if np.random.rand() < 0.25:
        image = do_elastic_transform(
            image,
            alpha=image_size,
            sigma=image_size* 0.05,
            alpha_affine=image_size* 0.03
        )

    if np.random.rand() < 0.5:
        for func in np.random.choice([
            lambda image: do_random_contrast(image),
            lambda image: do_random_noise(image, m=0.08),
        ], 1):
            image = func(image)
            pass

    return image

def train_augment_v00(image, mask):
    image, mask = do_random_flip(image, mask) # hflip, vflip or both
    #image, target = do_random_hflip(image, target)

    if np.random.rand() < 0.7:
        for func in np.random.choice([
            lambda image, mask : do_random_affine(image, mask, degree=30, translate=0.1, scale=0.3, shear=20),
            lambda image, mask : do_random_rotate(image, mask,  degree=30),
            lambda image, mask : do_random_stretch(image, mask, stretch=(0.3,0.3)),
        ], 1):
            image, mask = func(image, mask)

    if np.random.rand() < 0.5:
        image, mask = do_elastic_transform(
            image, 
            mask,
            alpha=image_size,
            sigma=image_size* 0.05,
            alpha_affine=image_size* 0.03
        )
    if np.random.rand() < 0.5:
        image, mask = do_random_cutout(
            image, 
            mask,
            num_block=5,
            block_size=[0.1,0.3],
            fill='constant'
        )

    if np.random.rand() < 0.5:
        for func in np.random.choice([
            lambda image, mask: do_random_contrast(image, mask),
            lambda image, mask: do_random_noise(image, mask, m=0.1),
        ], 1):
            image, mask = func(image, mask)
            pass

    return image, mask



#################################################################################

def run_check_dataset():
    train_df, valid_df = make_train_test_df()
    dataset = CRCDataset(train_df, mode='train', transforms=train_augment_v00)
    print(dataset)

    for i in range(100):
        i = 0 #240*8+ i#np.random.choice(len(dataset))
        r = dataset[i]
        print(r['index'], 'id = ', r['patient_id'], '-----------')
        for k in tensor_key :
            v = r[k]
            print(k)
            print('\t', 'dtype:', v.dtype)
            print('\t', 'shape:', v.shape)
            if len(v)!=0:
                print('\t', 'min/max:', v.min().item(),'/', v.max().item())
                print('\t', 'is_contiguous:', v.is_contiguous())
                print('\t', 'values:')
                print('\t\t', v.reshape(-1)[:8].data.numpy().tolist(), '...')
                print('\t\t', v.reshape(-1)[-8:].data.numpy().tolist())
        print('')
        # if 1:
        #     image  = r['image'].data.cpu().numpy()

        #     image_show_norm('image', image)
        #     cv2.waitKey(0)


    loader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        # sampler=SequentialSampler(dataset),
        # sampler=BalanceSampler(dataset),
        batch_size=8,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=null_collate,
    )
    print(loader.batch_size, len(loader), len(dataset))
    print('')

    for t, batch in enumerate(loader):
        if t > 5: break
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

# def run_check_augment():
#     train_df, valid_df = make_fold()
#     dataset = RsnaDataset(train_df)
#     print(dataset)

#     #---------------------------------------------------------------
#     def augment(image):
#         # image, target = do_random_hflip(image, target)
#         #image, target = do_random_flip(image, target)

#         #image, target = do_random_affine( image, target, degree=10, translate=0.1, scale=0.2, shear=10)
#         #image, target = do_random_rotate(image, target, degree=45)
#         #image, target = do_random_rotate90(image, target)

#         #image, target = do_random_perspective(image, target, m=0.3)
#         #image, target = do_random_zoom_small(image, target)

#         #image = do_random_hsv(image, h=20, s=50, v=50)
#         # image = do_random_contrast(image)
#         # image = do_random_gray(image)
#         # image = do_random_guassian_blur(image, k=[3, 5], s=[0.1, 2.0])
#         # image = do_random_noise(image, m=0.08)
#         return image

#     for i in range(10):
#         #i = 2424 #np.random.choice(len(dataset))#272 #2627
#         print(i)
#         r = dataset[i]

#         image  = r['image'].data.cpu().numpy()
#         image_show_norm('image',image, min=0, max=1,resize=1)
#         #cv2.waitKey(0)

#         for t in range(100):
#             #image1 = augment(image.copy())
#             image1 = train_augment_v00(image.copy())
#             image_show_norm('image1', image1, min=0, max=1,resize=1)
#             cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    run_check_dataset()
    #run_check_augment()


# import copy
# import pdb
# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import SimpleITK as sitk
# import pandas as pd
# import SimpleITK as sitk
# from torch.utils.data import Dataset, DataLoader, SequentialSampler
# from sklearn.model_selection import KFold
# from .augmentation import gaussian_noise, brightness_additive, gamma, crop_3d, random_scale_rotate_translate_3d
# from Config import CFG

# cfg = CFG()

# # training_size = [32, 256, 256]
# training_size = cfg.training_size


# def make_fold(fold):

#     # pdb.set_trace()

#     # df = pd.read_csv('/home/project/AMP_mysef/MyDataset/cls_train.csv')
#     df = pd.read_csv('/home/project/AMP_mysef/MyDataset/cls_train_autoseg.csv')

#     num_fold = cfg.fold_num

#     kf = KFold(n_splits=num_fold, shuffle=True, random_state=27)

#     for f, (train_idx, val_idx) in enumerate(kf.split(df)):
#         df.loc[val_idx, 'fold'] = f

#     train_df = df[df.fold!=fold].reset_index(drop=True)
#     valid_df = df[df.fold==fold].reset_index(drop=True)

#     return train_df, valid_df



# class CRCDataset(Dataset):
#     def __init__(self, df, transforms=None):

#         self.df = df
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.df)
    
#     def __str__(self):

#         num_image = len(self.df)
#         string = ''
#         string += f'\tnum_image = {num_image}\n'

#         count = dict(self.df.T_stage.value_counts())
        
#         # print(count)
#         # print(len(count))
#         for k in [0, 1]:
#             string += f'\t\label{k} = {count[k]:5d} ({count[k]/len(self.df):0.3f})\n'

#         return string

#     def read_image(self, d):
#         itk_img = sitk.ReadImage(f'{d.img_path}')
#         return itk_img

#     def preprocess(self, itk_img):
#         img = sitk.GetArrayFromImage(itk_img)

#         max98 = np.percentile(img, 98)
#         img = np.clip(img, 0, max98)

#         z, y, x = img.shape
        
#         # pad if the image size is smaller than trainig size
#         # pdb.set_trace()
#         if z < training_size[0]:
#             if z % 2 == 0:
#                 # diff = (training_size[0]+2 - z) // 2
#                 diff = (training_size[0] - z) // 2
#                 img = np.pad(img, ((diff, diff), (0,0), (0,0)))

#             else:
#                 diff = (training_size[0] - z) // 2
#                 img = np.pad(img, ((diff, diff+1), (0,0), (0,0)))

#         if y < training_size[1]:
#             # diff = (training_size[1]+2 - y) // 2
#             diff = (training_size[1] - y) // 2
#             img = np.pad(img, ((0,0), (diff, diff), (0,0)))
        
#         if x < training_size[2]:
#             # diff = (training_size[2]+2 - x) // 2
#             diff = (training_size[2] - x) // 2
#             img = np.pad(img, ((0,0), (0,0), (diff, diff)))

#         # img = img / max98

#         def remove_background(img, size=training_size):
#             z, y, x = img.shape
#             if y > size[1]:
#                 img = img[:, y//2-size[1]//2:y//2+size[1]//2, :]

#             if x > size[2]:
#                 img = img[:, :, x//2-size[2]//2:x//2+size[2]//2]

#             if z > size[0]:
#                 img = img[z//2-size[0]//2:z//2+size[0]//2, :, :]
#             return img
    

#         img = remove_background(img, training_size)
        
        
#         img = img.astype(np.float32)

#         tensor_img = torch.from_numpy(img).float()
        
#         return tensor_img



#     def __getitem__(self, index):
        
#         # pdb.set_trace()
#         dd = copy.deepcopy(self.df.iloc[index])

#         image_itk = self.read_image(dd)

#         image_tensor = self.preprocess(image_itk)

#         image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
#         # image_tensor = image_tensor.unsqueeze(0)

#         # ---------------------------------------------------

#         if self.transforms is not None:
#             # # # Gaussian Noise
#             # # tensor_img = augmentation.gaussian_noise(tensor_img, std=self.args.gaussian_noise_std)

#             # # # Additive brightness
#             # # tensor_img = augmentation.brightness_additive(tensor_img, std=self.args.additive_brightness_std)

#             # # # gamma
#             # # tensor_img = augmentation.gamma(tensor_img, gamma_range=self.args.gamma_range, retain_stats=True)
            
#             # # tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
#             # # tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='random')

#             # # Gaussian Noise
#             # image = gaussian_noise(image_tensor, std = , )

#             # # Additive brightness
#             # image = brightness_additive(image_tensor, std = ,)

#             # # gamma
#             # image = gamma(image, gamma_range=, retain_stats=True)

#             # image = random_scale_rotate_translate_3d(image_tensor, scale=, rotate=, translate=)
#             # image = crop_3d(image_tensor, crop_size=, mode='random')

#             image_tensor = self.transforms(image_tensor)

#         # pdb.set_trace()
#         image_tensor = image_tensor.squeeze(0)
#         # ---------------------------------------------------
#         r = {}
#         r['index'] = index
#         r['d'] = dd
#         r['name'] = dd.img_name
#         r['image'] = image_tensor
#         r['label'] = torch.FloatTensor([dd.T_stage])

#         return r

# tensor_key = ['image', 'label']
# def null_collate(batch):
#     d = {}
#     key = batch[0].keys()
#     for k in key:
#         v = [b[k] for b in batch]
#         if k in tensor_key:
#             v = torch.stack(v,0)
#         d[k] = v
    
#     d['label'] = d['label'].reshape(-1)
#     return d

# def train_augmentation_v0(image_tensor):
#     # # Gaussian Noise
#     # tensor_img = augmentation.gaussian_noise(tensor_img, std=self.args.gaussian_noise_std)

#     # # Additive brightness
#     # tensor_img = augmentation.brightness_additive(tensor_img, std=self.args.additive_brightness_std)

#     # # gamma
#     # tensor_img = augmentation.gamma(tensor_img, gamma_range=self.args.gamma_range, retain_stats=True)
    
#     # tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
#     # tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='random')

#     # Gaussian Noise
#     image = gaussian_noise(image_tensor, std = cfg.gaussian_noise_std, )

#     # Additive brightness
#     image = brightness_additive(image_tensor, std = cfg.additive_brightness_std, )

#     # gamma
#     image = gamma(image, gamma_range=cfg.gamma_range, retain_stats=True)

#     # scale, rotate, translate
#     image = random_scale_rotate_translate_3d(image_tensor, scale=cfg.scale, rotate=cfg.rotate, translate=cfg.translate)

#     # crop
#     image = crop_3d(image_tensor, crop_size=cfg.training_size, mode='random')

#     return image


# def run_check():
#     train_df, valid_df = make_fold(fold=4)
#     dataset = CRCDataset(train_df, transforms=None)
#     print(dataset)

#     for i in range(3):
#         # i = 0
#         r = dataset[i]
#         # print(r)
#         print(r['index'], 'id = ', r['name'], '-----------------')
#         for k in tensor_key:
#             v = r[k]
#             print(k)
#             print('\t', 'dtype', v.dtype)
#             print('\t', 'shape', v.shape)

#             if len(v) != 0:
#                 print('\t', 'min/max:', v.min().item(),'/', v.max().item())
#                 print('\t', 'is_contiguous:', v.is_contiguous())
#                 print('\t', 'values:')
#                 print('\t\t', v.reshape(-1)[:8].data.numpy().tolist(), '...')
#                 print('\t\t', v.reshape(-1)[-8:].data.numpy().tolist())

#         print(' ')

#     loader = DataLoader(
#         dataset,
#         sampler=SequentialSampler(dataset),
#         batch_size=8,
#         drop_last=False,
#         num_workers=0,
#         pin_memory=False,
#         worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
#         collate_fn=null_collate,
#     )

#     print(loader.batch_size, len(loader), len(dataset))
#     print('')

#     for t, batch in enumerate(loader):
#         # if t > 5: break
#         print('batch', t, '===========================')
#         print('index', batch['index'])

#         for k in tensor_key:
#             v = batch[k]
#             print(k)
#             print('\t', 'shape:', v.shape)
#             print('\t', 'dtype:', v.dtype)
#             print('\t', 'is_contiguous:', v.is_contiguous())
#             print('\t', 'value:')
#             print('\t\t', v.reshape(-1)[:8].data.numpy().tolist())
#             if k=='cancer':
#                 print('\t\tsum ', v.sum().item())

#         print('')

#     # return train_df, valid_df


# if 0:   # for debug
#     # train_df, valid_df = run_check()
#     run_check()
#     # print(valid_df)