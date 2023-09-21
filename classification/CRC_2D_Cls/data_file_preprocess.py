        
import numpy as np
import pandas as pd
import os

###############################################################
##### part0: data preprocess
###############################################################
# def create_df_cls(path):
#     # # col_name = ['img_name', 'img_path', 'img_label']
#     # col_name = ['img_name', 'img_path']
#     # imgs_info = []      # img_name, img_path, img_label
#     # # for img_name in os.listdir(cfg.tampered_img_paths):
#     # for img_name in sorted(os.listdir(path)):
#     #     if img_name.endswith('.png'):
#     #         imgs_info.append([img_name[:], os.path.join(path, img_name)])
    
#     # # for img_name in os.listdir(path2):
#     # #     if img_name.endswith('.nii.gz'):
#     # #         imgs_info.append([img_name, os.path.join(path2, img_name)])
    
#     # imgs_info_array = np.array(imgs_info)
#     # df = pd.DataFrame(imgs_info_array, columns=col_name)
#     # df.to_csv('./cls_autoseg_demo.csv')
#     import os
def create_df_cls(path):
    # col_name = ['img_name', 'img_path', 'img_label']
    col_name = ['img_name', 'img_path']
    imgs_info = []      # img_name, img_path, img_label
    # for img_name in os.listdir(cfg.tampered_img_paths):
    for img_name in sorted(os.listdir(path)):
        if img_name.endswith('.png'):
            imgs_info.append([img_name, os.path.join(path, img_name)])
    
    # for img_name in os.listdir(path2):
    #     if img_name.endswith('.nii.gz'):
    #         imgs_info.append([img_name, os.path.join(path2, img_name)])
    
    imgs_info_array = np.array(imgs_info)
    df = pd.DataFrame(imgs_info_array, columns=col_name)
    df.to_csv('./cls_autoseg_demo.csv')


def create_df_test(path_test):
    col_name = ['img_name', 'img_path']
    imgs_info = []
    for img_name in sorted(os.listdir(path_test)):
        if img_name.endswith('.jpg'):
            imgs_info.append([img_name, os.path.join(path_test, img_name)])
    
    imgs_info_array = np.array(imgs_info)
    df = pd.DataFrame(imgs_info_array, columns=col_name)
    df.to_csv('./cls_test.csv')

    pass



if __name__ == '__main__':

    # path = f'/home/workspace/AMP_mysef/data/imagesTr/'
    # path = f'/home/project/AMP_mysef/data/imagesTr'
    path = f'/home/project/AMP_mysef/data/autoseg/imagesTr'
    create_df_cls(path)


    # path_test = f'/home/workspace/KData/test/imgs'
    # create_df_test(path_test)