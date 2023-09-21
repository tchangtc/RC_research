# class CFG:

#     def __init__(self):
#         ################## preprocess ################
#         # self.padding_size1       = [512, 512, 512]
#         # self.cropping_size1      = [512, 512, 512]
#         self.padding_size1       = [456, 456, 456]
#         self.cropping_size1      = [456, 456, 456]
#         self.padding_size2       = [128, 128, 128]
#         self.cropping_size2      = [128, 128, 128]
#         self.distance            = [5, 20, 20]

#         self.image_size          = 256

#         self.seed                = 982742


#         self.in_channels         = 1
#         # self.start_lr            = 1e-5    
#         self.start_lr            = 2e-5
#         # self.min_lr              = 1e-7

#         self.train_batch_size    = 8
#         self.val_batch_size      = 8

#         self.num_workers         = 16

#         self.initial_checkpoint  = None
#         self.epochs              = 300
#         self.decay_epochs_steplr = 25 
#         # self.decay_epochs        = 150 

#         self.weight_decay        = 0.001

#         self.fold_num            = 5

#         self.weight_bce         = 0.2
#         self.weight_l1s         = 0.4
#         self.weight_dice_loss   = 0.2


#     def info_print(self):
#         print( 
#                'image_size:{}, \n',
#                'start_lr:{},\n' 
#                'train_batch_size:{},\n' 
#                'val_batch_size:{},\n' 
#                'num_workers:{},\n'
#                'initial_checkpoint:{},\n' 
#                'fold_num:{},\n'
#                'epochs:{},'.format(
#                                    self.image_size,
#                                    self.start_lr, 
#                                    self.train_batch_size, 
#                                    self.val_batch_size,
#                                    self.num_workers,
#                                    self.initial_checkpoint,
#                                    self.fold_num,
#                                    self.epochs))