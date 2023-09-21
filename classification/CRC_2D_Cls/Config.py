class CFG:

    def __init__(self):

        # self.image_size          = 1024
        # self.image_size          = 128
        self.image_size          = 256
        # self.image_size          = 384

        self.seed                = 982742

        # Linear decay LR
        self.start_lr            = 1e-5   
        self.min_lr              = 5e-7
        
        # stepLR
        self.step_size           = 10
        self.gamma               = 0.5


        self.train_batch_size    = 32
        self.val_batch_size      = 64
        self.num_workers         = 8

        self.initial_checkpoint  = None
        self.epochs              = 600
        # self.decay_epochs_linear        = 300 
        self.decay_epochs_steplr = 48 

        self.fold_num            = 5


        # loss
        # self.weight_bce         = 0.3
        # self.weight_l1s         = 0.3
        # self.weight_dice_loss   = 0.4

        self.weight_bce         = 0.5
        self.weight_l1s         = 0.5
        self.weight_dice_loss   = 0.0


        # # augmentation
        # self.scale               = [0.1, 0.3, 0.3]  # scale for data augmentation  0.1 0.3 0.3
        # self.rotate              = [30, 0, 0] # rotation angle for data augmentation 
        # self.translate           = [0, 0, 0]
        # self.gaussian_noise_std  = 0.02
        # self.additive_brightness_std = 0.7 
        # self.gamma_range         = [0.5, 1.6]

    def info_print(self):
        print( 
               'image_size:{}, \n',
               'start_lr:{},\n' 
               'train_batch_size:{},\n' 
               'val_batch_size:{},\n' 
               'num_workers:{},\n'
               'initial_checkpoint:{},\n' 
               'fold_num:{},\n'
               'epochs:{},'.format(
                                   self.image_size,
                                   self.start_lr, 
                                   self.train_batch_size, 
                                   self.val_batch_size,
                                   self.num_workers,
                                   self.initial_checkpoint,
                                   self.fold_num,
                                   self.epochs))