
import os
import albumentations as A
import cv2
import numpy as np

abs_path = os.path.dirname(__file__)

args = {
    'model_path':'../models/',
    'data_path':'/kaggle/input/landmark-recognition-2021/',
    'data_path_valid':'/kaggle/input/google-landmark-2021-validation/',
    'valid_csv_fn':'valid.csv',
    'train_csv_fn':'train.csv',
    'checkpoint_path':'',
    
    'train_slice':[280000,350000],
    'valid_slice':[28000,35000],
    
    'filter_warnings':True,
    'logger': 'tensorboard',
    'num_sanity_val_steps': 0,

    'gpus':1,
    'distributed_backend': "",
    'sync_batchnorm': True,

    'gradient_accumulation_steps': 1,
    'precision':32,

    'seed':1337,
    
    'drop_last_n': 0,
    
    'hardmining': False,

    'save_weights_only': False,
    'resume_from_checkpoint': None,

    'p_trainable': True,

    'normalization':'imagenet',

    'backbone':'tf_efficientnet_b3_ns',
    'embedding_size': 512,
    'pool': "gem",
    'arcface_s': 45,
    'arcface_m': 0.35,

    'head': 'arc_margin',
    'neck': 'option-D',

    'loss': 'arcface',
    'crit': "bce",
   # 'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm': "batch",
    
    'optimizer': "sgd",
    'lr': 0.06,
    'weight_decay': 1e-4,
    'batch_size': 8,

    'max_epochs': 3,

    'scheduler': {"method": "cosine", "warmup_epochs": 1},
    
    'pretrained_weights': None,

    'n_classes':81313,
    'data_frac':1.,
    
    'num_workers': 8,
    
    'crop_size': 600,

    'neptune_project':'xx/kaggle-landmark',
}


args['tr_aug'] = A.Compose([ A.LongestMaxSize(664,p=1),
                            A.PadIfNeeded(664, 664, border_mode=cv2.BORDER_CONSTANT,p=1),
                            A.RandomCrop(always_apply=False, p=1.0, height=args['crop_size'], width=args['crop_size']), 
                            A.HorizontalFlip(always_apply=False, p=0.5), 
                           ],
                            p=1.0
                            )

args['val_aug'] = A.Compose([ A.LongestMaxSize(664,p=1),
                             A.PadIfNeeded(664, 664, border_mode=cv2.BORDER_CONSTANT,p=1),
                            A.CenterCrop(always_apply=False, p=1.0, height=args['crop_size'], width=args['crop_size']), 
                            ], 
                            p=1.0
                            )

