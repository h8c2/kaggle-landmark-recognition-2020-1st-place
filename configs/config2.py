
import os
import albumentations as A
abs_path = os.path.dirname(__file__)

# os.environ["CUDA_VISIBLE_DEVICES"]= "0"

args = {
      'model_path':'../models/',
    'data_path':'/kaggle/input/landmark-recognition-2021/',
    'data_path_valid':'/kaggle/input/google-landmark-2021-validation/',
    'valid_csv_fn':'valid.csv',
    'train_csv_fn':'train.csv',
    'checkpoint_path':'',
    
    'train_slice':[72000,140000],
    'valid_slice':[7000,14000],
    
    'gpus':1,
    'filter_warnings':True,
    'logger': 'tensorboard',
    'num_sanity_val_steps': 0,

    'distributed_backend': '',
    'channels_last':False,

    'gradient_accumulation_steps':1,
    'precision':32,
    'sync_batchnorm':False,
    
    'seed':1138,
    'num_workers':4,
    'save_weights_only':True,

    'resume_from_checkpoint': None,
    'pretrained_weights':None,

    'normalization':'imagenet',
    'crop_size':448,

    'backbone':'gluon_seresnext101_32x4d',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45,
    'arcface_m': 0.3,

    'p_trainable': True,

    'neck': 'option-D',
    'head':'arc_margin',

    'crit': "bce",
    'loss':'arcface',
    #'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm' :'batch',
    
    'optimizer': "sgd",
    'weight_decay':1e-4,
    'lr': 0.05,
    'batch_size': 12,
    'max_epochs': 4,
    'scheduler': {"method":"cosine","warmup_epochs": 1},
    

    'n_classes':81313,
    'data_frac':1.,

    'neptune_project':'xx/kaggle-landmark',
}

args['tr_aug'] = A.Compose([
    A.SmallestMaxSize(512),
    A.RandomCrop(height=args['crop_size'],width=args['crop_size'],p=1.),
    A.HorizontalFlip(p=0.5),
    ])

args['val_aug'] = A.Compose([
    A.SmallestMaxSize(512),
    A.CenterCrop(height=args['crop_size'],width=args['crop_size'],p=1.)
])
