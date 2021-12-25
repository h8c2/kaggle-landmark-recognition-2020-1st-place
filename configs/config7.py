
import os
import albumentations as A
abs_path = os.path.dirname(__file__)

args = {
  'model_path':'../models/',
    'data_path':'E:/download/.kaggle/landmark-recognition-2021/',
    'data_path_valid':'E:/download/.kaggle/google-landmark-2021-validation/',
    'valid_csv_fn':'valid.csv',
    'train_csv_fn':'train.csv',
    'checkpoint_path':'E:/download/.kaggle/res2net101_26w_4s-02a759a1.pth',
    
    'train_slice':[480000,540000],
    'valid_slice':[60000,66000],
    
    'gpus':1,
    'filter_warnings':True,
    'logger': 'tensorboard',
    'num_sanity_val_steps': 50,

    'distributed_backend': '',

    'gradient_accumulation_steps':1,
    'precision':32,
    'sync_batchnorm':False,

    'seed':5553,
    'num_workers':4,
    'save_weights_only':True,

    'resume_from_checkpoint': None,
    'pretrained_weights':None,
    'normalization':'imagenet',
    'crop_size':512,

    'backbone':'res2net101_26w_4s',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45,
    'arcface_m': 0.4,

    'neck': 'option-D',
    'head':'arc_margin',
    'p_trainable':False,

    'crit': "bce",
    'loss':'arcface',
    #'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm' :'batch',
    
    'optimizer': "sgd",
    'weight_decay':1e-4,
    'lr': 0.06,
    'batch_size': 10,
    'max_epochs': 4,
    'scheduler': {"method":"cosine","warmup_epochs": 1},
    
    'n_classes':81313,
    'data_frac':1.,

    'neptune_project':'xx/kaggle-landmark',
}

args['tr_aug'] = A.Compose([A.Resize(height=544,width=672,p=1.),
    A.RandomCrop(height=args['crop_size'],width=args['crop_size'],p=1.),
    A.HorizontalFlip(p=0.5),
    ])

args['val_aug'] = A.Compose([A.Resize(height=544,width=672,p=1.),
    A.CenterCrop(height=args['crop_size'],width=args['crop_size'],p=1.)
])
