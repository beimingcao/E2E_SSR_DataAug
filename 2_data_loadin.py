import time
import yaml
import os
import numpy as np
import torch
import pickle
from utils.database import HaskinsData_SSR
from torch.utils.data import Dataset, DataLoader
from utils.transforms import Pair_Transform_Compose
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_ATS_data
from utils.utils import prepare_Haskins_lists
from shutil import copyfile
from utils.transforms import  apply_EMA_MVN, apply_MVN
import random

import os
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/SSR_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    sel_sensors = config['articulatory_data']['sel_sensors']
    sel_dim = config['articulatory_data']['sel_dim'] 
    delta = config['articulatory_data']['delta']
    ema_dim = len(sel_sensors)*len(sel_dim)
    prepared_data_path = os.path.join(args.buff_dir, 'data')
    prepared_data_CV_path = os.path.join(args.buff_dir, 'data_CV')

    normalize_input = config['articulatory_data']['normalize_input']

    train_transforms = []
    valid_transforms = []
    test_transforms = []

    exp_train_lists, exp_valid_lists, exp_test_lists = prepare_Haskins_lists(args)

    for i in range(len(exp_test_lists)):
  #      CV = 'CV' + format(i, '02d')
        CV = exp_test_lists[i][0][:3]
        CV_data_dir = os.path.join(prepared_data_CV_path, CV)
        if not os.path.exists(CV_data_dir):
            os.makedirs(CV_data_dir)

        train_list = exp_train_lists[i]
        valid_list = exp_valid_lists[i]
        test_list = exp_test_lists[i]

        train_dataset = HaskinsData_SSR(prepared_data_path, train_list, ema_dim) 

  #      if normalize_input == True:
  #          EMA_all = {}
  #          i = 0
  #          for file_id, EMA, WAV in train_dataset:
  #              EMA_all[i] = EMA
  #              print(EMA.shape)
  #              i+=1
  #          EMA_block = np.concatenate([EMA_all[x] for x in EMA_all], 0)
  #          EMA_mean, EMA_std  = np.mean(EMA_block, 0), np.std(EMA_block, 0)

  #          train_transforms.append(apply_EMA_MVN(EMA_mean, EMA_std))
  #          valid_transforms.append(apply_EMA_MVN(EMA_mean, EMA_std))
  #          test_transforms.append(apply_EMA_MVN(EMA_mean, EMA_std))

  #      train_dataset = HaskinsData_SSR(prepared_data_path, train_list, ema_dim, transforms = Pair_Transform_Compose(train_transforms))
  #      valid_dataset = HaskinsData_SSR(prepared_data_path, valid_list, ema_dim, transforms = Pair_Transform_Compose(valid_transforms))
  #      test_dataset = HaskinsData_SSR(prepared_data_path, test_list, ema_dim, transforms = Pair_Transform_Compose(test_transforms))
  
        train_dataset = HaskinsData_SSR(prepared_data_path, train_list, ema_dim, transforms = None)
        valid_dataset = HaskinsData_SSR(prepared_data_path, valid_list, ema_dim, transforms = None)
        test_dataset = HaskinsData_SSR(prepared_data_path, test_list, ema_dim, transforms = None)
        
        train_pkl_path = os.path.join(CV_data_dir, 'train_data.pkl')
        tr = open(train_pkl_path, 'wb')
        pickle.dump(train_dataset, tr)
        valid_pkl_path = os.path.join(CV_data_dir, 'valid_data.pkl')
        va = open(valid_pkl_path, 'wb')
        pickle.dump(valid_dataset, va)
        test_pkl_path = os.path.join(CV_data_dir, 'test_data.pkl')
        te = open(test_pkl_path, 'wb')
        pickle.dump(test_dataset, te)

        

