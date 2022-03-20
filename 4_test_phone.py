import time
import yaml
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.models import MyLSTM, SpeechRecognitionModel
from utils.models import RegressionLoss
from utils.models import save_model
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import ndimage
from jiwer import wer
from utils.utils import data_processing_DeepSpeech, GreedyDecoder
import torch.nn.functional as F
from utils.transforms import apply_delta_deltadelta, Transform_Compose
import numpy as np
from utils.transforms import apply_MVN

def test_DeepSpeech(test_SPK, train_dataest, test_dataset, exp_output_folder, args):
    ### Dimension setup ###
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    sel_sensors = config['articulatory_data']['sel_sensors']
    sel_dim = config['articulatory_data']['sel_dim'] 
    delta = config['articulatory_data']['delta']
    d = 3 if delta == True else 1
    D_in = len(sel_sensors)*len(sel_dim)*d
    D_out = 41
    
    ### Model setup ###
    n_cnn_layers = config['deep_speech_setup']['n_cnn_layers']
    n_rnn_layers = config['deep_speech_setup']['n_rnn_layers']    
    rnn_dim = config['deep_speech_setup']['rnn_dim']
    stride = config['deep_speech_setup']['stride']
    dropout = config['deep_speech_setup']['dropout']
    normalize_input = config['articulatory_data']['normalize_input']
    
    test_transform = []
    test_transform.append(apply_delta_deltadelta())
    
    if normalize_input == True:
        norm_transform = [apply_delta_deltadelta()]
        norm_transforms_all = Transform_Compose(norm_transform)

        train_loader_norm = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = norm_transforms_all))

        EMA_all = {}
        i = 0
        for batch_idx, _data in enumerate(train_loader_norm):
            file_id, EMA, labels, input_lengths, label_lengths = _data 
            ema = EMA[0][0].T
            EMA_all[i] = ema
            i+=1

        EMA_block = np.concatenate([EMA_all[x] for x in EMA_all], 0)
        EMA_mean, EMA_std  = np.mean(EMA_block, 0), np.std(EMA_block, 0)
        
        test_transform.append(apply_MVN(EMA_mean, EMA_std))
    
    test_transforms_all = Transform_Compose(test_transform)
    
    ### Test ###
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = test_transforms_all))
                            
    model_out_folder = os.path.join(exp_output_folder, 'trained_models')
        
    SPK_model_path = os.path.join(model_out_folder)
    model_path = os.path.join(SPK_model_path, test_SPK + '_DS')
    model = SpeechRecognitionModel(n_cnn_layers, n_rnn_layers, rnn_dim, D_out, D_in, stride, dropout)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    pred = []
    label = []

    for batch_idx, _data in enumerate(test_loader):
        fid, ema, labels, input_lengths, label_lengths = _data 
        ema, labels = ema, labels

        output = model(ema)  # (batch, time, n_class)

        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)

        pred.append(' '.join(decoded_preds[0]))
        label.append(' '.join(decoded_targets[0]))
    
    error = wer(pred, label)
    return error




if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/SSR_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    data_path = os.path.join(args.buff_dir, 'data_CV')
    SPK_list = config['data_setup']['spk_list']

    results_all = os.path.join(args.buff_dir, 'results_all.txt')
    with open(results_all, 'w') as r:
        for test_SPK in SPK_list:
            data_path_SPK = os.path.join(data_path, test_SPK)
            
            tr = open(os.path.join(data_path_SPK, 'train_data.pkl'), 'rb') 
       
            train_dataset = pickle.load(tr)
            
            te = open(os.path.join(data_path_SPK, 'test_data.pkl'), 'rb')
            test_dataset = pickle.load(te)
        
            #avg_vacc = test_LSTM(test_SPK, test_dataset, args.buff_dir, args)
            #print(test_SPK, '\t', file = r)
            #print('MCD = %0.3f' % avg_vacc, file = r)

            WER = test_DeepSpeech(test_SPK, train_dataset, test_dataset, args.buff_dir, args)
            print(test_SPK, '\t', file = r)
            print('WER = %0.4f' % WER, file = r)
    r.close()
