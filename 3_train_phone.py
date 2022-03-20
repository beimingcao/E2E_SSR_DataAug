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
from torch.optim.lr_scheduler import StepLR
from utils.utils import EarlyStopping, IterMeter, data_processing_DeepSpeech
import torch.nn.functional as F

import random
from utils.transforms import ema_random_rotate
from utils.transforms import apply_delta_deltadelta, Transform_Compose
from utils.transforms import apply_MVN, 
import numpy as np

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class ema_random_rotate(object):
    def __init__(self, prob = 0.5, angle_range = [-30, 30]):
        self.prob = prob
        self.angle_range = angle_range
        
    def rotation(self, EMA, angle):
        import math

        rotate_matrix = np.matrix([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        EMA_rotated = np.zeros((EMA.shape[0], 1, EMA.shape[2], EMA.shape[3]))
        for j in range(EMA.shape[0]):
            for i in range(int((EMA.shape[2])/2)):
                sensor_2D = EMA[j,0,[2*i, 2*i+1],:]
                sensor_2D_rotated = np.dot(sensor_2D.T, rotate_matrix)
                EMA_rotated[j,0,[2*i, 2*i+1],:] = sensor_2D_rotated.T

        return EMA_rotated        
   
    def __call__(self, ema):
        if random.random() < self.prob:
            angle = random.randint(self.angle_range[0], self.angle_range[1])
            ema = torch.from_numpy(self.rotation(ema, angle))
                        
        return ema

def train_DeepSpeech(test_SPK, train_dataset, valid_dataset, exp_output_folder, args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    
    ### Training setup ###
    learning_rate = config['deep_speech_setup']['learning_rate']
    batch_size = config['deep_speech_setup']['batch_size']
    epochs = config['deep_speech_setup']['epochs']
    early_stop = config['deep_speech_setup']['early_stop']
    patient = config['deep_speech_setup']['patient']
    normalize_input = config['articulatory_data']['normalize_input']
    train_out_folder = os.path.join(exp_output_folder, 'training')
    if not os.path.exists(train_out_folder):
        os.makedirs(train_out_folder)
    results = os.path.join(train_out_folder, test_SPK + '_train.txt')
    
    ### Model training ###
    
    random_rotate_apply = config['data_augmentation']['random_rotate']
    random_noise_add = config['data_augmentation']['random_noise']
    random_mask = config['data_augmentation']['random_mask']
    
    train_transform = []
    valid_transform = []
    train_transform.append(apply_delta_deltadelta())
    valid_transform.append(apply_delta_deltadelta())
    
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
        
        train_transform.append(apply_MVN(EMA_mean, EMA_std))
        valid_transform.append(apply_MVN(EMA_mean, EMA_std))
    
    train_transforms_all = Transform_Compose(train_transform)
    valid_transforms_all = Transform_Compose(valid_transform)
        
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = train_transforms_all))
                                
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = valid_transforms_all))
                                
    model = SpeechRecognitionModel(n_cnn_layers, n_rnn_layers, rnn_dim, D_out, D_in, stride, dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    criterion = torch.nn.CTCLoss(blank=40).to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=int(len(train_loader)), epochs=epochs, anneal_strategy='linear')
    
    data_len = len(train_loader.dataset)
    if early_stop == True:
        print('Applying early stop.')
        early_stopping = EarlyStopping(patience=patient)
        
    iter_meter = IterMeter()
        
    with open(results, 'w') as r:    
        for epoch in range(epochs):
            model.train()
            loss_train = []
            for batch_idx, _data in enumerate(train_loader):
                file_id, ema, labels, input_lengths, label_lengths = _data 
                
                if random_rotate_apply == True:                
                    ratio = config['random_rotate']['ratio']
                    r_min = config['random_rotate']['r_min']
                    r_max = config['random_rotate']['r_max']                    
                
                    random_rotate = ema_random_rotate(prob = ratio, angle_range = [r_min, r_max])
                    ema = random_rotate(ema).float()
                    ema, labels = ema.to(device), labels.to(device)
                                       
                output = model(ema)  # (batch, time, n_class)

                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                iter_meter.step()
                
                loss_train.append(loss.detach().cpu().numpy())
            avg_loss_train = sum(loss_train)/len(loss_train)

            model.eval()
            loss_valid = []
            for batch_idx, _data in enumerate(valid_loader):  
                file_id, ema, labels, input_lengths, label_lengths = _data 
                ema, labels = ema.to(device), labels.to(device)           
                  
                output = model(ema)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)
                loss = criterion(output, labels, input_lengths, label_lengths)    
                loss_valid.append(loss.detach().cpu().numpy())
            avg_loss_valid = sum(loss_valid)/len(loss_valid) 
            SPK = file_id[0][:3]

            early_stopping(avg_loss_valid)
            if early_stopping.early_stop:
                break

            print('epoch %-3d \t train_loss = %0.5f \t valid_loss = %0.5f' % (epoch, avg_loss_train, avg_loss_valid))
            print('epoch %-3d \t train_loss = %0.5f \t valid_loss = %0.5f' % (epoch, avg_loss_train, avg_loss_valid), file = r)                           
                            
            model_out_folder = os.path.join(exp_output_folder, 'trained_models')
            if not os.path.exists(model_out_folder):
                os.makedirs(model_out_folder)
            if early_stopping.save_model == True:
                save_model(model, os.path.join(model_out_folder, test_SPK + '_DS'))
    r.close()
    print('Training for testing SPK: ' + test_SPK + ' is done.')       
           
def train_LSTM(test_SPK, train_dataset, valid_dataset, exp_output_folder, args):
    
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    sel_sensors = config['articulatory_data']['sel_sensors']
    sel_dim = config['articulatory_data']['sel_dim'] 
    delta = config['articulatory_data']['delta']
    d = 3 if delta == True else 1
    D_in = len(sel_sensors)*len(sel_dim)*d
    D_out = config['acoustic_feature']['n_mel_channels']
    hidden_size = config['training_setup']['hidden_size']
    num_layers = config['training_setup']['layer_num']
    batch_size = config['training_setup']['batch_size']

    learning_rate = config['training_setup']['learning_rate']
    weight_decay = config['training_setup']['weight_decay']
    num_epoch = config['training_setup']['num_epoch']
    early_stop = config['training_setup']['early_stop']
    patient = config['training_setup']['patient']
    
    #### data augmentation ####
    
    random_scale = config['data_augmentation']['random_scale']
    random_rotate = config['data_augmentation']['random_rotate']
    random_noise = config['data_augmentation']['random_noise']
    random_pause = config['data_augmentation']['random_pause']    

    model = MyLSTM(D_in, hidden_size, D_out, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = RegressionLoss()
    metric = MCD()

    train_data = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_data = DataLoader(valid_dataset, num_workers=0, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    train_out_folder = os.path.join(exp_output_folder, 'training')
    if not os.path.exists(train_out_folder):
        os.makedirs(train_out_folder)
    results = os.path.join(train_out_folder, test_SPK + '_train.txt')

    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
    if early_stop == True:
        print('Applying early stop.')
        early_stopping = EarlyStopping(patience=patient)
    with open(results, 'w') as r:
        for epoch in range(num_epoch):
            model.train()
            acc_vals = []
            for file_id, x, y in train_data:

                if random_scale == True:
                    random_scale = ema_wav_random_scale(prob = 0.2)
                    x, y = random_scale(x, y)    
                if random_rotate == True:
                    random_rotate = ema_wav_random_rotate()
                    x, y = random_rotate(x, y)             
                    
                x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
                h, c = model.init_hidden(x)
                h, c = h.to(device), c.to(device)
                y_head = model(x, h, c)

                loss_val = loss_func(y_head, y)
            #    acc_val = metric(y_head.squeeze(0), y.squeeze(0))
                acc_val = loss_val
                acc_vals.append(acc_val)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()    
            avg_acc = sum(acc_vals) / len(acc_vals)

            model.eval()
            acc_vals = []
            for file_id, x, y in valid_data:
                x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
                h, c = model.init_hidden(x)
                h, c = h.to(device), c.to(device)
                acc_vals.append(metric(model(x, h, c).squeeze(0), y.squeeze(0)))
            scheduler.step()
            avg_vacc = sum(acc_vals) / len(acc_vals)
            SPK = file_id[0][:3]

            early_stopping(avg_vacc)
            if early_stopping.early_stop:
                break

            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc), file = r)

            model_out_folder = os.path.join(exp_output_folder, 'trained_models')
            if not os.path.exists(model_out_folder):
                os.makedirs(model_out_folder)
            if early_stopping.save_model == True:
                save_model(model, os.path.join(model_out_folder, test_SPK + '_lstm'))
    r.close()
    print('Training for testing SPK: ' + test_SPK + ' is done.')




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/SSR_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    data_path = os.path.join(args.buff_dir, 'data_CV')
    SPK_list = config['data_setup']['spk_list']

    for test_SPK in SPK_list:
        data_path_SPK = os.path.join(data_path, test_SPK)

        tr = open(os.path.join(data_path_SPK, 'train_data.pkl'), 'rb') 
        va = open(os.path.join(data_path_SPK, 'valid_data.pkl'), 'rb')        
        train_dataset, valid_dataset = pickle.load(tr), pickle.load(va)

    #    train_LSTM(test_SPK, train_dataset, valid_dataset, args.buff_dir, args)  
        train_DeepSpeech(test_SPK, train_dataset, valid_dataset, args.buff_dir, args)   



