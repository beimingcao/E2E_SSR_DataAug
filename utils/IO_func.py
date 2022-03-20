import os
import glob
import numpy as np
import scipy.io as sio
import librosa

def save_word_label(word_label, save_path):
    punctuations_to_remove = ',?.!/;:~'
    
    for char in word_label:
        if char in punctuations_to_remove:
            word_label = word_label.replace(char,'')
    
    f = open(save_path, 'w')
    n = f.write(word_label)
    f.close()

def save_phone_label(phone_label, save_path):
    import re
    phone_time_list = phone_label[0]
    idx = 0
    with open(save_path, 'w') as f:        
        for phone, t in phone_time_list:
            phone_new = re.sub(r'[0-9]+', '', phone[0])
            f.write(phone_new)
            f.write('\t')
            f.write(str(t[0][0]))
            f.write('\t')
            f.write(str(t[0][1]))
            if idx < len(phone_time_list):
                f.write('\n')
                idx += 1
    f.close()

def load_binary_file(file_name, dimension):
    
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    features = features[:(dimension * (features.size // dimension))]
    features = features.reshape((-1, dimension))

    return  features

def read_file_list(file_name):

    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()

    return file_lists

def array_to_binary_file(data, output_file_name):
    data = np.array(data, 'float32')

    fid = open(output_file_name, 'wb')
    data.tofile(fid)
    fid.close()
    
def load_Haskins_SSR_data(data_path, file_id, sel_sensors, sel_dim):

    org_sensors = ['TR', 'TB', 'TT', 'UL', 'LL', 'ML', 'JAW', 'JAWL']
    org_dims = ['px', 'py', 'pz', 'ox', 'oy', 'oz'] 

    data = sio.loadmat(data_path)[file_id][0]
    sensor_index = [org_sensors.index(x)+1 for x in sel_sensors]
    dim_index = [org_dims.index(x) for x in sel_dim]

    idx = 0
    for i in sensor_index:

        sensor_name = data[i][0]
        sensor_data = data[i][2]
        sel_dim = sensor_data[:,dim_index]
        if idx == 0:
            EMA = sel_dim
            fs_ema = data[i][1]
        else:
            EMA = np.concatenate((EMA, sel_dim), axis = 1)
        idx += 1
    ### load text data ###
    wav = data[0][2]
    sent = data[0][4]
    word_label = data[0][5]
    phone_label = data[0][6]
    word_label_ms = data[0][7]

    return EMA, fs_ema, wav, sent, phone_label, word_label, word_label_ms

def load_Haskins_ATS_data(data_path, file_id, sel_sensors, sel_dim):

    org_sensors = ['TR', 'TB', 'TT', 'UL', 'LL', 'ML', 'JAW', 'JAWL']
    org_dims = ['px', 'py', 'pz', 'ox', 'oy', 'oz'] 

    data = sio.loadmat(data_path)[file_id][0]
    sensor_index = [org_sensors.index(x)+1 for x in sel_sensors]
    dim_index = [org_dims.index(x) for x in sel_dim]

    idx = 0
    for i in sensor_index:

        sensor_name = data[i][0]
        sensor_data = data[i][2]
        sel_dim = sensor_data[:,dim_index]
        if idx == 0:
            EMA = sel_dim
            fs_ema = data[i][1]
        else:
            EMA = np.concatenate((EMA, sel_dim), axis = 1)
        idx += 1
    ### load wav data ###
    fs_wav = data[0][1]
    WAV = data[0][2]

    return EMA, WAV, fs_ema, fs_wav
