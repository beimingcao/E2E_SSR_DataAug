import os
import glob
import numpy as np
from scipy import ndimage
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_ATS_data
import librosa

###################### Haskins IEEE dataset #####################################
'''

org_sensor_list = ['TR', 'TB', 'TT', 'UL', 'LL', 'ML', 'JAW', 'JAWL']
org_dim_per_sensor = ['px', 'py', 'pz', 'ox', 'oy', 'oz']


EMA trajectory format [nSamps x 6 dimensions]:
	posX (mm)
	posY
	posZ
	rotation around X (degrees)
	         around Y
	         around Z 

'''

def phn_file_parse(phn_path):
    
    import csv
    import numpy as np
    import re
    
    reader = csv.reader(open(phn_path))
    data_list = list(reader)
    
    phone_seq = []
    starts, ends = [], []
    for phone_start_end in data_list:       
        phone_tag = re.split(r'\t+', phone_start_end[0])
        phone_seq.append(phone_tag[0].upper())
        starts.append(float(phone_tag[1]))
        ends.append(float(phone_tag[2]))
        
    return phone_seq, starts, ends

class PhoneTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        SP 0
        AA 1
        AE 2
        AH 3
        AO 4
        AW 5
        AY 6
        B 7
        CH 8
        D 9
        DH 10
        EH 11
        ER 12
        EY 13
        F 14
        G 15
        HH 16
        IH 17
        IY 18
        JH 19
        K 20
        L 21
        M 22
        N 23
        NG 24
        OW 25
        OY 26
        P 27
        R 28
        S 29
        SH 30
        T 31
        TH 32
        UH 33
        UW 34
        V 35
        W 36
        Y 37
        Z 38
        ZH 39
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == 'SHH':
                ch = 16
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[int(i)])
        return string

################ Haskins dataset for SSR #######################

class HaskinsData_SSR(Dataset):
    def __init__(self, data_path, file_list, ema_dim, transforms=None):
        self.data_path = data_path
        self.file_list = file_list
        self.ema_dim = ema_dim
        self.transforms = transforms
        
        self.data = []
        for file_id in self.file_list:
            data_path_spk = os.path.join(self.data_path, file_id[:3])
            ema_path = os.path.join(data_path_spk, file_id + '.ema')
            phn_path = os.path.join(data_path_spk, file_id + '.phn')
            ema = load_binary_file(ema_path, self.ema_dim)
            phone_seq, starts, ends = phn_file_parse(phn_path)
            text_transform = PhoneTransform()
            label = torch.Tensor(text_transform.text_to_int(phone_seq))
            self.data.append((file_id, ema, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        file_id, EMA, TXT = self.data[idx]     
        if self.transforms is not None:
            EMA, TXT = self.transforms(EMA, TXT)         
        return (file_id, EMA, TXT)

