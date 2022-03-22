import numpy as np
import librosa
import torch
import random
import torchaudio

class ema_sin_noise(object):
    def __init__(self, prob = 0.5, noise_energy_ratio = 0.1, noise_freq  = 40, fs=100):
        self.prob = prob
        self.noise_energy_ratio = noise_energy_ratio
        self.noise_freq = noise_freq
        self.fs = fs
        
    def noise_injection(self, ema, noise_energy_ratio, noise_freq, fs):
        for i in range(ema.shape[0]):
            x = np.arange(ema.shape[3])
            sin_noise = torch.outer(torch.Tensor(np.sin(2 * np.pi * noise_freq * x / fs)), torch.abs(ema.mean(3)[i,0]))
            ema[i,0] = ema[i,0] + sin_noise.T
        return ema
        
    def __call__(self, ema):
        if random.random() < self.prob:
            ema = self.noise_injection(ema, self.noise_energy_ratio, self.noise_freq, self.fs) 
        return ema

class ema_time_mask(object):
    def __init__(self, prob = 0.5, mask_num = 20):
        self.prob = prob
        self.mask_num = mask_num
           
    def __call__(self, ema):
        masking = torchaudio.transforms.TimeMasking(time_mask_param=self.mask_num)
        if random.random() < self.prob:
            for j in range(ema.shape[0]):
                ema[j,0,:,:] = masking(ema[j,0,:,:])                 
        return ema
        
        
class ema_freq_mask(object):
    def __init__(self, prob = 0.5, mask_num = 20):
        self.prob = prob
        self.mask_num = mask_num
           
    def __call__(self, ema):
        masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.mask_num)
        if random.random() < self.prob:
            for j in range(ema.shape[0]):
                ema[j,0,:,:] = masking(ema[j,0,:,:])                      
        return ema

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

class Transform_Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X):
        for t in self.transforms:
            X = t(X)
        return X

class Pair_Transform_Compose(object):
    '''
    In an order of EMA, WAV
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X, Y):
        for t in self.transforms:
            X, Y = t(X, Y)
        return X, Y

class apply_MVN(object):
    def __init__(self, X_mean, X_std):
        self.X_mean = X_mean
        self.X_std = X_std
    def __call__(self, X):
        X_norm = (X - self.X_mean)/self.X_std
        return X_norm

############### EMA transformation #######################

class FixMissingValues(object):  

    def __call__(self, ema):
        
        from scipy.interpolate import interp1d
        
        ema_fixed = np.zeros(ema.shape)

        for i in range(ema.shape[1]):

            xnew = np.arange(len(ema[:,i]))
            zero_idx = np.where(np.isnan(ema[:,i]))

            xold = np.delete(xnew,zero_idx)
            yold = np.delete(ema[:,i], zero_idx)  

            f = interp1d(xold,yold)
            ynew = f(xnew)
            ema_fixed[:,i] = f(xnew)

        return ema_fixed

class apply_delta_deltadelta(object):
    # Adopted from nnmnkwii source code https://github.com/r9y9/nnmnkwii
    
    def delta(self, x, window):

        T, D = x.shape
        y = np.zeros_like(x)
        for d in range(D):
            y[:, d] = np.correlate(x[:, d], window, mode = "same")
        return y
    
    def apply_delta_windows(self, x, windows):

        T, D = x.shape
        assert len(windows) > 0
        combined_features = np.empty((T, D * len(windows)), dtype=x.dtype)
        for idx, (_, _, window) in enumerate(windows):
            combined_features[:, D * idx:D * idx + D] = self.delta(x, window)
        return combined_features
    
    def __call__(self, ema):
    
        windows = [(0, 0, np.array([1.0])), 
                   (1, 1, np.array([-0.5, 0.0, 0.5])),
                   (1, 1, np.array([1.0, -2.0, 1.0]))]
        
        ema_delta = self.apply_delta_windows(ema, windows)
        
        return ema_delta

class low_pass_filtering(object):
    def __init__(self, cutoff_freq, fs):
        
        self.fs = fs
        self.cutoff_freq = cutoff_freq
    def __call__(self, I):
        from scipy.signal import butter, lfilter
        cutoff_norm = self.cutoff_freq/(self.fs/2)
        b, a = butter(5, cutoff_norm, btype='low', analog=False)
        I_filtered = lfilter(b, a, I)

        return I_filtered

class ProcrustesMatching(object):
    def __init__(self, sel_sensors, sel_dim, lateral = 'xz'):        
        self.sel_sensors = sel_sensors
        self.sel_dim = sel_dim
        self.lateral = lateral        
        self.sel_header = []
        for sensor in self.sel_sensors:
            for d in self.sel_dim:
                sel_header_element = sensor + '_' + d
                self.sel_header.append(sel_header_element)
       
    def procrustes_translation(self, EMA_2D):

        sensor_centroid = np.mean(EMA_2D, axis = 0)
        shape_centroid = np.mean(np.reshape(sensor_centroid, (-1, 2)), axis = 0)
        shape_centroid_duplicated = np.resize(shape_centroid, (1,EMA_2D.shape[1]))
        EMA_translated = EMA_2D - shape_centroid_duplicated
        return EMA_translated 
    
    def procrustes_rotation(self, EMA_2D, lip_dim):
        
        import math                                            
        ul_x, ul_y = lip_dim[0], lip_dim[1]
        ll_x, ll_y = lip_dim[2], lip_dim[3]
                      
        UL = np.concatenate((EMA_2D[:,ul_x:ul_x+1], EMA_2D[:,ul_y:ul_y+1]), axis = 1)
        LL = np.concatenate((EMA_2D[:,ll_x:ll_x+1], EMA_2D[:,ll_y:ll_y+1]), axis = 1)

        UL_cent = np.mean(UL, axis = 0)
        LL_cent = np.mean(LL, axis = 0)

        Dis = UL_cent - LL_cent
        tan_angle = Dis[0]/Dis[1]

        angle = np.arctan(tan_angle)
        rotate_matrix = [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]
        EMA_2D_rotated = np.zeros((EMA_2D.shape[0], EMA_2D.shape[1]))

        for i in range(int((EMA_2D.shape[1])/2)):

            sensor_2D = EMA_2D[:,2*i:2*i+2]
            sensor_2D_rotated = np.dot(sensor_2D, rotate_matrix)
            EMA_2D_rotated[:,2*i:2*i+2] = sensor_2D_rotated

        return EMA_2D_rotated
        
    
    def __call__(self, ema):
        lip_idx = []
        for lip in ['UL', 'LL']:       
            for d in self.lateral:
                lip_dim = lip + '_p' + d
                lip_idx.append(self.sel_header.index(lip_dim)) # UL_px, UL_pz, LL_px, LL_pz

        idx_2D = []
        for i in range(len(self.sel_header)):
            dim = self.sel_header[i]
            if dim[dim.index('_')+1:] == 'p'+self.lateral[0] or dim[dim.index('_')+1:] == 'p'+self.lateral[1]:
                idx_2D.append(self.sel_header.index(dim))
        
        lip_dim_in_2D = [idx_2D.index(x) for x in lip_idx]        
        ema_2D = ema[:, idx_2D]
        
        ema_2D_translate = self.procrustes_translation(ema_2D)
        ema_2D_translate_rotate = self.procrustes_rotation(ema_2D_translate, lip_dim_in_2D)
        
        ### Put the transformed ema_2D back to ema_3D or with delta       
        ema_P = ema        
        for i in range(len(idx_2D)):
            ema_P[:,idx_2D[i]] = ema_2D_translate_rotate[:,i]
                   
        return ema_P

############### Audio transformation #######################
class change_wav_sampling_rate(object):
    def __init__(self, org_fs = 44100, tar_fs=22050):
        self.tar_fs = tar_fs
        self.org_fs = org_fs
    
    def __call__(self, wav):
        import librosa
        y_out = np.expand_dims(librosa.resample(wav[:,0], self.org_fs, self.tar_fs), axis = 1)
        return y_out

class wav2melspec(object):
    def __init__(self, sampling_rate=22050, filter_length=1024, hop_length=256, win_length=1024, 
                 n_mel_channels=80, mel_fmin=0.0, mel_fmax=8000.0):
        import audio_processing.layers
        import torch
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin, self.mel_fmax = mel_fmin, mel_fmax
        self.stft = audio_processing.layers.TacotronSTFT(self.filter_length, self.hop_length, self.win_length,
                    self.n_mel_channels, self.sampling_rate, self.mel_fmin, self.mel_fmax)
    
    def __call__(self, wav):
        audio = torch.FloatTensor(wav.astype(np.float32))
     #   audio_norm = audio / max_wav_value
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec.T

############### ATS transformation #######################

class Fix_EMA_MissingValues_ATS(FixMissingValues):  
    def __call__(self, ema, wav):        
        return super().__call__(ema), wav

class apply_delta_deltadelta_EMA_ATS(apply_delta_deltadelta):
    def __call__(self, ema, wav):        
        return super().__call__(ema), wav

class ProcrustesMatching_ATS(ProcrustesMatching):
    def __call__(self, ema, wav):        
        return super().__call__(ema), wav




class apply_EMA_MVN(object):
    def __init__(self, X_mean, X_std):
        self.X_mean = X_mean
        self.X_std = X_std
    def __call__(self, X, Y):
        X_norm = (X - self.X_mean)/self.X_std
        return X_norm, Y

class apply_WAV_MVN(object):
    def __init__(self, Y_mean, Y_std):
        self.Y_mean = Y_mean
        self.Y_std = Y_std
    def __call__(self, X, Y):
        Y_norm = (Y - self.Y_mean)/self.Y_std
        return X, Y_norm

class apply_EMA_MinMax(object):
    def __init__(self, X_min, X_max):
        self.X_min = X_min
        self.X_max = X_max
    def __call__(self, X, Y):
        X_norm = (X - self.X_min)/(self.X_max - self.X_min)
        return X_norm, Y

class apply_WAV_MinMax(object):
    def __init__(self, Y_min, Y_max):
        self.Y_min = Y_min
        self.Y_max = Y_max
    def __call__(self, X, Y):
        Y_norm = (Y - self.Y_min)/(self.Y_max - self.Y_min)
        return X, Y_norm
        
####### Data augmentation transform ########
'''
class ema_wav_random_scale(object):
    def __init__(self, prob = 0.5, rates = [0.8, 0.9, 1.1, 1.2]):
        self.prob = prob
        self.rates = rates
    def __call__(self, ema, wav):
        from scipy import ndimage
        if random.random() < self.prob:
            rate = random.choice(self.rates)
            print(rate)
            ema_align = np.empty([int(ema.shape[0]*rate+0.5), ema.shape[1]])
            wav_align = np.empty([int(wav.shape[0]*rate+0.5), wav.shape[1]])   
            for i in range(ema.shape[1]):
                ema_align[:,i] = ndimage.zoom(ema[:,i], rate)
            for i in range(wav.shape[1]):
                wav_align[:,i] = ndimage.zoom(wav[:,i], rate)
        return ema_align, wav_align
'''
class ema_wav_random_scale(object):
    def __init__(self, prob = 0.5, rates = [0.8, 0.9, 1.1, 1.2]):
        self.prob = prob
        self.rates = rates
    def __call__(self, ema, wav):
        from scipy import ndimage
        if random.random() < self.prob:
            rate = random.choice(self.rates)
            ema_align = np.empty([ema.shape[0], int(ema.shape[1]*rate+0.5), ema.shape[2]])
            wav_align = np.empty([wav.shape[0], int(wav.shape[1]*rate+0.5), wav.shape[2]])   
            for i in range(ema.shape[2]):
                ema_align[:,:,i] = ndimage.zoom(ema[:,:,i], rate)
            for i in range(wav.shape[2]):
                wav_align[:,:,i] = ndimage.zoom(wav[:,:,i], rate)
            ema, wav = torch.from_numpy(ema_align), torch.from_numpy(wav_align)
        return ema, wav
        
class ema_wav_random_rotate(object):
    def __init__(self, prob = 0.5, angle_range = [-30, 30]):
        self.prob = prob
        self.angle_range = angle_range
        
    def rotation(self, EMA, angle):
        import math
        rotate_matrix = [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]
        EMA_rotated = np.zeros((EMA.shape[0], EMA.shape[1], EMA.shape[2]))

        for i in range(int((EMA.shape[2])/2)):

            sensor_2D = EMA[:,:,2*i:2*i+2]
            sensor_2D_rotated = np.dot(sensor_2D, rotate_matrix)
            EMA_rotated[:,:,2*i:2*i+2] = sensor_2D_rotated

        return EMA_rotated        
   
    def __call__(self, ema, wav):
        if random.random() < self.prob:
            angle = random.randint(self.angle_range[0], self.angle_range[1])
            ema = torch.from_numpy(self.rotation(ema, angle))
                        
        return ema, wav
