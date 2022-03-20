import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import librosa
import numpy as np


class MCD(object):
    def __init__(self, n_mfcc=13):
        self.n_mfcc = n_mfcc

    def mcd(self, C, C_hat):
        """C and C_hat are NumPy arrays of shape (T, D),
        representing mel-cepstral coefficients.

        """
        K = 10 / np.log(10) * np.sqrt(2)
        return K * np.mean(np.sqrt(np.sum((C - C_hat) ** 2, axis=1)))

    def __call__(self, y_head, y):
        pred_log = np.log10((y_head.detach().cpu().numpy())**2)
        org_log = np.log10((y.detach().cpu().numpy())**2)
        mfccs_pred = librosa.feature.mfcc(S=pred_log.T,
                                         dct_type=2, n_mfcc=self.n_mfcc, norm='ortho', lifter=0)
        mfccs_org = librosa.feature.mfcc(S=org_log.T,
                                         dct_type=2, n_mfcc=self.n_mfcc, norm='ortho', lifter=0)

        mfcc_pred_T = mfccs_pred.T
        mfcc_org_T = mfccs_org.T
        MCD = self.mcd(mfcc_pred_T[:,1:12], mfcc_org_T[:,1:12])
        return MCD
