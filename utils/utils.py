import os
import yaml
import glob
import numpy as np
import scipy.io as sio
import librosa
import torch
from utils.IO_func import read_file_list
from utils.database import PhoneTransform

def data_processing_DeepSpeech(data, transforms = None):
    ema = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    for file_id, x, y in data:
        if transforms is not None:
            x = transforms(x)

        ema.append(torch.FloatTensor(x))
        labels.append(y)
        input_lengths.append(x.shape[0] // 2)
        label_lengths.append(len(y))
        
    ema = torch.nn.utils.rnn.pad_sequence(ema, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)        
    
    return file_id, ema, labels, input_lengths, label_lengths

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def GreedyDecoder(output, labels, label_lengths, blank_label=40, collapse_repeated=True):

    text_transform = PhoneTransform()

    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):

        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets

def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)
    
def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]

class EarlyStopping():

    # Adopted from: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_model = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            self.save_model = True
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            self.save_model = False
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def prepare_Haskins_lists(args):

    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    exp_type = config['experimental_setup']['experiment_type']  
    data_path = config['corpus']['path']
    fileset_path = os.path.join(data_path, 'filesets')
    spk_list = config['data_setup']['spk_list']
    num_exp = len(spk_list)
    
    exp_train_lists = {}
    exp_valid_lists = {}
    exp_test_lists = {}
    if exp_type == 'SD':
        for i in range(len(spk_list)):
            spk_fileset_path = os.path.join(fileset_path, spk_list[i])
            exp_train_lists[i] = read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
            exp_valid_lists[i] = read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
            exp_test_lists[i] = read_file_list(os.path.join(spk_fileset_path, 'test_id_list.scp'))
     
    elif exp_type == 'SI':
        for i in range(len(spk_list)):
            train_spk_list = spk_list.copy()
            train_spk_list.remove(spk_list[i])
            idx = 0
            train_lists, valid_lists = [], []
            for train_spk in train_spk_list:
                spk_fileset_path = os.path.join(fileset_path, train_spk)
                if idx == 0:
                    train_lists = read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                    valid_lists = read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
                else:
                    train_lists = train_lists + read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                    valid_lists = valid_lists + read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
                idx += 1
            test_lists = read_file_list(os.path.join(os.path.join(fileset_path, spk_list[i]), 'test_id_list.scp'))

            exp_train_lists[i] = train_lists
            exp_valid_lists[i] = valid_lists
            exp_test_lists[i] = test_lists

    elif exp_type == 'SA':
        idx = 0     
        for train_spk in spk_list:
            spk_fileset_path = os.path.join(fileset_path, train_spk)
            if idx == 0:
                train_lists = read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                valid_lists = read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
            else:
                train_lists = train_lists + read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                valid_lists = valid_lists + read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))

            exp_train_lists[idx] = train_lists
            exp_valid_lists[idx] = valid_lists            
            exp_test_lists[idx] = read_file_list(os.path.join(spk_fileset_path, 'test_id_list.scp')) 
            idx += 1           
    else:
        raise ValueError('Unrecognized experiment type')

    return exp_train_lists, exp_valid_lists, exp_test_lists

