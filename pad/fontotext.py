# ./src/fontotext.py

import os
from IPython.display import Audio, display
from scipy.io import wavfile
import unicodedata
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
import zipfile
import csv
import random
import os
from typing import Tuple, Optional
import csv
import random
import tensorflow as tf
from torch import Tensor



def get_data(datatype): # can be either train or test. Any other format will throw an error.
  if datatype != "valid":
    de=datatype
  else:
    de = "test"

  with open('./FonAudio/pyFongbe-master/data/{}.csv'.format(de), newline='',encoding='UTF-8') as f:
      reader = csv.reader(f)
      data = list(reader)
      data = [data[i] for i in range(len(data)) if i!=0]

  if datatype == "test":
    test_data = [data[i] for i in range(len(data)) if i not in valid_indices]
    return test_data

  if datatype == "valid": # then we should get out some for valid
    val_data = [data[i] for i in valid_indices]
    return val_data

  return data

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)

def load_audio_item(d: list):
    utterance = d[2]
    wav_path = d[0]
    wav_path = wav_path.replace("/home/frejus/Projects/Fongbe_ASR/pyFongbe","./FonAudio/pyFongbe-master")
    print(wav_path)
    # Load audio
    waveform, sample_rate = torchaudio.load(wav_path)
    
    return (waveform, 
        sample_rate,
        utterance
    )


class FonASR(torch.utils.data.Dataset):
    """Create a Dataset for Fon ASR.
    Args:
    data_type could be either 'test', 'train' or 'valid'
    """
    def __init__(self, data_type):

      """data_type could be either 'test', 'train' or 'valid' """
      self.data = get_data(data_type)

    def __getitem__(self, n: int):
      """Load the n-th sample from the dataset.

      Args:
          n (int): The index of the sample to be loaded

      Returns:
          tuple: ``(waveform, sample_rate, utterance)``
      """
      fileid = self.data[n]
      return load_audio_item(fileid)


    def __len__(self) -> int:
      return len(self.data)

accent_code = [b'\u0301',b'\u0300',b'\u0306',b'\u0308',b'\u0303']
alpha = {'ɔ':0,'ɛ':5}
accents = {b'\u0301':1,b'\u0300':2,b'\u0306':3,b'\u0308':4,b'\u0303':5}
mapping={
    1:'ɔ́',2:'ɔ̀',3:'ɔ̆',6:'έ',7:'ὲ',8:'ɛ̆'
}
#we are following the idea that the composition gives the letter first followed by the sign(accent)
def get_better_mapping(text):
  t_arr = [t for t in text]
  s=[]
  for i in range(len(t_arr)):
    if t_arr[i].encode("unicode_escape") in accent_code:
      to_check = s[-1]
      try:
        val = mapping[alpha[to_check] + accents[t_arr[i].encode("unicode_escape")]]
        s.pop()
        s.append(val)
      except KeyError:
        print("Could not find for {} in sentence {} | Proceeding with default.".format(t_arr[i],text))
      
    else: 
      s.append(t_arr[i])
  return s

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

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
         1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        r 18
        s 19
        t 20
        u 21
        v 22
        w 23
        x 24
        y 25
        z 26
        à 27
        á 28
        è 29
        é 30
        ì 31
        í 32
        î 33
        ï 34
        ó 35
        ù 36
        ú 37
        ā 38
        ă 39
        ē 40
        ĕ 41
        ŏ 42
        ū 43
        ŭ 44
        ɔ 45
        ɖ 46
        ò 47
        ε 48
        έ 49
        ɔ̀ 50
        ɔ̆ 51
        ὲ 52
        ɔ́ 53
        ĭ 54
        ɛ̆ 55
        ɛ̃ 56
        . 57
        , 58
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
        text=unicodedata.normalize("NFC",text)
        for c in get_better_mapping(text):
            try:
              if c == ' ':
                  ch = self.char_map['']
              elif c =='̀':
                  ch=0
              else:
                  ch = self.char_map[c]
            except KeyError:
              print("Error for character {} in this sentence: {}".format(c,text))
              ch=0
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
test_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

text_transform = TextTransform()

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform,_,utterance in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'test':
            spec = test_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train, valid or test')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))
    
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def GreedyDecoder(output, labels, label_lengths, blank_label=59, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	targets = []
	for i, args in enumerate(arg_maxes):
		decode = []
		targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j-1]:
					continue
				decode.append(index.item())
		decodes.append(text_transform.int_to_text(decode))
	return decodes, targets



### THE MODEL

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()
        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, hidden = self.BiGRU(x)
        
        # Beginning of the attention part  
        if self.batch_first:
          hidden = hidden.transpose(0, 1).contiguous()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        w1 = nn.Linear(x.size()[2], x.size()[2]//2).to(device)
        w2 = nn.Linear(hidden.size()[2], hidden.size()[2]).to(device)
        
        v = nn.Linear(x.size()[2]//2, x.size()[2]).to(device)

        x1 = w1(x).to(device)
        hidden2 = w2(hidden).to(device)
        
        del w1, w2
        torch.cuda.empty_cache()

        # atrick to make size match
        # we add "0"s so that the addition with x1 doesn't change in principle anything
        # and tanh(x|x=0) = 0
        if hidden.size()[1] != x1.size()[1]:
          additional = np.full((hidden.size()[0], x1.size()[1]-hidden.size()[1],hidden.size()[2]), 0)
          hidden2 = torch.cat((hidden2, torch.tensor(additional, dtype=torch.float).to(device)), 1)
          del additional
          torch.cuda.empty_cache()
        
        m = nn.Tanh()
        score = v(m(x1+hidden2)).to(device) # compute attention scores

        del x1, hidden2, m, v
        torch.cuda.empty_cache()

        n = nn.Softmax()
        attention_weights = n(score) # get attention weights

        del score, n
        torch.cuda.empty_cache()
        context_vector = attention_weights * x # compute the attention vector
        x = torch.cat((context_vector, x), axis=-1) # apply context vector to the input

        del context_vector
        torch.cuda.empty_cache()
        # End of the attention part
        x = self.dropout(x)
        return x

class BidirectionalLSTM(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalLSTM, self).__init__()

        self.BiLSTM = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiLSTM(x) # enc_output, (hidden_state, cell_state)
        x = self.dropout(x)
        return x

class SpeechRecognitionModel(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        
        self.birnn_layers1 = nn.Sequential(*[
            BidirectionalLSTM(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])

        self.birnn_layers2 = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])

        self.birnn_layers2_attention = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim*2 if i==0 else rnn_dim*4,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=True)
            for i in range(n_rnn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
    
        self.classifier_attention = nn.Sequential(
            nn.Linear(rnn_dim*4, rnn_dim),  # birnn returns rnn_dim*4
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        x = self.birnn_layers1(x).to(device)        
        x = self.birnn_layers2_attention(x).to(device)
        x = self.classifier_attention(x)
        
        return x


### Main and Eval Scripts

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment,valid_loader,best_wer, cnn_lay, rnn_lay, model_path, rnn_dim):
    model.train()
    data_len = len(train_loader.dataset)
    train_loss=0
    batch_train_loss=0
    
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)
        loss = criterion(output, labels, input_lengths, label_lengths)
        train_loss += loss.item() / len(train_loader)
        # used if grad accumulation is used
        # train_loss += loss.item() / (len(train_loader)*BATCH_MULTIPLIER)
        loss.backward()

        # This is the idea of grad accumulation to overcome memory issue
        # if (batch_idx + 1) % BATCH_MULTIPLIER == 0:
        #     optimizer.step()
        #     scheduler.step()
        #     iter_meter.step()
        #     #model.zero_grad() #reset gradients
        #     optimizer.zero_grad()
        #     batch_train_loss+=train_loss
        #     train_loss=0
        #     # print('here after gradient')
       
        optimizer.step()
        scheduler.step()
        iter_meter.step()

        if batch_idx % 1000 == 0 or batch_idx == data_len:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(spectrograms), data_len,
                      100. * batch_idx / len(train_loader), loss.item()))
            
    experiment['loss'].append((train_loss,iter_meter.get()))
    val_wer = valid(model, device, valid_loader, criterion, epoch, iter_meter, experiment) # wer
    if val_wer < best_wer:
      best_wer = val_wer
      torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'val_loss':val_wer
              }, model_path)

    else:
      print("...No improvement in validation according to WER...")
    return best_wer

def valid(model, device, test_loader, criterion, epoch, iter_meter, experiment):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    experiment['val_loss'].append((test_loss, iter_meter.get()))
    experiment['cer'].append((avg_cer, iter_meter.get()))
    experiment['wer'].append((avg_wer, iter_meter.get()))

    print('Valid set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    return avg_wer
    

def test(model, device, test_loader, criterion, epoch, iter_meter, experiment, testing_wav):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    all_predictions = []
    with torch.no_grad():

        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            
            for j in range(len(decoded_preds)):
                wav_path = testing_wav[j][0]
                wav_path = wav_path.replace("/home/frejus/Projects/Fongbe_ASR/pyFongbe","./FonAudio/pyFongbe-master")
                rate, data = wavfile.read(wav_path)
                audio = Audio(data, rate=rate)
                display(audio)
                print("Decoding Speech's Content")
                print("Audio's Transcription: {}".format(decoded_preds[j]))
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
                current_prediction = "Decoded target: {}\nDecoded prediction: {}\n".format(decoded_targets[j], decoded_preds[j])
                all_predictions.append(current_prediction)

    def save_list(lines, filename):
      data = '\n'.join(lines)
      file = open(filename, 'w', encoding="utf-8")
      file.write(data)
      file.close()

    save_list(all_predictions, "best_model_predictions_1.txt")
    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))

def main(learning_rate, batch_size, epochs,experiment, cnn_layer, rnn_layer, model_path, rnn_dim, disabled=True):

    hparams = {
        "n_cnn_layers": cnn_layer,
        "n_rnn_layers": rnn_layer,
        "rnn_dim": rnn_dim,
        "n_class": 60,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("DEVICE: {}".format(device))

    if not os.path.isdir("./data"):
        print("Making dir of /data")
        os.makedirs("./data")

    
    train_dataset = FonASR("train")
    valid_dataset = FonASR("valid")
    test_dataset = FonASR("test")
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'test'),
                                **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)

    print("Model with CNN Layers == {} and RNN Layers == {}: and Rnn_Dim == {}\n\n".format(cnn_layer, rnn_layer, rnn_dim))
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device) # needs to be set to 59 for further words
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')
    
    iter_meter = IterMeter()
    best_wer= 1000
    
    if os.path.exists(model_path):
      checkpoint = torch.load(model_path, map_location='cpu')
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      epoch_saved = checkpoint['epoch']
      best_wer = checkpoint['val_loss']
      epoch_saved = epochs
      for epoch in range(epoch_saved+1, epochs + 1):
        print("Epoch Retrieved: {} with WER: {}".format(epoch_saved, best_wer))
        best_wer = train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment,valid_loader,best_wer, cnn_layer, rnn_layer, model_path, rnn_dim)
    else:      
      for epoch in range(1, epochs + 1):
          best_wer = train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment,valid_loader,best_wer, cnn_layer, rnn_layer, model_path, rnn_dim)    
    print("Evaluating on Test data:")
    wav_test = get_data("test")
    test(model, device, test_loader, criterion, epochs, iter_meter, experiment, wav_test)    
     
# %matplotlib inline
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
# import numpy as np
# import os
# model_path_ = '/content/drive/MyDrive/FonASR' # change accordingly to the model's folder location
# if not os.path.isdir(model_path_):
#   os.makedirs(model_path_)
# model_path_ = '/content/drive/MyDrive/FonASR/fonasr' # change accordingly to the model's base name

# def save_list(lines, filename):
#     data = '\n'.join(lines)
#     file = open(filename, 'w', encoding="utf-8")
#     file.write(data)
#     file.close()

# learning_rate = 5e-4
# batch_size = 20
# epochs = 500
# cnn_rnn_layers = [(5, 3, 512)]
# for cnn_rnn in cnn_rnn_layers:
#   model_path = model_path_+"_{}_{}_{}_gru_lstm_attention".format(cnn_rnn[0], cnn_rnn[1], cnn_rnn[2],epochs)
#   experiment={
#     'loss':[],
#     'val_loss':[],
#     'cer':[],
#     'wer':[]}
#   main(learning_rate, batch_size, epochs, experiment, cnn_rnn[0], cnn_rnn[1], model_path, cnn_rnn[2])
     