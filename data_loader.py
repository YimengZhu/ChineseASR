import json
from scipy.io.wavfile import read
import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas
from pdb import set_trace as bp

class SpeechDataset(Dataset):

    def __init__(self, csv_path, feature='spect', lexicon_path='lexicon.json', sample_rate=16000, window='hamming', window_size=.02, stride_size=.01):
        if feature in ['mfcc', 'spect']:
            self.feature = feature
        else:
            raise Error('Invalid feature type')

        with open(lexicon_path) as file:
            lexicon = str(''.join(json.load(file)))
        self.label = dict([(lexicon[i], i) for i in range(len(lexicon))])

        self.samples = pandas.read_csv(csv_path, header=None)

        self.sample_rate = sample_rate
        self.window = window
        self.window_size = window_size
        self.stride_size = stride_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        wav_path = self.samples.iloc[idx, 0]
        transcript_path = self.samples.iloc[idx, 1]

        feature = self.__parse_wav(wav_path)
        label = self.__parse_transcript(transcript_path)
        return feature, label

    def __parse_wav(self, wav_path):
        waveform, sample_rate = torchaudio.load(wav_path)

        if self.feature == 'spect':
            sample_rate, sound = read(wav_path)
            sound = sound.astype('float32') / 32767
            if len(sound.shape) > 1:
                sound = sound.squeeze() if sound.shape[1] == 1 else sound.mean(axis=1)
            D = librosa.stft(sound,
                             n_fft=int(self.sample_rate * self.window_size),
                             hop_length=int(self.sample_rate * self.stride_size),
                             win_length=int(self.sample_rate * self.window_size),
                             window=self.window)
            feature, phase = librosa.magphase(D)
            feature = torch.FloatTensor(np.log1p(feature))
            feature.add_(-feature.mean())
            feature.div_(feature.std())
            # n_fft = int(self.window_size * self.sample_rate)
            # feature = torchaudio.transforms.Spectrogram(n_fft=n_fft)(waveform)
            # feature = feature.abs().log1p()
            # feature = (feature -  feature.mean()) / feature.std()
            # feature = feature.squeeze(0)
            # bp()
        if self.feature == 'mfcc':
            feature = torchaudio.transforms.MFCC()(waveform)
            feature = feature.squeeze(0)

        return feature

    def __parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.readline().replace('\n', '')
        transcript = [self.label.get(c) for c in list(transcript)]
        transcript = list(filter(None, transcript))
        return transcript

def batch_collate(batch):
    # batch.shape = N * (D * T, transcript)
    batch = sorted(batch, key=lambda sample:sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=lambda sample:sample[0].size(1))[0]

    spect_dim = longest_sample.size(0)
    max_length = longest_sample.size(1)
    batch_size = len(batch)

    batch_spect = torch.zeros(batch_size, 1, spect_dim, max_length)
    batch_transcript = []
    spect_lengths = torch.IntTensor(batch_size)
    transcript_lengths = torch.IntTensor(batch_size)

    for i in range(batch_size):
        spect, transcript = batch[i]
        batch_spect[i][0].narrow(1, 0, spect.size(1)).copy_(spect)
        batch_transcript.extend(transcript)
        spect_lengths[i] = spect.size(1)
        transcript_lengths[i] = len(transcript)
    batch_transcript = torch.IntTensor(batch_transcript)
    return batch_spect, batch_transcript, spect_lengths, transcript_lengths 


class SpeechDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechDataloader, self).__init__(*args, **kwargs)
        self.collate_fn = batch_collate
