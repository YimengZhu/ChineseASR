import json

from scipy.io.wavfile import read
import librosa

import torch
from torch.utils.data import Dataset, Dataloader
import pandas

class WavDataset(Dataset):

    def __init__(self, csv_path, lexicon_path, sample_rate=16000,
                 window='hamming', window_size=.02, stride_size=.01):
        self.samples = pandas.read_csv(csv_path)
        with open(lexicon_path) as file:
            lexicon = str(''.join(json.load(file)))
        self.label = dict([(lexicon[i], i) for i in range(len(lexicon))])

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
        sample_rate, sound = read(wav_path)
        sound = sound.astype('float32') / 32767
        if len(sound.shape) > 1:
            sound = sound.squeeze() if sound.shape[1] == 1 else sound.mean(axis=1)
        
        window_length = int(self.sample_rate * self.window_size)
        stride_length = int(self.sample_rate * self.stride_size)

        stft = librosa.stft(sound, n_fft=window_length,
                            hop_length=stride_length, win_length=window_length,
                           window = self.window)
        
        magnitude, phase = librosa.magphase(stft)

        spect = torch.FloatTensor(np.log1p(magnitude))

        spect.add_(-spect.mean())
        spect.div_(spect.div())

        return spect


    def __parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as file:
            transcript = transcript_file.readline().replace('\n', '')
        transcript = [self.label.get(c) for c in list(trancript)]
        transcript = list(filter(None, transcript))
        return transcript

