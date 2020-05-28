import json
import numpy as np
import torch
from scipy.io.wavfile import read
import librosa
from data_aug import TimeStretch, SpectAugment
from torch.utils.data import Dataset, DataLoader
import pandas
from pdb import set_trace as bp

class SpeechDataset(Dataset):

    def __init__(self, csv_path, lexicon_path='lexicon.json', augment=False):

        with open(lexicon_path) as file:
            lexicon = str(''.join(json.load(file)))
        self.label = dict([(lexicon[i], i) for i in range(len(lexicon))])

        self.samples = pandas.read_csv(csv_path, header=None)
        self.stretch = TimeStretch(100) if augment else None
        self.spectAug = SpectAugment() if augment else None

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

        feature = librosa.feature.melspectrogram(y=sound, sr=sample_rate,
                                                 n_fft=512, hop_length=250,
                                                 n_mels=120)
        feature = librosa.core.amplitude_to_db(feature)
        feature = (feature - feature.mean()) / feature.std()
        if self.stretch is not None:
            feature = self.stretch(feature)

        if self.spectAug is not None:
           feature = self.spectAug(feature)

        return feature


    def __parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.readline().replace('\n', '')
        transcript = [self.label.get(c) for c in list(transcript)]
        transcript = list(filter(None, transcript))
        return transcript

def collate(batch):
    # batch.shape = N * (D * T, Transcript)
    batch = sorted(batch, key=lambda sample:sample[0].shape[1], reverse=True)
    longest_sample = max(batch, key=lambda sample:sample[0].shape[1])[0]
    spect_dim, max_length = longest_sample.shape
    batch_size = len(batch)

    batch_spect = torch.zeros(batch_size, max_length, spect_dim)
    batch_transcript = []
    spect_lengths = torch.IntTensor(batch_size)
    transcript_lengths = torch.IntTensor(batch_size)

    for i in range(batch_size):
        spect, transcript = batch[i]
        spect = torch.FloatTensor(spect)
        batch_spect[i].narrow(0, 0, spect.size(1)).copy_(spect.transpose(0, 1))
        batch_transcript.extend(transcript)
        spect_lengths[i] = spect.size(1)
        transcript_lengths[i] = len(transcript)
    batch_transcript = torch.IntTensor(batch_transcript)
    return batch_spect, batch_transcript, spect_lengths, transcript_lengths

class SpeechDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechDataloader, self).__init__(*args, **kwargs)
        self.collate_fn = collate


