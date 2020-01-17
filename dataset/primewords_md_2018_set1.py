from tqdm import tqdm
import fnmatch
import csv
import glob
import os
import json
import numpy as np


data_path = '/project/iwslt2014c/EN/student_projects/yzhu/data/primewords_md_2018_set1'

def create_trn(base_path):
    print('generating trn files.')
    label_path = os.path.join(data_path, 'set1_transcript.json')
    with open(label_path, 'r') as f:
        label_json = json.load(f)

    transcripts = {entry['file']: entry['text'] for entry in label_json}
    
    wav_path = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(base_path)
                for f in fnmatch.filter(files, '*.wav')]

    for wav in tqdm(wav_path):
        trn = os.path.splitext(wav)[0] + '.trn'
        with open(trn, 'w') as ftrn:
            wav_name = os.path.basename(wav)
            ftrn.write(transcripts[wav_name])


def generate_csv(base_path, dev_per=0.1, val_per=0.1):
    wav_path = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(base_path)
                for f in fnmatch.filter(files, '*.wav')]

    trn_path = list(map(lambda wav_path : os.path.splitext(wav_path)[0] +
                        '.trn', wav_path))

    indices = np.arange(0, len(wav_path))
    np.random.seed(12345)
    np.random.shuffle(indices)

    val_indices = indices[int(- len(indices) * val_per) : ]
    dev_indices = indices[int(- len(indices) * (dev_per + val_per)) : 
                          int(- len(indices) * val_per)]
    train_indices = indices[: int(- len(indices) * (dev_per + val_per))]
 
    print('generating train_manifest_primeword.csv')
    with open('train_manifest_primeword.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in tqdm(range(len(train_indices))):
            writer.writerow([wav_path[train_indices[i]], trn_path[train_indices[i]]])

    print('generating dev_manifest_primeword.csv')
    with open('dev_manifest_primeword.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in tqdm(range(len(dev_indices))):
            writer.writerow([wav_path[dev_indices[i]], trn_path[dev_indices[i]]])

    print('generating val_manifest_primeword.csv')
    with open('val_manifest_primeword.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in tqdm(range(len(val_indices))):
            writer.writerow([wav_path[val_indices[i]], trn_path[val_indices[i]]])


create_trn(data_path)
generate_csv(data_path)
