import os
import fnmatch
import csv

dev_path = '/home/yzhu/thchs30/data_thchs30/dev'
train_path = '/home/yzhu/thchs30/data_thchs30/train'
val_path = '/home/yzhu/thchs30/data_thchs30/test'

train_wav_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(train_path)
                  for f in fnmatch.filter(files, '*.wav')]

train_transcript_path = list(map(lambda wav_path : wav_path + '.trn', train_wav_paths))

with open('train_manifest.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(train_wav_paths)):
        writer.writerow([train_wav_paths[i], train_transcript_path[i]])


dev_wav_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(dev_path)
                  for f in fnmatch.filter(files, '*.wav')]

dev_transcript_path = list(map(lambda wav_path : wav_path + '.trn', dev_wav_paths))

with open('dev_manifest.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(dev_wav_paths)):
        writer.writerow([dev_wav_paths[i], dev_transcript_path[i]])


val_wav_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(val_path)
                  for f in fnmatch.filter(files, '*.wav')]

val_transcript_path = list(map(lambda wav_path : wav_path + '.trn', val_wav_paths))

with open('val_manifest.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(val_wav_paths)):
        writer.writerow([val_wav_paths[i], val_transcript_path[i]])
