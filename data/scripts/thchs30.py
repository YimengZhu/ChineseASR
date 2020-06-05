from tqdm import tqdm
import os
import fnmatch
import csv

dev_path = '/project/iwslt2014c/EN/student_projects/yzhu/data/thchs30/data_thchs30/dev'
train_path = '/project/iwslt2014c/EN/student_projects/yzhu/data/thchs30/data_thchs30/train'
val_path = '/project/iwslt2014c/EN/student_projects/yzhu/data/thchs30/data_thchs30/test'
base_path = '/project/iwslt2014c/EN/student_projects/yzhu/data/thchs30/data_thchs30/data'

train_wav_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(train_path)
                  for f in fnmatch.filter(files, '*.wav')]

train_transcript_path = list(map(lambda wav_path : 
                                os.path.join(base_path,
                                            os.path.basename(wav_path) + '.trn'),
                                train_wav_paths))

print('generating train_manifest_thchs30.csv')
with open('train_manifest_thchs30.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in tqdm(range(len(train_wav_paths))):
        writer.writerow([train_wav_paths[i], train_transcript_path[i]])


dev_wav_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(dev_path)
                  for f in fnmatch.filter(files, '*.wav')]

dev_transcript_path = list(map(lambda wav_path :
                                os.path.join(base_path,
                                            os.path.basename(wav_path) + '.trn'),
                                dev_wav_paths))

print('generating dev_manifest_thchs30.csv')
with open('dev_manifest_thchs30.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in tqdm(range(len(dev_wav_paths))):
        writer.writerow([dev_wav_paths[i], dev_transcript_path[i]])


val_wav_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(val_path)
                  for f in fnmatch.filter(files, '*.wav')]

val_transcript_path = list(map(lambda wav_path : 
                                os.path.join(base_path,
                                            os.path.basename(wav_path) + '.trn'),
                                val_wav_paths))

print('generating val_manifest_thchs30.csv')
with open('val_manifest_thchs30.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in tqdm(range(len(val_wav_paths))):
        writer.writerow([val_wav_paths[i], val_transcript_path[i]])
