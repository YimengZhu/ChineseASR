import tarfile
import glob
import os
import fnmatch
import csv


data_folder_path = '/project/student_projects3/yzhu/data/aidatatang_200zh'

def extract_data(data_folder):
    print('extracting data in aidatatang_200zh folder')

    for targz in glob.glob(os.path.join(data_folder, 'corpus', '*', '*.tar.gz')):
        print('extracting {}'.format(targz))
        with tarfile.open(targz) as tar:
            tar.extractall(os.path.dirname(targz))
        os.remove(targz)

def generate_csv(base_path):
    train_path = os.path.join(base_path, 'corpus', 'train')
    dev_path = os.path.join(base_path, 'corpus', 'dev')
    test_path = os.path.join(base_path, 'corpus', 'test')

    train_wav_path = [os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(train_path)
                    for f in fnmatch.filter(files, '*.wav')]

    train_transcript_path = list(map(lambda wav_path :
                                    os.path.splitext(wav_path)[0] + '.trn',
                                    train_wav_path))

    with open('train_manifest_aidatatang_200zh.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(train_wav_path)):
            writer.writerow([train_wav_path[i], train_transcript_path[i]])


    dev_wav_path = [os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(dev_path)
                    for f in fnmatch.filter(files, '*.wav')]

    dev_transcript_path = list(map(lambda wav_path :
                                    os.path.splitext(wav_path)[0] + '.trn',
                                    dev_wav_path))

    with open('dev_manifest_aidatatang_200zh.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(dev_wav_path)):
            writer.writerow([dev_wav_path[i], dev_transcript_path[i]])
    
    
    test_wav_path = [os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(test_path)
                    for f in fnmatch.filter(files, '*.wav')]

    
    test_transcript_path = list(map(lambda wav_path :
                                    os.path.splitext(wav_path)[0] + '.trn',
                                    test_wav_path))

    with open('test_manifest_aidatatang_200zh.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(test_wav_path)):
            writer.writerow([test_wav_path[i], test_transcript_path[i]])


extract_data(data_folder_path)
generate_csv(data_folder_path)
