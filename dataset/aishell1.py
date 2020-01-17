import fnmatch
import tarfile
import glob
import os
import csv
from tqdm import tqdm


base_path = '/project/iwslt2014c/EN/student_projects/yzhu/data/aishell1/data_aishell'

def extract_data(data_folder):
    print('extracting data in aishell1 folder')

    for targz in glob.glob(os.path.join(data_folder, 'wav', '*.tar.gz')):
        print('extracting {}'.format(targz))
        with tarfile.open(targz) as tar:
            tar.extractall(os.path.dirname(targz))
            os.remove(targz)


def generate_trn(data_folder):
    transcripts_path = os.path.join(data_folder, 'transcript', 
                                    'aishell_transcript_v0.8.txt')

    with open(transcripts_path) as fin:
        transcripts = dict((line.split(' ', maxsplit=1) for line in fin))

    wav_path = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(data_folder)
                for f in fnmatch.filter(files, '*.wav')]

    print('generating trn files...')
    for wav in tqdm(wav_path):
        trn = os.path.splitext(wav)[0] + '.trn'
        with open(trn, 'w') as ftrn:
            wav_name = os.path.splitext(os.path.basename(wav))[0]
            try:
                ftrn.write(transcripts[wav_name].strip('\n'))
            except KeyError:
                os.rename(wav, wav + '.nolabel')
                print('Warning: Missing transcript for WAV file {}.'.format(wav))

def generate_csv(data_folder):
    train_path = os.path.join(data_folder, 'wav', 'train')
    
    train_wav_path = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(train_path)
                for f in fnmatch.filter(files, '*.wav')]

    train_trn_path = list(map(lambda wav_path : os.path.splitext(wav_path)[0] +
                                                    '.trn', train_wav_path))

    print('generating train_manifest_aishell1.csv')
    with open('train_manifest_aishell1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in tqdm(range(len(train_wav_path))):
            writer.writerow([train_wav_path[i], train_trn_path[i]])

    
    dev_path = os.path.join(data_folder, 'wav', 'dev')

    dev_wav_path = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(dev_path)
                for f in fnmatch.filter(files, '*.wav')]

    dev_trn_path = list(map(lambda wav_path : os.path.splitext(wav_path)[0] +
                                                    '.trn', dev_wav_path))
    
    print('generating dev_manifest_aishell1.csv')
    with open('dev_manifest_aishell1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in tqdm(range(len(dev_wav_path))):
            writer.writerow([dev_wav_path[i], dev_trn_path[i]])


    test_path = os.path.join(data_folder, 'wav', 'test')

    test_wav_path = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(test_path)
                for f in fnmatch.filter(files, '*.wav')]

    test_trn_path = list(map(lambda wav_path : os.path.splitext(wav_path)[0] +
                                                    '.trn', test_wav_path))
    print('generating val_manifest_aishell1.csv')
    with open('val_manifest_aishell1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in tqdm(range(len(test_wav_path))):
            writer.writerow([test_wav_path[i], test_trn_path[i]])


def process(data_path):
    extract_data(data_path)
    generate_trn(data_path)
    generate_csv(data_path)


if __name__ == '__main__':
    process(base_path)
