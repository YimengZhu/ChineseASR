import numpy as np
import fnmatch
import tarfile
import glob
import os
import csv


base_path = '/project/student_projects3/yzhu/data/aishell2'

def extract_data(data_folder):
    print('extracting data in aishell2 folder')

    for targz in glob.glob(os.path.join(data_folder, 'data', 'wav', '*.tar.gz')):
        print('extracting {}'.format(targz))
        with tarfile.open(targz) as tar:
            tar.extractall(os.path.dirname(targz))
            os.remove(targz)


def generate_trn(data_folder):
    transcripts_path = os.path.join(data_folder, 'data', 'trans.txt')

    with open(transcripts_path) as fin:
        transcripts = dict(line.split('\t', maxsplit=1) for line in fin)

    wav_path = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(data_folder)
                for f in fnmatch.filter(files, '*.wav')]

    for wav in wav_path:
        trn = os.path.splitext(wav)[0] + '.trn'
        print('genearting trn files for {}'.format(trn))
        with open(trn, 'w') as ftrn:
            wav_name = os.path.splitext(os.path.basename(wav))[0]
            try:
                ftrn.write(transcripts[wav_name].strip('\n'))
            except KeyError:
                os.rename(wav, wav + '.nolabel')
                print('Warning: Missing transcript for WAV file {}.'.format(wav))


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
 
    with open('train_manifest_aishell2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(train_indices)):
            writer.writerow([wav_path[train_indices[i]], trn_path[train_indices[i]]])

    with open('dev_manifest_aishell2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(dev_indices)):
            writer.writerow([wav_path[dev_indices[i]], trn_path[dev_indices[i]]])

    with open('val_manifest_aishell2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(val_indices)):
            writer.writerow([wav_path[val_indices[i]], trn_path[val_indices[i]]])


def process(data_path):
    extract_data(data_path)
    generate_trn(data_path)
    generate_csv(data_path)


if __name__ == '__main__':
    process(base_path)
