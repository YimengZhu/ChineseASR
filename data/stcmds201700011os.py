import csv
import numpy as np
import fnmatch
import os


base_path = '/project/student_projects3/yzhu/data/ST-CMDS-20170001_1-OS'


def generate_csv(base_path, dev_per=0.1, val_per=0.1):
    wav_path = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(base_path)
                for f in fnmatch.filter(files, '*.wav')]

    trn_path = list(map(lambda wav_path : os.path.splitext(wav_path)[0] +
                        '.txt', wav_path))

    indices = np.arange(0, len(wav_path))
    np.random.seed(12345)
    np.random.shuffle(indices)

    val_indices = indices[int(- len(indices) * val_per) : ] 
    dev_indices = indices[int(- len(indices) * (dev_per + val_per)) : 
                          int(- len(indices) * val_per)]
    train_indices = indices[: int(- len(indices) * (dev_per + val_per))]
 
    with open('train_manifest_freestmandarin.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(train_indices)):
            writer.writerow([wav_path[train_indices[i]], trn_path[train_indices[i]]])

    with open('dev_manifest_freestmandarin.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(dev_indices)):
            writer.writerow([wav_path[dev_indices[i]], trn_path[dev_indices[i]]])

    with open('val_manifest_freestmandarin.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(val_indices)):
            writer.writerow([wav_path[val_indices[i]], trn_path[val_indices[i]]])


if __name__ == '__main__':
    generate_csv(base_path)
