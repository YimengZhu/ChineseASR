from tqdm import tqdm
import subprocess
import torch
import numpy as np


def set_deterministic(seed=123456):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_log(epoch, batch_step, total_step, loss):
    print('Epoch {}, {} / {}, loss: {}'.format(epoch, batch_step, total_step,
                                               loss.item()))


def sort_utt(csv_file):
    utt_label_list = []
    with open(csv_file, 'r') as path_file:
        utt_label_list += path_file.readlines()

    print('reading length...')
    path_length_list = [(data_path, float(subprocess.check_output(
          ['soxi -D \"%s\"' % data_path.split(',')[0]], shell=True)
        )) for data_path in tqdm(utt_label_list, total=len(utt_label_list))]

    print('sorting...')
    path_length_list.sort(key=lambda path_length_pair: path_length_pair[1])
    with open('sorted' + csv_file, 'w') as sorted_file:
        for path_length_pair in tqdm(path_length_list, total=len(path_length_list)):
            sorted_file.write(path_length_pair[0])


if __name__ == '__main__':
    sort_utt('train.csv')
