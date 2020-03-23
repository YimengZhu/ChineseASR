import os
import json
import torch
import numpy as np
import argparse

from model import DeepSpeech, DeepSpeechTransformer
from data_loader import SpeechDataset, SpeechDataloader
from utils import train_log
from pdb import set_trace as bp


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DeepSpeech')

def set_deterministic(seed=123456):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def train(model='DeepSpeech', epochs=10):

    with open('lexicon.json') as label_file:
        labels = str(''.join(json.load(label_file)))

    train_dataset = SpeechDataset('train.csv')
    train_loader = SpeechDataloader(train_dataset, batch_size=8)

    if model=='DeepSpeech':
        model = DeepSpeech(len(labels)).cuda()
    else:
        model = DeepSpeechTransformer(len(labels)).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

    print('initialed')
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            feature, label, spect_lengths, transcript_lengths = data
            predict, pred_lengths = model(feature.cuda(), spect_lengths.cuda())
            predict = predict.float().cpu()
            loss = criterion(predict, label, pred_lengths.cpu(), transcript_lengths)
            if torch.isnan(loss).any():
                print('recieved nan loss.')
            else:
                loss = loss / feature.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch {}, {}/{}, loss: {}'.format(epoch, i,
                                                     len(train_loader),
                                                     loss.item()), flush=True)
        save_path = os.path.join(os.getcwd(), 'checkpoints', 'model{}.pt'.format(epoch))
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    args = parser.parse_args()
    train(model=args.model)
