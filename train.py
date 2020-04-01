import os
import json
import torch
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse

from model import DeepSpeech, DeepSpeechTransformer, DeepTransformer
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

def train(model, train_loader, epochs=10):


    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

    # tbwriter = SummaryWriter('runs/{}'.format(type(model).__name__))

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
                torch.nn.utils.clip_grad_value_(model.parameters(), 400)
                optimizer.step()
            print('epoch {}, {}/{}, loss: {}'.format(epoch, i,
                                                     len(train_loader),
                                                     loss.item()), flush=True)
          # if i / 1000 == 0:
                # writer.add_scalar('training loss', loss, epoch * len(train_loader) + i)

        save_path = os.path.join(os.getcwd(),
                                 'checkpoints_{}'.format(type(model).__name__), 'model{}.pt'.format(epoch))
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    args = parser.parse_args()

    with open('lexicon.json') as label_file:
        labels = str(''.join(json.load(label_file)))

    if args.model == 'DeepSpeech':
        model = DeepSpeech(520, len(labels)).cuda()
        train_dataset = SpeechDataset('train.csv')

    elif args.model == 'DeepSpeechTransformer':
        model = DeepSpeechTransformer(len(labels)).cuda()
        train_dataset = SpeechDataset('train.csv')

    elif args.model == 'Transformer':
        model = DeepTransformer(len(labels)).cuda()
        train_dataset = SpeechDataset('train.csv', feature='mfcc')


    print(model, flush=True)
    print('Number of trained parameter: {}'.
          format(sum(p.numel() for p in model.parameters() if
                     p.requires_grad)), flush=True)

    train_loader = SpeechDataloader(train_dataset, batch_size=8)
    train(model, train_loader)
