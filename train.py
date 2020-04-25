import os
import json
import torch
from torchvision.transforms import Compose, ToTensor
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse

from model import DeepSpeech, DeepSpeechTransformer, DeepTransformer
from data_loader import SpeechDataset, SpeechDataloader
from data_aug import TimeStretch, SpectAugment
from utils import train_log
from pdb import set_trace as bp


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DeepSpeech')
parser.add_argument('--from_epoch', type=int, default=0)
parser.add_argument('--augment', action='store_true')


def set_deterministic(seed=123456):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def train(model, train_loader, epochs=30):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True).cuda()
    # tbwriter = SummaryWriter('runs/{}'.format(type(model).__name__))

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            feature, label, spect_lengths, transcript_lengths = data
            predict, pred_lengths = model(feature.cuda(), spect_lengths.cuda())
            predict = predict.float()
            loss = criterion(predict, label.cuda(), pred_lengths,
                             transcript_lengths.cuda())
            if torch.isnan(loss).any():
                print('recieved nan loss.')
            else:
                loss = loss / feature.size(0)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 400)
                optimizer.step()
            print('epoch {}, {}/{}, loss: {}'.format(args.from_epoch+epoch, i,
                                                     len(train_loader),
                                                     float(loss)), flush=True)
          # if i / 1000 == 0:
                # writer.add_scalar('training loss', loss, epoch * len(train_loader) + i)

        save_path = os.path.join(os.getcwd(),
                                 'checkpoints_{}'.format(type(model).__name__),
                                 'model{}.pt'.format(args.from_epoch+epoch))
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    args = parser.parse_args()

    with open('lexicon.json') as label_file:
        labels = str(''.join(json.load(label_file)))

    train_dataset = SpeechDataset('train.csv', augment=args.augment)
    train_loader = SpeechDataloader(train_dataset, batch_size=8, num_workers=4,
                                   pin_memory=True)

    if args.model == 'DeepSpeech':
        model = DeepSpeech(800, len(labels)).cuda()
    elif args.model == 'DeepSpeechTransformer':
        model = DeepSpeechTransformer(len(labels)).cuda()
    elif args.model == 'DeepTransformer':
        model = DeepTransformer(len(labels)).cuda()

    if args.from_epoch != 0:
        model_path =  'checkpoints_{}/model{}.pt'.format(args.model, args.from_epoch - 1)
        model.load_state_dict(torch.load(model_path))


    print(model, flush=True)
    print('Number of trained parameter: {}'.
          format(sum(p.numel() for p in model.parameters() if
                     p.requires_grad)), flush=True)

    train(model, train_loader)
