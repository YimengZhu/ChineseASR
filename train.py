import os
import json
from tqdm import tqdm
import torch
import numpy as np
import argparse
from models.deepspeech import DeepSpeech
from models.gated_cnn import GatedCNN
from models.attention import SAN
from data.loader import SpeechDataset, SpeechDataloader, StochasticBucketSampler
from decoder import GreedyDecoder
from test import evaluate
from utils import train_log, set_deterministic
from optimizer import TransformerOptimizer
from pdb import set_trace as bp


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DeepSpeech')
parser.add_argument('--from_epoch', type=int, default=0)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--k', type=float, default=0.1)
parser.add_argument('--warmup', type=int, default=8000)
parser.add_argument('--clip_norm', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.6)
parser.add_argument('--num_worker', type=int, default=16)

set_deterministic()

def train(model, train_loader, optimizer,  epochs, scheduler):
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True).cuda()

    for epoch in range(epochs):
        total_loss = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            feature, label, spect_lengths, transcript_lengths = data
            predict, pred_lengths = model(feature.cuda(), spect_lengths.cuda())
            predict = predict.float()
            try:
                loss = criterion(predict, label.cuda(),
                                 pred_lengths, transcript_lengths.cuda())
            except RuntimeError:
                continue
            if torch.isnan(loss).any():
                print('recieved nan loss.')
            else:
                loss = loss / feature.size(0)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                total_loss += float(loss)

            if i % 1000 == 0 and i != 0:
                print('batch {} to {}, loss: {}'.format(i-1000, i, total_loss/1000), flush=True)
                total_loss = 0

        train_acc, test_acc = acc(model)
        print('epoch {}, train_acc: {}, test_acc: {}'.
              format(args.from_epoch+epoch, train_acc, test_acc),
              flush=True)
        if scheduler is not None:
            scheduler.step()
        save_path = os.path.join(os.getcwd(),
                                 'checkpoints_{}'.format(type(model).__name__),
                                 'model{}.pt'.format(args.from_epoch+epoch))
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, save_path)

def acc(model):
    decoder = GreedyDecoder()
    train_loader = SpeechDataloader(SpeechDataset('uf.csv'), batch_size=16)
    train_acc = evaluate(model, train_loader, decoder)
    test_loader = SpeechDataloader(SpeechDataset('test.csv'), batch_size=16)
    test_acc = evaluate(model, test_loader, decoder)
    return train_acc, test_acc


if __name__ == '__main__':
    args = parser.parse_args()

    with open('lexicon.json') as label_file:
        labels = str(''.join(json.load(label_file)))

    model_dict = {
        'DeepSpeech': DeepSpeech(750, len(labels)),
        #'DeepSpeechTransformer': DeepSpeechTransformer(len(labels)),
        'SAN': SAN(len(labels)),
        'GatedCNN': GatedCNN(len(labels))
    }

    model = model_dict.get(args.model, 'DeepSpeech').cuda()

    train_dataset = SpeechDataset('train.csv', augment=args.augment)
    # train_sampler = StochasticBucketSampler(train_dataset,
    #                                        batch_size=args.batch_size)
    train_loader = SpeechDataloader(train_dataset, num_workers=args.num_worker,
                                    batch_size=args.batch_size, shuffle=True)

    if args.model == 'SAN':
        optimizer = TransformerOptimizer(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            scale_factor = args.k,
            warmup_step = args.warmup
        )
        scheduler = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)
    if args.from_epoch != 0:
        model_path =  'checkpoints_{}/model{}.pt'.format(args.model, args.from_epoch - 1)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(model, flush=True)
    print('Number of trained parameter: {}'.
          format(sum(p.numel() for p in model.parameters() if
                     p.requires_grad)), flush=True)

    train(model, train_loader, optimizer, args.epochs, scheduler)
