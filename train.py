import os
import json
import torch
import numpy as np

from model import DeepSpeech
from data_loader import SpeechDataset, SpeechDataloader
from utils import train_log

def set_deterministic(seed=123456):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def train(epochs=10):

    with open('lexicon.json') as label_file:
        labels = str(''.join(json.load(label_file)))

    train_dataset = SpeechDataset('train.csv')
    train_loader = SpeechDataloader(train_dataset, batch_size=8)

    model = DeepSpeech(len(labels)).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            feature, label, spect_lengths, transcript_lengths = data
            predict, pred_lengths = model(feature.cuda(), spect_lengths.cuda())
            loss = criterion(predict, label.cuda(), pred_lengths, transcript_lengths.cuda())
            loss = loss / feature.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch {}, {}/{}, loss: {}'.format(epoch, i,
                                                     len(train_loader),
                                                     loss.item()), flush=True)
        save_path = os.path.join(os.getcwd(), 'checkpoints', 'model.{}'.format(epoch))
        torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch), save_path)


if __name__ == '__main__':
    train()
