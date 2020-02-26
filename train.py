import os
import json
import torch
import numpy as np

from model import DeepSpeech
from data_loader import SpeechDataset, SpeechDataloader

def set_deterministic(seed=123456):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def train(epochs=10):

    with open('lexicon.json') as label_file:
        labels = str(''.join(json.load(label_file)))

    train_dataset = SpeechDataset('train.csv') 
    train_loader = SpeechDataloader(train_dataset) 
    print('opened dataset')
    model = DeepSpeech(len(labels))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            feature, label = data
            predict = model(data)
            loss = criterion(predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
        save_path = os.path.join(os.getcwd(), 'checkpoints', 'model.{}'.format(epoch))
        torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch), save_path)


if __name__ == '__main__':
    train()
