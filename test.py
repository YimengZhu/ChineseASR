import argparse
import json
from tqdm import tqdm
from Levenshtein import distance
from data_loader import SpeechDataset, SpeechDataloader
from decoder import GreedyDecoder
from model import DeepSpeech, DeepSpeechTransformer, DeepTransformer
import torch
# from torch.utils.tensorboard import SummaryWriter
from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DeepSpeech')

def evaluate(model, test_loader, decoder):
    cer, num_char = 0.0, 0
    for data in tqdm(test_loader, total=len(test_loader)):
        feature, label, spect_lengths, transcript_lengths = data
        split_label = []
        offset = 0
        for length in transcript_lengths:
            split_label.append(label[offset:offset + length])
            offset += length

        predict, pred_lengths = model(feature.cuda(), spect_lengths.cuda())
        predict_str = decoder.decode(predict, pred_lengths)
        label_str = decoder.label2string(split_label)

        for j, true_transcript in enumerate(label_str):
            pred_transcript = predict_str[j]
            cer += distance(pred_transcript, true_transcript)
            num_char += len(true_transcript)

    avg_cer = cer / num_char
    # print('number of test samples: {}, number of characters: {}, everage cer: {}.'.
    #      format(len(test_dataset), num_char, avg_cer), flush=True)
    return avg_cer


if __name__ == '__main__':
    args = parser.parse_args()

    # writer = SummaryWriter('runs/{}'.format(args.model))

    with open('lexicon.json') as label_file:
        labels = str(''.join(json.load(label_file)))


    if args.model == 'DeepSpeech':
        model = DeepSpeech(800, len(labels)).cuda()
    elif args.model == 'DeepSpeechTransformer':
        model = DeepSpeechTransformer(len(labels)).cuda()
    elif args.model == 'DeepTransformer':
        model = DeepTransformer(len(labels)).cuda()

    # print(model, flush=True)
    print('Number of trained parameter: {}'.
          format(sum(p.numel() for p in model.parameters() if
                     p.requires_grad)), flush=True)

    decoder = GreedyDecoder()

    for i in range(29):
        train_loader = SpeechDataloader(SpeechDataset('uf.csv'),
                                        batch_size=4)
        model_path = 'checkpoints_{}'.format(args.model) + '/model{}.pt'.format(i)
        model.load_state_dict(torch.load(model_path)['model'])
        #model.eval()
        cer = evaluate(model, train_loader, decoder)
        print(cer)

    print('==============')

    #test_loader = SpeechDataloader(SpeechDataset('test.csv'), batch_size=8)
    #for i in range(29):
    #    model_path = 'checkpoints_{}'.format(args.model) + '/model{}.pt'.format(i)
    #    model.load_state_dict(torch.load(model_path)['model'])
    #    decoder = GreedyDecoder()
    #    #model.eval()
    #    cer = evaluate(model, test_loader, decoder)
    #    print(cer)
