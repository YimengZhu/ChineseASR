import argparse
import json
from tqdm import tqdm
from Levenshtein import distance
from data_loader import SpeechDataset, SpeechDataloader
from decoder import GreedyDecoder
from model import DeepSpeech, DeepSpeechTransformer
import torch
# from torch.utils.tensorboard import SummaryWriter
from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DeepSpeech')
parser.add_argument('--model_folder', default='checkpoints')

def evaluate(model, decoder):

    test_dataset = SpeechDataset('test.csv')
    test_loader = SpeechDataloader(test_dataset, batch_size=8)

    cer, num_char = 0, 0
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
            num_char += max(len(true_transcript), len(pred_transcript))

    avg_cer = cer / num_char
    print('number of test samples: {}, number of characters: {}, everage cer{}.'.
          format(len(test_dataset), num_char, avg_cer))

    return avg_cer


if __name__ == '__main__':
    args = parser.parse_args()

    # writer = SummaryWriter('runs/{}'.format(args.model))

    with open('lexicon.json') as label_file:
        labels = str(''.join(json.load(label_file)))

    if args.model == 'DeepSpeech':
        model = DeepSpeech(len(labels)).cuda()
    elif args.model == 'DeepSpeechTransformer':
        model = DeepSpeechTransformer(len(labels)).cuda()

    # print(model, flush=True)
    print('Number of trained parameter: {}'.
          format(sum(p.numel() for p in model.parameters() if
                     p.requires_grad)), flush=True)

    for i in range(4, 5):
        model_path = args.model_folder + '/model{}.pt'.format(i)
        model.load_state_dict(torch.load(model_path))
        # model.eval()
        decoder = GreedyDecoder()
        cer = evaluate(model, decoder)
        # writer.add_scalar('acc', cer, i)
