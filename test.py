from tqdm import tqdm
from Levenshtein import distance
from data_loader import SpeechDataset, SpeechDataloader
from decoer import GreedyDecoder

def evaluate(model, decoder):
    test_dataset = SpeechDataset('train.csv')
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
            pred_transcript.replace(' ', '')
            true_transcript.replace(' ', '')
            cer += distance(pred_transcript, true_transcript)
            num_char += len(true_transcript)

    print('number of test samples: {}, number of characters: {}, everage cer{}.'.
          format(len(test_dataset), num_char, cer / num_char))
