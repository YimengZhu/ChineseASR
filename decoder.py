import json
import torch
from pdb import set_trace as bp


class Decoder:
    def  __init__(self):
        with open('lexicon.json') as label_file:
            self.labels = str(''.join(json.load(label_file)))
        self.int2char = dict([(i, c) for (i, c) in enumerate(self.labels)])

    def decode(self, probs, prob_lengths):
        raise NotImplementedError

    def label2string(self, label_tensor_list):
        raise NotImplementedError


class GreedyDecoder(Decoder):

    def label2string(self, label_tensor_list):
        strings = []
        for label_tensor in label_tensor_list:
            string = ''
            size = len(label_tensor)
            for i in range(size):
                char = self.int2char[label_tensor[i].item()]
                if i != 0 and char == self.int2char[label_tensor[i - 1].item()]:
                    pass
                else:
                    string += char
            string = string.replace(' ', '')
            strings.append(string)
        return strings

    def decode(self, probs, prob_lengths):
        #prob.shape = T * N * D
        _, max_probs = torch.max(probs, 2)
        max_probs = max_probs.transpose(1, 0)
        prob_list = []
        for i, sample in enumerate(max_probs):
            max_prob = sample[:prob_lengths[i]]
            prob_list.append(max_prob)
        decoded_transcript = self.label2string(prob_list)
        return decoded_transcript


class BeamDecoder(Decoder):
    def __init__(self):
        super(BeamDecoder, self).__init__()
        try:
            from ctcdecode import CTCBeamDecoder
            lm_path = 'zh_giga.no_cna_cmn.prune01244.klm'
            alpha, beta, cutoff_top_n, cutoff_prob = 0, 0, 40, 1.0
            beam_width, num_proc,blank_dix = 100, 4, 0
            self.decoder = CTCBeamDecoder(self.labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,num_proc, blank_idx)
        except ImportError:
            raise ImportError('please install https://github.com/parlance/ctcdecode')

    def label2string(self, out, seq_len):
        result = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int2char[x.item()], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def decode(self, probs, prob_lengths):
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)
        decoded_transcript = self.label2string(out, seq_lens)
        return decoded_transcript
