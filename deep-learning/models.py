import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)


class RNNTagger(nn.Module):
    def __init__(self, nemb, nhid, nlayers, drop, ntags):
        super(RNNTagger, self).__init__()
        self.tagger_rnn = nn.LSTM(
            input_size=nemb,
            hidden_size=nhid,
            num_layers=nlayers,
            dropout=drop,
            bidirectional=True)
        self.linear = nn.Linear(in_features=nhid * 2, out_features=ntags)

    def forward(self, x, seq_len):
        packed_sequence = pack_padded_sequence(x, seq_len, batch_first=False)
        out, _ = self.tagger_rnn(packed_sequence)
        out, lengths = pad_packed_sequence(out, batch_first=False)
        logits = self.linear(out)
        return logits


class CRFTagger(nn.Module):
    def __init__(self, rnn_tagger, ntags):
        super(CRFTagger, self).__init__()
        self.rnn_tagger = rnn_tagger
        self.crf_tagger = CRF(ntags)

    def forward(self, x, seq_len, y):
        features = self.rnn_tagger(x, seq_len).transpose(1, 0)
        llikelihood = self.crf_tagger(features, y)

        return -llikelihood

    def decode(self, x, seq_len):
        features = self.rnn_tagger(x, seq_len)
        # print(features.size())
        features = features.transpose(1, 0)
        # print(features.size())
        result = self.crf_tagger.decode(features)

        return result
