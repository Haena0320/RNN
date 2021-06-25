# seq2seq
import sys, os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from src.module import *

class Seq2seq(nn.Module):
    def __init__(self, config, args, device):
        super(Seq2seq, self).__init__()
        self.device = device
        self.vocab_size = config.model.vocab_size
        embed_dim = config.embed.dimension
        lstm_dim = config.lstm.hidden
        lstm_layers = config.lstm.num_layers
        max_sent = config.model.max_sent_len
        self.input_feed = args.input_feed
        dropout = config.model.dropout

        self.encoder = Encoder(self.vocab_size, embed_dim, lstm_dim,lstm_layers, dropout)
        self.decoder = Decoder(self.vocab_size,embed_dim, lstm_dim,lstm_layers, dropout, max_sent,self.input_feed, device)
        self.linear = nn.Linear(lstm_dim, self.vocab_size, bias=False)
        
    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, en_input, de_input, hidden, hht):
        bs, max_len = de_input.size()
        h_s, hidden = self.encoder(en_input, hidden)
        output = torch.zeros(bs, max_len, self.vocab_size)
        if self.input_feed:
            for i in range(max_sent):
                hht, hidden = self.decoder(de_input[:,i].unsqueeze(1), hht, hidden, h_s)
                out = self.linear(hht)
                output[:, i] = out.squeeze()
        else:
            hht, hidden = self.decoder(de_input, hht, hidden, h_s)
            output = self.linear(hht)
        return output















