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
        vocab_size = config.model.vocab_size
        hidden = config.lstm.hidden
        num_layers = config.lstm.num_layers
        self.max_sent = config.model.max_sent_len
        method = args.method

        self.encoder = Encoder(vocab_size, hidden, num_layers)
        self.decoder = Decoder(vocab_size, hidden, self.max_sent, num_layers, method)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, en_input, de_input, predict=False):
        h_s, _ = self.encoder(en_input)
        attn_mask = en_input.eq(0).float()

        if predict == False:
            total_loss = 0
            bs, seq= de_input.size()
            # for idx in range(seq):
            #     h_t = de_input[:,idx]
            #     output = self.decoder(h_s, h_t, attn_mask)
            #     loss = self.criterion(output,h_t)
            #     total_loss += loss
            output = self.decoder(h_s, de_input, attn_mask)
            loss = self.criterion(output.view(bs*seq, -1), de_input.view(-1))
            return loss

        else: # predict mode
            h_t = de_input[:,0]
            total_prediction = torch.zeros(h_t.size(0), device="cuda:0").unsqueeze(1)
            for idx in range(self.max_sent):
                output = self.decoder.search(h_s, h_t, attn_mask)
                output = torch.argmax(output, dim=-1).unsqueeze(1)
                total_prediction= torch.cat([total_prediction, output], dim=-1)
                h_t = output.squeeze(1)

            return total_prediction[1:, :].long()












