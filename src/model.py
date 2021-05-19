# seq2seq
import sys, os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.funcional as F
from src.module import *

class Seq2seq(nn.Module):
    def __init__(self, config, args, device):
        self.device = device
        vocab_size = config.model.vocab_size
        hidden = config.lstm.hidden
        num_layers = config.lstm.num_layers
        max_sent = config.modeld.max_sent_len
        method = args.method

        self.encoder = Encoder(vocab_size, hidden, num_layers)
        self.decoder = Decoder(vocab_size, hidden, max_sent, num_layers, method)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, en_input, de_input, predict=False):
        h_s, _ = self.encoder(en_input)

        attn_mask = en_input.eq(0).float()*(-2**32)
        if predict == False:
            total_loss = 0
            for idx in range(len(de_input)):
                h_t = de_input[idx]
                output = self.decoder(h_s, h_t, attn_mask)
                loss = self.criterion(output,h_t)
                total_loss += loss

            return total_loss

        else: # predict mode
            h_t = de_input[0]
            total_prediction = torch.zeros(h_t.size(0))

            for idx in range(max_sent):
                output = self.decoder(h_s, h_t, attn_mask)
                torch.cat([total_prediction, output],dim=-1)
                h_t = output


            return total_prediction[1:,:]












