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
        self.criterion = nn.CrossEntropyLoss(ignore_index=3)
        self.cnt = 0
        
    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, en_input, de_input, predict=False):
        """
        :param en_input: [4,3,2,1]
        :param de_input: [0,1,2,3]
        :param predict: [1,2,3,4]
        :return:
        """

        h_s, _ = self.encoder(en_input)
        attn_mask = en_input.eq(0).float() # (bs, seq) = (128, 50)

        if predict == False:
            self.cnt +=1
            truth = de_input[:, 1:].contiguous()

            pred = self.decoder(h_s, de_input, attn_mask)
            pred = pred[:, :-1].contiguous()
            bs, seq,_ = pred.size()
            loss = self.criterion(pred.view(bs*seq, -1), truth.view(-1))

            pred_ = torch.argmax(pred, dim=-1)

            truth_ = sum(pred_.eq(truth).float()) / sum(1-truth.eq(0).float())
            truth_ = torch.mean(truth_*100)

            return loss, truth_

        else: # predict mode
            h_t = de_input[:,0] #(bs)
            h_t = h_t.unsqueeze(1) #(bs, 1)
            for idx in range(self.max_sent):
                output = self.decoder.search(h_s, h_t, attn_mask) #(bs, n, vocab)
                output = torch.argmax(output, dim=-1) #(bs, n)
                if len(output.size()) > 1:
                    output = output[:,-1].unsqueeze(1)
                else:
                    output = output.unsqueeze(1)
                h_t = torch.cat([h_t, output], dim=-1)
            return h_t[:, 1:].long() #(bs, max_sent-1)












