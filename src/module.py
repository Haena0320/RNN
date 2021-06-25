import torch
import torch.nn as nn
import torch.nn.functional as F
# attention
# decoder (lstm, attention)
# encoder (lstm)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_dim,lstm_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, lstm_dim, lstm_layers, batch_first=True, dropout=dropout)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        for layer in self.lstm.all_weights:
            for weight in layer:
                if weight.ndim == 2:
                    weight.data.uniform_(-0.1,0.1) # weight matrix
                else:
                    weight.data.fill_(0) # bias
        return None

    def forward(self, x, hidden):
        output =self.embedding(x)
        output =  self.dropout(output)
        output ,h = self.lstm(output, hidden)
        return output, h

class Decoder(nn.Module):
    def __init__(self, vocab_size,embed_dim, lstm_dim,lstm_layer, dropout, max_sent,input_feed, device):
        super(Decoder, self).__init__()
        self.input_feed = input_feed
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if input_feed:
            self.lstm = nn.LSTM(embed_dim+lstm_dim, lstm_dim, lstm_layer,batch_first=True ,dropout=dropout)
        else:
            self.lstm = nn.LSTM(embed_dim, lstm_dim, lstm_layer,batch_first=True ,dropout=dropout)
        self.attention = Attention(lstm_dim, max_sent,input_feed, device)
        self.linear = nn.Linear(2*lstm_dim, lstm_dim, bias=False)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        for layer in self.lstm.all_weights:
            for weight in layer:
                if weight.ndim == 2:
                    weight.data.uniform_(-0.1, 0.1) # weight matrix
                else:
                    weight.data.fill_(0) # bias

        self.attention.init_weights()
        self.linear.weight.data.uniform_(-0.1, 0.1)


    def forward(self, x, prev_hht, hidden, encoder_outputs):
        x = self.embedding(x)
        x = self.dropout(x)
        if self.input_feed:
            x = torch.cat((x, prev_hht), dim=2)

        h_t, hidden = self.lstm(x, hidden)
        attention_weights = self.attention(h_t, encoder_outputs)
        context = torch.bmm(attention_weights, encoder_outputs)
        output = torch.cat((context, h_t), dim=-1)
        output = torch.tanh(self.linear(output))
        return output, hidden


class Attention(nn.Module):
    def __init__(self, lstm_dim, max_sent,input_feed, device):
        super(Attention, self).__init__()
        self.att_score = nn.Linear(lstm_dim, max_sent, bias=False)
        self.input_feed = input_feed
        self.device = device

    def init_weights(self):
        self.att_score.weight.data.uniform_(-0.1, 0.1)

    def forward(self,h_t, h_s):
        """
        :param h_s: (bs, seq_s, dim)
        :param h_t: (bs, 1, dim)
        :param mask: (bs, seq_s, dim)
        :return:
        """
        bs, max_len, _ = h_s.shape
        attn_weight = self.att_score(h_t) # (bs, seq_s, seq_s)

        if self.input_feed:
            attn_weight = score.view(bs, 1, max_len)

        attn_weight =F.softmax(attn_weight, dim=-1) #(bs , seq_s)
        return attn_weight


if __name__ =='__main__':
    encoder = Encoder(3000, 1000)
    decoder = Decoder(3000, 1000,4, method="general")

    encoder_x = torch.arange(0, 1000).view(20, -1).long()
    decoder_x = torch.arange(0, 1000).view(20,-1).long() # (bs, 1)

    h_s, _ = encoder(encoder_x)
    context = decoder(h_s,decoder_x,encoder_x)
    print(torch.argmax(context, dim=-1))
    print(decoder_x)


