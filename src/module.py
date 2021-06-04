import torch
import torch.nn as nn
import torch.nn.functional as F
# attention
# decoder (lstm, attention)
# encoder (lstm)

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden=1000, num_layers=4):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers, dropout=0.2)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        output =self.embedding(x)
        output ,h = self.lstm(output)
        return output, h

class Decoder(nn.Module):
    def __init__(self, vocab_size=None, hidden=1000, max_sent=50, num_layers=4,  method = None):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers, dropout=0.2)
        self.attn = Attention(method, hidden, max_sent)
        self.linear = nn.Linear(2*hidden, hidden)
        self.activation = nn.Tanh()
        self.span = nn.Linear(hidden, vocab_size)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.span.weight.data.uniform_(-0.1, 0.1)

    def forward(self, h_s, decoder_input, attn_mask=None):
        h_t, _ = self.lstm(self.embedding(decoder_input))
        context = self.attn(h_s, h_t, mask=attn_mask)
        output = torch.cat([context, h_t], dim=-1)
        output = self.activation(self.linear(output))
        output = self.span(output)
        return output


    def search(self,h_s,decoder_input, attn_mask=None):
        """
        :param decoder_input: (bs, 1)
        :param h_s: (bs, seq, dim)
        :param attn_mask : (bs, seq)
        """

        h_t, _ = self.lstm(self.embedding(decoder_input)) # h_t : (bs, 1, dim)
        context = self.attn(h_s, h_t, mask=attn_mask) # (bs,1,dim)
        output = torch.cat([context,h_t], dim=-1) #(bs, 1, 2*dim)
        output = self.activation(self.linear(output)) # (bs, 1, dim)
        output = self.span(output) #(bs, 1, vocab)
        return output.squeeze(1) #(bs, vocab)

class Attention(nn.Module):
    def __init__(self, method, hidden, max_sent=None):
        super(Attention, self).__init__()
        self.method = method
        if method =="location":
            self.linear = nn.Linear(hidden, max_sent+1, bias=False)
        else:
            self.linear = nn.Linear(hidden, hidden, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_s, h_t, mask=None):
        """
        :param h_s: (bs, seq_s, dim)
        :param h_t: (bs, 1, dim)
        :param mask: (bs, seq_s, dim)
        :return:
        """
        if self.method =="location":
            attn_weight = self.linear(h_t) # (bs, seq_s, seq_s)
        else: # general
            attn_weight = self.linear(h_s) #(bs, seq_s, dim)
            query = h_t.transpose(1,2).contiguous() #(bs, dim, 1)
            attn_weight = torch.bmm(attn_weight, query).squeeze(-1) #(bs, seq_s)
            
        if mask is not None:  # padding -> -inf
            attn_mask = mask.unsqueeze(1)*(-1e10) # (bs, 1, seq_s)
            attn_weight = attn_weight + attn_mask

        attn_weight =self.softmax(attn_weight) #(bs , seq_s)
        context = torch.bmm(attn_weight, h_s) # (bs, seq_s, seq_s)
        return context


if __name__ =='__main__':
    encoder = Encoder(3000, 1000)
    decoder = Decoder(3000, 1000,4, method="general")

    encoder_x = torch.arange(0, 1000).view(20, -1).long()
    decoder_x = torch.arange(0, 1000).view(20,-1).long() # (bs, 1)

    h_s, _ = encoder(encoder_x)
    context = decoder(h_s,decoder_x,encoder_x)
    print(torch.argmax(context, dim=-1))
    print(decoder_x)

