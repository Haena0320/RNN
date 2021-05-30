import sentencepiece as spm
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

def encoding(input_file, vocab_size, vocab_path, model_name,model_type):
    pad=0
    bos=1
    eos=2
    unk=3
    input_argument ="--input=%s --model_prefix=%s --vocab_size=%s --pad_id=%s --bos_id=%s --eos_id=%s --unk_id=%s --model_type=%s"
    cmd = input_argument % (input_file, model_name, vocab_size, pad, bos, eos, unk, model_type)

    spm.SentencePieceTrainer.Train(cmd)
    logging.info("model, vocab finished ! ")
    f = open(model_name+".vocab", encoding='utf-8')
    v = [clean_str(doc).strip().split("\t") for doc in f]
    word2idx = {w[0]: i for i, w in enumerate(v)}
    torch.save(word2idx, vocab_path)

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\??", " ?? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string


def data_prepro(input_path, save_path, model_path):
    f = open(input_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    sp.SetEncodeExtraOptions('reverse:bos:eos')
    ids = [np.array(sp.EncodeAsIds(line)) for line in f.readlines()]
    torch.save(ids, save_path)
    logging.info("data save ! ")

def get_data_loader(data_list, batch_size, shuffle=True, num_workers=10, drop_last=True):
    dataset = Make_Dataset(data_list)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=make_padding)
    return data_loader

def make_padding(samples):
    def padd(samples):
        length = [len(s) for s in samples]
        max_length = 51
        batch = torch.zeros(len(length), max_length).to(torch.long)
        for idx, sample in enumerate(samples):
            if length[idx] < max_length:
                batch[idx, :length[idx]] = torch.LongTensor(sample)
            else:
                batch[idx, :max_length] = torch.LongTensor(sample[:max_length])

        return torch.LongTensor(batch)
    encoder = [sample["encoder"] for sample in samples]
    decoder = [sample["decoder"] for sample in samples]
    encoder = padd(encoder)
    decoder = padd(decoder)
    return {"encoder":encoder.contiguous(), "decoder":decoder.contiguous()}

class Make_Dataset(Dataset):
    def __init__(self, path):
        self.encoder_input = torch.load(path[0])
        self.decoder_input = torch.load(path[1])

        self.encoder_input = np.array(self.encoder_input, dtype=object)
        self.decoder_input = np.array(self.decoder_input, dtype=object)

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return {'encoder':torch.LongTensor(self.encoder_input[idx]), "decoder":torch.LongTensor(self.decoder_input[idx])}

