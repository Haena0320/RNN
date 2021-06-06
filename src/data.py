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
    data = open(input_file, 'r')
    v = list()
    for line in data.readlines():
        line_ = line.strip().replace("\n", "").split()
        v.extend(line_)
    word2idx = {w: i for i, w in enumerate(v)}
    torch.save(word2idx, vocab_path)


def data_prepro(input_path, save_path, model_path):
    f = open(input_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    sp.SetEncodeExtraOptions('reverse:bos:eos')
    ids = [np.array(sp.EncodeAsIds(line)) for line in f.readlines()]
    torch.save(ids, save_path)
    logging.info("data save ! ")

def get_data_loader(data_list, batch_size, shuffle=True, num_workers=3, drop_last=True):
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

        self.encoder_input = np.array(self.encoder_input)
        self.decoder_input = np.array(self.decoder_input)
        assert len(self.encoder_input) == len(self.decoder_input)

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return {'encoder':torch.LongTensor(self.encoder_input[idx]), "decoder":torch.LongTensor(self.decoder_input[idx])}


# #######################################################################################################################
# with open("/hdd1/user15/workspace/RNN/data/raw/en_de/vocab/vocab.50K.de.txt") as f:
#     de_vocab = f.readlines()
#     de_vocab = [de.split('\n')[0] for de in de_vocab]
#     vocab = dict()
#     vocab["[pad]"] = 0
#     vocab["[bos]"] = 1
#     vocab['[eos]'] = 2
#     vocab["[unk]"] = 3
#     for d in de_vocab:
#         vocab[d] = len(vocab)
#     torch.save(vocab, "/hdd1/user15/workspace/RNN/data/prepro/en_de/de_vocab.pkl")
#
# with open("/hdd1/user15/workspace/RNN/data/raw/en_de/vocab/vocab.50K.en.txt") as f:
#     en_vocab = f.readlines()
#     en_vocab = [en.split('\n')[0] for en in en_vocab]
#     vocab = dict()
#     vocab["[pad]"] = 0
#     vocab["[bos]"] = 1
#     vocab['[eos]'] = 2
#     vocab["[unk]"] = 3
#     for d in en_vocab:
#         vocab[d] = len(vocab)
#     torch.save(vocab, "/hdd1/user15/workspace/RNN/data/prepro/en_de/en_vocab.pkl")
#
#
# vocab = torch.load("/hdd1/user15/workspace/RNN/data/prepro/en_de/en_vocab.pkl")
#
# with open("/hdd1/user15/workspace/RNN/data/raw/en_de/test/newstest2014.en.txt") as f:
#     en_test = f.readlines()
#     en_test = [en.split("\n")[0] for en in en_test]
#     en_test_prepro = list()
#     for en in en_test:
#         new_list =[vocab.get("[bos]")]
#         for e in en.split()[::-1]:
#             if e in vocab:
#                 new_list.append(vocab.get(e))
#             else:
#                 continue
#         new_list += [vocab.get("[eos]")]
#
#         if len(new_list) > 50:
#             new_list = new_list[:50]
#         else:
#             pad_ = 50 - len(new_list)
#             new_list += pad_*[0]
#         assert len(new_list) == 50
#         en_test_prepro.append(new_list)
#
# torch.save(en_test_prepro, "/hdd1/user15/workspace/RNN/data/prepro/en_de/test/test.en.pkl")
#
#
# with open("/hdd1/user15/workspace/RNN/data/raw/en_de/train/train.en.txt") as f:
#     en_test = f.readlines()
#     en_test = [en.split("\n")[0] for en in en_test]
#     en_test_prepro = list()
#     for en in en_test:
#         new_list = [vocab.get("[bos]")]
#         for e in en.split()[::-1]:
#             if e in vocab:
#                 new_list.append(vocab.get(e))
#             else:
#                 continue
#         new_list += [vocab.get("[eos]")]
#
#         if len(new_list) > 50:
#             new_list = new_list[:50]
#         else:
#             pad_ = 50 - len(new_list)
#             new_list += pad_*[0]
#         assert len(new_list) == 50
#         en_test_prepro.append(new_list)
# torch.save(en_test_prepro, "/hdd1/user15/workspace/RNN/data/prepro/en_de/train/train.en.pkl")
#
#
# vocab = torch.load("/hdd1/user15/workspace/RNN/data/prepro/en_de/de_vocab.pkl")
#
# with open("/hdd1/user15/workspace/RNN/data/raw/en_de/test/newstest2014.de.txt") as f:
#     en_test = f.readlines()
#     en_test = [en.split("\n")[0] for en in en_test]
#     en_test_prepro = list()
#     for en in en_test:
#         new_list = [vocab.get("[bos]")]
#         for e in en.split()[::-1]:
#             if e in vocab:
#                 new_list.append(vocab.get(e))
#             else:
#                 continue
#         new_list += [vocab.get("[eos]")]
#         if len(new_list) > 50:
#             new_list = new_list[:50]
#         else:
#             pad_ = 50 - len(new_list)
#             new_list += pad_*[0]
#         assert len(new_list) == 50
#         en_test_prepro.append(new_list)
# torch.save(en_test_prepro, "/hdd1/user15/workspace/RNN/data/prepro/en_de/test/test.de.pkl")
#
# with open("/hdd1/user15/workspace/RNN/data/raw/en_de/train/train.de.txt") as f:
#     en_test = f.readlines()
#     en_test = [en.split("\n")[0] for en in en_test]
#     en_test_prepro = list()
#     for en in en_test:
#         new_list = [vocab.get("[bos]")]
#         for e in en.split()[::-1]:
#             if e in vocab:
#                 new_list.append(vocab.get(e))
#             else:
#                 continue
#         new_list += [vocab.get("[eos]")]
#         if len(new_list) > 50:
#             new_list = new_list[:50]
#         else:
#             pad_ = 50 - len(new_list)
#             new_list += pad_*[0]
#         assert len(new_list) == 50
#         en_test_prepro.append(new_list)
# torch.save(en_test_prepro, "/hdd1/user15/workspace/RNN/data/prepro/en_de/train/train.de.pkl")
# #######################################################################################################################