import nltk.translate.bleu_score as bleu
import sys, os
sys.path.append(os.getcwd())
import torch
import argparse
from src.utils import *
from src.data import *
from tqdm import tqdm
import sentencepiece as spm
from src.model import *

parser = argparse.ArgumentParser()
parser.add_argument("--mode",type=str, default="><")
parser.add_argument("--dataset", type=str, default="en_de")
parser.add_argument("--model", type=str, default="base")
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--total_steps", type=int, default=10000)
parser.add_argument("--eval_period", type=int, default=1000)
parser.add_argument("--method", type=str, default="location")
parser.add_argument("--load", type=str, default="model.1.ckpt")
parser.add_argument("--input_feed", type=int, default=0)

args = parser.parse_args()
config = load_config(args.config)
assert args.model in ['base', "large"]
assert args.dataset in ["en_de", "en_fr"]

config = load_config(args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")
print("current device {}".format(device))
## model load -> valid -> best_ckpnt
checkpoint = torch.load("/hdd1/user15/workspace/RNN/log/g/ckpntckpnt_8", map_location=device)
model  = Seq2seq(config, args, device)
model.to(device)
model.load_state_dict(checkpoint["model_sate_dict"])
model.eval()
## data load -> test data load
data = config.data_info
data_list = [data.prepro_te_en, data.prepro_te_de]
data_loader = get_data_loader(data_list, config.train.bs)

# 1(sos) + token_id, token_id, token_id, + 2(eos)
## decoding

## bleu score calculation

import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang="du")


sp = torch.load("./data/prepro/en_de/de_vocab.pkl")
sp = {v:k for k,v in sp.items()}

import os
# eval_dir = "./log/g/eval/"
# pred_f = os.path.join(eval_dir, "predict.txt")
# truth_f = os.path.join(eval_dir, "truth.txt")
# pred = open(pred_f, "w")
# truth = open(truth_f, "w")
bs = config.train.bs
lstm_dim = config.lstm.hidden
lstm_layer = config.lstm.num_layers
max_sent = config.model.max_sent_len

total_bleu = []

for data_iter in tqdm(data_loader):
    en_x = data_iter["encoder"].to(device)
    dn_x = data_iter["decoder"].to(device)

    h_0 = torch.zeros(lstm_layer, bs, lstm_dim).to(device)
    c_0 = torch.zeros(lstm_layer, bs, lstm_dim).to(device)
    hidden = (h_0, c_0)

    hht = torch.zeros(bs, 1, lstm_dim).to(device)
    de_x = torch.ones(bs, 1).to(torch.long).to(device)
    hidden = [state.to(device) for state in hidden]

    for i in range(max_sent):
        out = model(en_x, de_x, hidden, hht)
        out = out.to(device)
        pred = torch.max(out, dim=-1)[1]
        de_x = torch.cat((de_x, pred[:, i].unsqueeze(1)), dim=1)

    pred_tokens = de_x[:, 1:]
    pred_tokens = pred_tokens.tolist()

    for i, token in enumerate(pred_tokens):
        for j in range(len(token)):
            if token[j] == 2:
                token = token[:j]
                break
            if token[j] == 8:
                token = token[:j+1]
                break
        decode_tokens = [sp[t] for t in token]
        decode_tokens = " ".join(decode_tokens)
        print(decode_tokens)

        decode_truth = dn_x[i, :].tolist()

        for idx in range(len(decode_truth)):
            if decode_truth[idx] == 2:
                decode_truth = decode_truth[1:idx]
                break
        decode_truth = [sp[t] for t in decode_truth]
        decode_truth = " ".join(decode_truth)
        print(decode_truth)
        pred = md.detokenize(decode_tokens.strip().split())
        truth = md.detokenize(decode_truth.strip().split())
        bleu = sacrebleu.corpus_bleu(pred, truth)
        print(bleu.score)
        print("---------------------------------------------------------")
        total_bleu.append(bleu.score)

print("total bleu :{}".format(sum(total_bleu)/len(total_bleu)))
print("prediction finished..")

