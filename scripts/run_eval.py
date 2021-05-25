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

args = parser.parse_args()
assert args.model in ['base', "large"]
assert args.dataset in ["en_de", "en_fr"]

config = load_config(args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")
print("current device {}".format(device))
## model load -> valid -> best_ckpnt
checkpoint = torch.load("/user15/workspace/RNN/log/g/ckpntckpnt_1", map_location=device)
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
sp = spm.SentencePieceProcessor()
sp.Load("word_piece_encoding.model")

import os
eval_dir = "./log/g/eval/"
pred_f = os.path.join(eval_dir, "predict.txt")
truth_f = os.path.join(eval_dir, "truth.txt")
pred = open(pred_f, "w")
truth = open(truth_f, "w")

for data_iter in tqdm(data_loader):
    encoder_input = data_iter["encoder"].to(device)
    decoder_input = data_iter["decoder"].to(device)

    y = model(encoder_input, decoder_input, predict=True) #(bs, seq)
    tokens = y.tolist()
    for i, token in enumerate(tokens):
        for j in range(len(token)):
            if token[j] == 2:
                token = token[:j]
                break
        decode_tokens = sp.DecodeIds(token)
        mask = 1-decoder_input[i,:].eq(2).float()
        decoder_input[i,:] *= mask.long()
        decode_truth = sp.DecodeIds(decoder_input[i,1:].tolist())
        # write txt file
        pred.write(decode_tokens+"\n")
        truth.write(decode_truth+"\n")

pred.close()
truth.close()
print("prediction finished..")
############################################ blue score calculation ####################################################
import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang="du")


p = open(pred_f, "r")
t = open(truth_f, "r")

pred = [i.replace("\n", '') for i in p.readlines()]
pred = [md.detokenize(i.strip().split()) for i in pred]
truth = [i.replace("\n", '') for i in t.readlines()]
truth = [md.detokenize(i.strip().split()) for i in truth]

print("sample 1st pred sentence:", pred[:10])
print("sample 1st truth sentence:", truth[:10])

assert len(pred) == len(truth)
print(len(pred))
print(len(truth))

total_bleu = []
for i in range(len(pred)):
    bleu = sacrebleu.corpus_bleu(pred[i],truth[i])
    total_bleu.append(bleu.score)

print('-----------------------------------------------------------------------------------------------------------------')
print("total bleu :{}".format(sum(total_bleu)/len(total_bleu)))