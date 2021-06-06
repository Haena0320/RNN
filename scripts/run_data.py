import sys, os
sys.path.append(os.getcwd())
import torch
import logging
from tqdm import tqdm
from src.utils import *
from src.data import *

config = load_config("default")

logging.info("Building vocab")
data_info = config.data_info
encoding(data_info.raw_de_total, data_info.vocab_size, data_info.vocab_path.de, data_info.model_name.de, data_info.model_type)
encoding(data_info.raw_en_total, data_info.vocab_size, data_info.vocab_path.en, data_info.model_name.en, data_info.model_type)

logging.info("make data ! ")
en_raw = [data_info.raw_tr_en, data_info.raw_te_en]
en_prepro = [data_info.prepro_tr_en, data_info.prepro_te_en]

for input, output in tqdm(list(zip(en_raw, en_prepro))):
    data_prepro(input, output, data_info.model_name.en+'.model')

de_raw = [data_info.raw_tr_de, data_info.raw_te_de]
de_prepro = [data_info.prepro_tr_de, data_info.prepro_te_de]

for input, output in tqdm(list(zip(de_raw, de_prepro))):
    data_prepro(input, output, data_info.model_name.de+'.model')

print("finished !! ")



