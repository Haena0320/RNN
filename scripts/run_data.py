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
encoding(data_info.raw_tr_total, data_info.vocab_size, data_info.vocab_path, data_info.model_name, data_info.model_type)

logging.info("make data ! ")
raw = [data_info.raw_tr_de, data_info.raw_tr_en, data_info.raw_te_de, data_info.raw_te_en]
prepro = [data_info.prepro_tr_de, data_info.prepro_tr_en, data_info.prepro_te_de, data_info.prepro_te_en]

for input, output in tqdm(list(zip(raw, prepro))):
    data_prepro(input, output, data_info.model_name+'.model')

print("finished !! ")



