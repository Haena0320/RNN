import sys, os
sys.path.append(os.getcwd())
from src.utils import *
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument("--mode",type=str, default="!!")
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--dataset", type=str, default="en_de")
parser.add_argument("--model", type=str, default='g')
parser.add_argument("--reverse", type=str, default=0) # 0:-> on, 1: off
parser.add_argument("--method", type=str, default="location")
parser.add_argument("--optim", type=str, default="sgd")
parser.add_argument("--use_pretrained", type=int, default=0)

args = parser.parse_args()
config = load_config(args.config)
assert args.model in ["g", "l"]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")

## log directory
oj = os.path.join
log_dir = "./log/"
model_dir = oj(log_dir, args.model)
loss_dir = oj(model_dir,"loss")
ckpnt_dir = oj(model_dir, "ckpnt")
eval_dir = oj(model_dir, "eval")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    os.mkdir(loss_dir)
    os.mkdir(ckpnt_dir)
    os.mkdir(eval_dir)
writer = SummaryWriter(loss_dir)

## data load
from src.data import *
data = config.data_info
train_data = [data.prepro_tr_en, data.prepro_tr_de]

test_data = [data.prepro_te_en, data.prepro_te_de]
train_loader = get_data_loader(train_data, config.train.bs)
test_loader = get_data_loader(test_data, config.train.bs)

## model load
from src.model import Seq2seq
import src.train as train

model = Seq2seq(config, args, device)
model.init_weights()

# trainer load
trainer = train.get_trainer(config, args,device, train_loader, writer, "train")

if args.use_pretrained:
    ck_path = oj(ckpnt_loc, "/ckpnt_{}".format(args.use_pretrained))
    checkpoint = torch.load(ck_path, map_location=device)
    model.load_state_dict(checkpoint.load_state_dict)

    optimizer = train.get_optimizer(model, args.optim)
    optimizer.load_state_dict(checkpoint.optimizer_stata_dict)

    schedular = train.get_lr_scheduler(optimizer, config)
    schedular._step = checkpoint.lr_step

    trainer.init_optimizer(optimizer)
    trainer.init_schedular(schedular)

    total_epoch = checkpoint.epoch
    model.train()

else:
    optimizer = train.get_optimizer(model, args.optim)
    schedular = train.get_lr_scheduler(optimizer, config)

    trainer.init_optimizer(optimizer)
    trainer.init_schedular(schedular)

    total_epoch = 12
    print("total epoch {}".format(total_epoch))

for epoch in tqdm(range(1, total_epoch+1)):
    trainer.train_epoch(model, epoch, save_path=ckpnt_dir)
print('finished...')



