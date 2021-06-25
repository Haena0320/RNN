import sys, os
sys.path.append(os.getcwd())
import torch
from torch.cuda import amp
import sacrebleu
from sacremoses import MosesDetokenizer
from tqdm import tqdm
import tqdm
import torch.nn as nn

def get_trainer(config, args, device, data_loader, writer, type):
    return Trainer(config, args, device, data_loader, writer, type)

def get_optimizer(model, args_optim):
    if args_optim =="sgd":
        return torch.optim.SGD(model.parameters(), lr=1)
    else:
        return torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-09)

def get_lr_scheduler(optimizer, config):
    warmup = config.train.warmup
    return Warmupscheduler(optimizer, warmup)

class Warmupscheduler:
    def __init__(self,optimizer, warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self._rate = 1

    def step(self):
        self._step += 1
        if self._step > self.warmup:
            self.rate()
            for p in self.optimizer.param_groups:
                p["lr"] = self._rate

        self.optimizer.step()

    def rate(self):
        self._rate /= 2





class Trainer:
    def __init__(self, config, args, device,data_loader,writer, type):
        self.config = config
        self.args = args
        self.device = device
        self.data_loader = data_loader
        self.type = type
        self.writer = writer
        self.global_step = 0
        self.ckpnt_step = 1000
        self.lstm_layer = config.lstm.num_layers
        self.bs = config.train.bs
        self.lstm_dim = config.lstm.hidden
        self.max_sent = config.model.max_sent_len
        self.gradscaler = amp.GradScaler()
        if self.type != "train":
            self.md = MosesDetokenizer(lang="du")

    def init_optimizer(self,optimizer):
        self.optimizer = optimizer

    def init_scheduler(self, scheduler):
        self.scheduler = scheduler

    def log_writer(self, log, step):
        if self.type =="train":
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss", log,step)
            self.writer.add_scalar("train/lr", lr, step)
        else:
            self.writer.add_scalar("valid/loss", log, step)

    def train_epoch(self, model, epoch, save_path=None, sp=None, md=None):
        if self.type =="train":
            model.train()
            self.criterion = nn.CrossEntropyLoss(ignore_index=2)

        else:
            model.eval()
            total_bleu = list()

        model.to(self.device)

        for iter in tqdm.tqdm(self.data_loader):
            with amp.autocast():
                en_x = iter['encoder'].to(self.device) # 128, 51 [0,4,3,2,1]
                dn_x = iter["decoder"].to(self.device) # 128, 51 [0,1,2,3,4]

                if self.type =="train":
                    h_0 = torch.zeros(self.lstm_layer, self.bs, self.lstm_dim).to(self.device)
                    c_0 = torch.zeros(self.lstm_layer, self.bs, self.lstm_dim).to(self.device)
                    hidden = (h_0, c_0)
                    hidden = [state.detach().to(self.device) for state in hidden]

                    hht = torch.zeros(self.bs, self.max_sent, self.lstm_dim)
                    output  = model(en_x, dn_x, hidden, hht)
                    max_sent = 50
                    loss = self.criterion(output[:,:-1, :].reshape(self.bs*max_sent, -1), dn_x[:,1:].reshape(-1))

                    self.global_step += 1
                    if self.global_step % self.ckpnt_step ==0:
                        torch.save({"epoch":epoch,
                                    "model_sate_dict":model.state_dict(),
                                    "optimizer_state_dict":self.optimizer.state_dict(), 
                                    "lr_step":self.scheduler._step},
                                   save_path+'ckpnt_{}'.format(epoch))

                    self.log_writer(loss.item(), self.global_step)
                    self.gradscaler.scale(loss).backward()
                    self.gradscaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)
                    self.gradscaler.step(self.optimizer)
                    self.gradscaler.update()
                    self.optimizer.zero_grad()

                else:
                    h_0 = torch.zeros(self.lstm_layer, self.bs, self.lstm_dim).to(self.device)
                    c_0 = torch.zeros(self.lstm_layer, self.bs, self.lstm_dim).to(self.device)
                    hidden = (h_0, c_0)

                    hht = torch.zeros(self.bs, 1, self.lstm_dim).to(self.device)
                    de_x = torch.ones(self.bs, 1).to(torch.long).to(self.device)
                    hidden = [state.to(self.device) for state in hidden]

                    for i in range(self.max_sent):
                        out = model(en_x, de_x, hidden, hht)
                        out = out.to(self.device)
                        pred = torch.max(out, dim=-1)[1]
                        de_x = torch.cat((de_x, pred[:, i].unsqueeze(1)), dim=1)

                    pred_tokens= de_x[:,1:]
                    pred_tokens = pred_tokens.tolist()

                    for i, token in enumerate(pred_tokens):
                        for j in range(len(token)):
                            if token[j] == 2:
                                token = token[:j]
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


        if self.type == "train":
            self.scheduler.step()
            return None
        else:
            print("total_bleu per epoch : {}".format(sum(total_bleu)/len(total_bleu)))


