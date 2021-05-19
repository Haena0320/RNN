import sys, os
sys.path.append(os.getcwd())
import torch
from torch.cuda.amp import autocast
import sacrebleu
from sacremoses import MosesDetokenizer
from tqdm import tqdm
import tqdm

def get_trainer(config, args, device, data_loader, writer, type):
    return Trainer(config, args, device, data_loader, writer, type)

def get_optimizer(model, args_optim):
    if args_optim =="sgd":
        return torch.optim.SGD(model.parameters(), lr=1)
    else:
        print("optim error ")

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
        self.ckpnt_step = 10000
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

    def train_epoch(self, model, epoch, save_path=None, sp=None):
        if self.type =="train":
            model.train()

        else:
            model.eval()
            total_bleu = list()

        model.to(self.device)

        for iter in tqdm.tqdm(self.data_loader):
            with autocast():
                en_x = iter['encoder'][:,1:].to(self.device) # 128, 49
                dn_x = iter["decoder"].to(self.device) # 128, 50

                if self.type =="train":
                    loss = model(en_x, dn_x, predict=False)
                    self.global_step += 1
                    if self.global_step % self.ckpnt_step ==0:
                        torch.save({"epoch":epoch,
                                    "model_sate_dict":model.state_dict(),
                                    "optimizer_state_dict":self.optimizer.state_dict()},
                                   save_path+'ckpnt_{}'.format(epoch))

                    self.log_writer(loss.data, self.global_step)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                else:
                    prediction = model(en_x, dn_x, predict=True)
                    bs, _ =prediction.size()

                    pred = list()
                    truth = list()
                    for i in range(bs):
                        p = prediction[i,:]
                        p = p.tolist()
                        for j in range(len(p)):
                            if p[j] ==2: # end token
                                p = p[:j]
                                break
                        de_token = sp.DecodeIds(p)
                        de_truth = sp.DecodeIds(dn_x[i,:].tolist())
                        pred.append(de_token)
                        truth.append(de_truth)
                    pred = [self.md.detokenize(i.strip().split()) for i in pred]
                    truth = [self.md.detokenize(i.strip().split()) for i in truth]

                    assert len(pred) == len(truth)
                    bleu = [sacrebleu.corpus_bleu(pred[i], truth[i]).score for i in range(len(pred))]
                    total_bleu.append(sum(bleu)/len(bleu))

        self.scheduler.step()
        print("total_bleu per epoch : {}".format(sum(total_bleu)/len(total_bleu)))











