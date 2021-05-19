import sys, os
sys.path.append(os.getcwd())
import torch
from torch.cuda.amp import autocast
import sacrebleu
from scremoses import MosesDetokenizer

def get_trainer(config, args, device, data_loader, writer, type):
    return Trainer(config, args, device, data_loader, writer, type)

def get_optimizer(model, args_optim):
    if args_optim =="adam":
        return torch.optim.SGD(model.parameters(), lr=1)

def get_lr_scheduler(optimizer, config):
    warmup = config.train.warmup
    return Wamupscheduler(optimizer, warmup)

class Warmupscheduler:
    def __init__(self,optimizer, warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self._rate = 1

    def step(self):
        self._step += 1
        if self._step > self.warmup:
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p["lr"] = rate

        self.optimizer.step()

    def rate(self):
        self._rate /= 2


class Trainer:
    def __init__(self, config, args, device, writer, data_loader, type):
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

    def log_writer(self, log):
        if self.type =="train":
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss", log, self.global_step)
            self.writer.add_scalar("train/lr", lr, self.global_step)

    def train_epoch(self, model, epoch, save_path=None, sp=None):
        if self.type =="train":
            model.train()
        else:
            model.eval()

        model.to(self.device)

        for data in tqdm(self.data_loader):
            with autocast():
                en_x = data['encoder'][:,1:].to(self.device) # 128, 49
                dn_x = data["decoder"].to(self.device) # 128, 50
                if self.type =="train":
                    loss = model(en_x, dn_x, predict=False)
                    if self.global_step % self.ckpnt_step ==0:
                        torch.save({"epoch":epoch,
                                    "model_sate_dict":model.state_dict(),
                                    "optimizer_state_dict":self.optimizer.state_dict()},
                                   save_path+'ckpnt_{}'.format(epoch))

                    self.log_writer(loss.data)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), self.config.train.clip)
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                else:

                    prediction = model(en_x, dn_x, predict=True)
                    bs, _ =prediction.size()

                    pred = list()
                    truth = list()
                    for i in range(bs):
                        p = prediction[i,:]
                        p = p.list()
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
                    total_bleu = [sacrebleu.corpus_bleu(pred[i], truth[i]).score for i in range(len(pred))]
                    print("total_bleu per iter : {}".format(sum(total_bleu)/len(total_bleu)))











