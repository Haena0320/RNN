import sys, os
sys.path.append(os.getcwd())
import torch
from torch.cuda import amp
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

    def train_epoch(self, model, epoch, save_path=None, sp=None):
        if self.type =="train":
            model.train()

        else:
            model.eval()
            total_bleu = list()

        model.to(self.device)

        for iter in tqdm.tqdm(self.data_loader):
            with amp.autocast():
                en_x = iter['encoder'].to(self.device) # 128, 51 [0,4,3,2,1]
                dn_x = iter["decoder"].to(self.device) # 128, 51 [0,1,2,3,4]

                if self.type =="train":
                    loss, pred = model(en_x, dn_x, predict=False)
                    self.global_step += 1
                    if self.global_step % self.ckpnt_step ==0:
                        torch.save({"epoch":epoch,
                                    "model_sate_dict":model.state_dict(),
                                    "optimizer_state_dict":self.optimizer.state_dict(), 
                                    "lr_step":self.scheduler._step},
                                   save_path+'ckpnt_{}'.format(epoch))

                    self.log_writer(loss.item(), self.global_step)
                    self.writer.add_scalar("train/accuracy",pred.item(), self.global_step)
                    self.gradscaler.scale(loss).backward()
                    self.gradscaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)
                    self.gradscaler.step(self.optimizer)
                    self.gradscaler.update()
                    self.optimizer.zero_grad()

                else:
                    y = model(encoder_input, decoder_input, predict=True)  # (bs, seq)
                    tokens = y.tolist()

                    for i, token in enumerate(tokens):
                        for j in range(len(token)):
                            if token[j] == 2:
                                token = token[:j]
                                break
                        decode_tokens = [sp[t] for t in token] # sp : id2word
                        print(decode_tokens)

                        decode_truth = [sp[t] for t in decoder_input[i, :].tolist()]
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











