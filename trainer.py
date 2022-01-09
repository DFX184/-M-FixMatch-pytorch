import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchmetrics.functional.classification.f_beta import f1
from tqdm import tqdm
import torch.nn.functional as F
import torchmetrics.functional as metrics
import pandas as pd
"""
This is MixMatch Trainer Class
"""
from utils.tool import *
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from sklearn.metrics import f1_score

class MixMatch:
    def __init__(self, network, optimizer, ema_network, ramp_function, *, KL=False, ema_decay=0.999, alpha=0.75, T=50, scheduler=None):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.model = network.to(self.device)
        self.ema_model = ema_network.to(self.device)
        self.ramp_function = ramp_function
        self.scheduler = scheduler
        self.T = T
        self.is_KL = KL
        self.alpha = alpha
        self.ema_iters = 0
        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.losses = []
        self.accuracy = []
        self.f1_score = []

    def mix_loss(self, logits_labeled, labeled_out, logits_unlabeled, unlabeled_out, iters):
        prob_unlabeled = torch.softmax(unlabeled_out, dim=1)
        L = -torch.mean(torch.sum(F.log_softmax(logits_labeled,
                        dim=1) * labeled_out, dim=1))
        U = F.mse_loss(prob_unlabeled, unlabeled_out)
        w = self.ramp_function(iters)
        if self.is_KL:
            U += (KL_div(prob_unlabeled, unlabeled_out))
        return L, U, w*self.T

    def sharpen(self, probability1, probability2):
        mean_prob = (torch.softmax(probability1, dim=1) +
                     torch.softmax(probability2, dim=1))/2.0
        mean_prob = mean_prob**(1/0.5)
        sharpen_prob = (mean_prob/mean_prob.sum(dim=1, keepdim=True)).detach()
        return sharpen_prob

    def mix_up(self, mix_data, mix_label):
        l = np.random.beta(0.75, 0.75)
        l = max(l, 1-l)
        random_index = torch.randperm(mix_data.size(0))
        mix_shuffle_data = mix_data[random_index]
        mix_shuffle_label = mix_label[random_index]
        mixup_data = l*mix_data + (1 - l) * mix_shuffle_data
        mixup_label = l*mix_label + (1 - l) * mix_shuffle_label
        return mixup_data, mixup_label

    def evaluate(self, loader):
        Bar = tqdm(loader, ascii=True)
        ema_hit = 0
        hit = 0
        total_hit = 0
        ema_f1,model_f1= 0.0,0.0

        with torch.no_grad():
            for data, label in Bar:
                data, label = data.to(self.device), label.to(self.device)
                ema_out = self.ema_model(data)
                out = self.model(data)
                preds_mode = out.argmax(dim=-1)
                preds_ema  = ema_out.argmax(dim=-1)
                
                
                ema_f1 += f1_score(label.detach().cpu().numpy(),preds_ema.detach().cpu().numpy(),average='macro')
                model_f1 += f1_score(label.detach().cpu().numpy(),preds_mode.detach().cpu().numpy(),average='macro')



                ema_hit += (ema_out.argmax(-1) == label).float().sum().item()
                hit += (out.argmax(-1) == label).float().sum().item()
                total_hit += data.size(0)
        return (
            ema_hit/total_hit,
            hit/total_hit,
            ema_f1/len(loader),
            model_f1/len(loader)
        )

    def save_checkpoint(self, epoch, filename):
        torch.save({
            'epoch': epoch,
            'ema_model': self.ema_model.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler != None else 0.0
        }, filename)

    def load_state_dict(self, PATH):
        check_point = torch.load(PATH)
        # print(check_point)
        self.ema_model.load_state_dict(check_point['ema_model'])
        self.model.load_state_dict(check_point['ema_model'])
        self.optimizer.load_state_dict(check_point['optimizer_state_dict'])

    def fit(self, epochs, labeled_loader, unlabeled_loader, test_loader, train_iters, *, resume_path=None, fix_params=None, fix=False):
        best_ema_acc = 0.0
        best_model_acc = 0.0
        if fix and fix_params is not None:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 0, train_iters * epochs)
        if resume_path != None:
            self.load_state_dict(resume_path)
        for epoch in range(epochs):
            if fix and fix_params != None:
                self.train_fixmath_epoch(
                    labeled_loader, unlabeled_loader, epoch, train_iters, fix_params)
            else:
                self.train_epoch(
                    labeled_loader, unlabeled_loader, epoch, train_iters)
            ema_acc, acc,ema_f1,ema_model = self.evaluate(test_loader)
            train_ema_acc, train_acc,_,_ = self.evaluate(labeled_loader)
            if ema_acc > best_ema_acc:
                best_ema_acc = ema_acc
            if train_acc > best_model_acc:
                best_model_acc = train_acc
            print(
                f"| EPOCH : {epoch + 1} | TEST-EMA-ACC : {round(ema_acc,3)} | TEST-ACC : {round(acc,3)} | ")
            print(
                f"| EPOCH : {epoch + 1} | TRAIN-EMA-ACC : {round(train_ema_acc,3)} | TRAIN-ACC : {round(train_acc,3)} | ")
            if self.scheduler != None:
                self.scheduler.step()
            self.accuracy.append([ema_acc, acc])
            self.f1_score.append([ema_f1,ema_model])

        print(
            f" | Best EMA-ACCURACY : {best_ema_acc} | Best ACC : {best_model_acc}  | ")

    def update_ema(self, global_iter):
        alpha = min(1 - 1/(global_iter+1), self.ema_decay)
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(
                model_param.data, alpha=1 - alpha)

    def update_model(self):
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            model_param.data = ema_param.data.detach()

    def update_bn(self):
        '''
        ref : https://github.com/iBelieveCJM/Tricks-of-Semi-supervisedDeepLeanring-Pytorch
        '''
        for m2, m1 in zip(self.ema_model.named_modules(), self.model.named_modules()):
            if ('bn' in m2[0]) and ('bn' in m1[0]):
                bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
                bn2['running_mean'].data.copy_(bn1['running_mean'].data)
                bn2['running_var'].data.copy_(bn1['running_var'].data)
                bn2['num_batches_tracked'].data.copy_(
                    bn1['num_batches_tracked'].data)

    def train_epoch(self, labeled_loader, unlabeled_loader, current_epoch, train_iters):
        labeled_loader_iter = iter(labeled_loader)
        unlabeled_loader_iter = iter(unlabeled_loader)
        L_log = AverageMeter()
        U_log = AverageMeter()
        W_log = AverageMeter()
        SUM = AverageMeter()
        Bar = tqdm(range(train_iters), ascii=True)

        for i in Bar:
            try:
                labeled_x, labeled_y = labeled_loader_iter.next()
            except:
                labeled_loader_iter = iter(labeled_loader)
                labeled_x, labeled_y = labeled_loader_iter.next()
            try:
                (unlabeled_x1, unlabeled_x2), _ = unlabeled_loader_iter.next()

            except:
                unlabeled_loader_iter = iter(unlabeled_loader)
                (unlabeled_x1, unlabeled_x2), _ = unlabeled_loader_iter.next()

            batch_size = labeled_x.size(0)
            #print(unlabeled_x1.size(0) == unlabeled_x2.size(0))
            labeled_x, labeled_y, unlabeled_x1, unlabeled_x2 = labeled_x.to(self.device), labeled_y.to(
                self.device), unlabeled_x1.to(self.device), unlabeled_x2.to(self.device)
            self.ema_iters += 1
            with torch.no_grad():
                unlabeled_probability1 = self.model(unlabeled_x1)
                unlabeled_probability2 = self.model(unlabeled_x2)
                sharpen_prob = self.sharpen(
                    unlabeled_probability1, unlabeled_probability2)
            labeled_one_hot = F.one_hot(labeled_y, num_classes=10)
            mix_data = torch.cat(
                [labeled_x, unlabeled_x1, unlabeled_x2], dim=0)
            mix_label = torch.cat(
                [labeled_one_hot, sharpen_prob, sharpen_prob], dim=0)

            mixup_data, mixup_label = self.mix_up(mix_data, mix_label)

            # labeled_out = self.model(mixup_data[:batch_size])
            # unlabeled_out = self.model(mixup_data[batch_size:])
            mixed_input = list(torch.split(mix_data, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [self.model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(self.model(input))

            # put interleaved samples back
            logits = interleave(logits, batch_size)
            labeled_out = logits[0]
            unlabeled_out = torch.cat(logits[1:], dim=0)

            L, U, W = self.mix_loss(
                labeled_out, mixup_label[:batch_size], unlabeled_out, mixup_label[batch_size:], current_epoch + (i/train_iters))

            loss = L + W * U

            L_log.update(L.item(), labeled_x.size(0))
            U_log.update(U.item(), labeled_x.size(0))
            SUM.update(loss.item(), labeled_x.size(0))
            W_log.update(W, labeled_x.size(0))

            Bar.set_postfix(
                LABEL=round(L_log.avg, 3),
                UNLABEL=round(U_log.avg, 3),
                LAMBDA=round(W_log.avg, 3),
                SUM_LOSS=round(SUM.avg, 3)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.losses.append(loss.item())
            self.optimizer.step()

            self.update_ema(self.ema_iters)
            self.losses.append(loss.item())
        self.update_bn()

    def train_fixmath_epoch(self, labeled_loader, unlabeled_loader, current_epoch, train_iters, Fix_params):
        labeled_loader_iter = iter(labeled_loader)
        unlabeled_loader_iter = iter(unlabeled_loader)
        Bar = tqdm(range(train_iters))
        L_loss = AverageMeter()
        U_loss = AverageMeter()
        SUM_LOSS = AverageMeter()
        for i in Bar:
            self.ema_iters += 1
            try:
                labeled_x, labeled_y = labeled_loader_iter.next()
            except:
                labeled_loader_iter = iter(labeled_loader)
                labeled_x, labeled_y = labeled_loader_iter.next()
            try:
                (unlabeled_weak, unlabeled_strong), _ = unlabeled_loader_iter.next()

            except:
                unlabeled_loader_iter = iter(unlabeled_loader)
                (unlabeled_weak, unlabeled_strong), _ = unlabeled_loader_iter.next()

            batch_size = labeled_x.size(0)
            all_data = fix_interleave(
                torch.cat((labeled_x, unlabeled_weak, unlabeled_strong)), 2*Fix_params['mu']+2).to(self.device)
            labeled_y = labeled_y.to(self.device)

            all_logits = self.model(all_data)

            all_logits = de_fix_interleave(all_logits, 2*Fix_params['mu'] + 2)

            labeled_logits = all_logits[:batch_size]
            unlabeled_weak_logits, unlabeled_strong_logits = all_logits[batch_size:].chunk(
                2)  # split weak and strong logits

            L = F.cross_entropy(labeled_logits, labeled_y, reduction='mean')
            guess_label = torch.softmax(
                unlabeled_weak_logits.detach()/Fix_params['T'], dim=-1)
            max_prob, target_u = torch.max(guess_label, dim=-1)
            mask = max_prob.ge(Fix_params['threshold']).float()
            U = (F.cross_entropy(unlabeled_strong_logits,
                 target_u, reduction='none')*mask).mean()

            loss = L + Fix_params['lambda'] * U

            L_loss.update(L.item(), labeled_x.size(0))
            U_loss.update(U.item(), labeled_x.size(0))
            SUM_LOSS.update(loss.item(), labeled_x.size(0))
            Bar.set_postfix(EPOCH=current_epoch + 1, LOSS=round(SUM_LOSS.avg, 3),
                            UNLABEL=round(U_loss.avg, 3), LABEL=round(L_loss.avg, 3))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_ema(self.ema_iters)
            self.scheduler.step()
            # self.update_model()
        self.update_bn()

    def showlog(self):
        return np.array(self.losses), np.array(self.accuracy),np.array(self.f1_score)
    def savelog(self,filename):
        accuracy = np.array(self.accuracy)
        f1_score = np.array(self.f1_score)
        log_df = pd.DataFrame(
            {
                "ema_accuracy" : accuracy[:,0],
                "model_accuracy" : accuracy[:,1],
                "ema_f1"  : f1_score[:,0],
                "model_f1" : f1_score[:,1]
            }
        )
        log_df.to_csv(filename,index=False)