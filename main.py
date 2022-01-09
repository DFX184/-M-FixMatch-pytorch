from trainer import *
from models.model import*
from data.dataset import*
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt
from torchvision import models
from config import*
############### Setting  Random Seed ############

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
############### building models ######################


model = build_resnet18(num_classes=num_classes)
model_ema = build_resnet18(num_classes = num_classes)
model.apply(init_weight)

DATASET = {
    'mnist':get_mnist_dataset,
    'cifar10':get_cifar_dataset,
    'stl10':get_stl10_dataset,
    'pets':get_pets_dataset
}

for param in model_ema.parameters():
    param.detach_()

############### building dataset #######################


# configures
# train_set, _, test_set, unlabeled_set = build_stl_dataset(10000,0.3)
# labeled_loader = DataLoader(
#     train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
# unlabeled_loader = DataLoader(
#     unlabeled_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
# #val_loader = DataLoader(val_set, batch_size=64, shuffle=True, pin_memory=True)
# test_loader = DataLoader(test_set, batch_size=batch_size,
#                          shuffle=True, pin_memory=True)
# train model

labeled_loader,unlabeled_loader,test_loader = DATASET[dataset](root_path,num_labeled)


optimizer = torch.optim.SGD(model.parameters(
), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0005)


scheduler_steplr = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=epochs, after_scheduler=scheduler_steplr)

trainer = MixMatch(model, optimizer, model_ema, linear_rampup(
    epochs), KL=False, scheduler=scheduler_warmup, T=7)


trainer.fit(epochs, labeled_loader, unlabeled_loader,
            test_loader, train_iters=eval_iters, fix_params=fix_params, fix=True, resume_path=resume_path)

train_loss, accuracy,f1_score= trainer.showlog()
trainer.savelog("test.csv")