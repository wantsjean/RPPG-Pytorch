import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
import config
from models import RPPG
from utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

def main():
    cfg = config.Cfg()

    # choose model
    model = RPPG(losses=cfg.losses, num_classes=cfg.num_classes,drop_rate=0.25).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # construct loss
    losses = {}
    if 'ce' in cfg.losses:
        losses['ce'] = nn.CrossEntropyLoss()
    if 'mse' in cfg.losses:
        losses['mse'] = nn.MSELoss()  

    # construct optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        cfg.lr,
        weight_decay=1e-4
    )
    scheduler = MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.1)

    # construct meter
    train_meter = AverageMeter()
    val_meter = AverageMeter()

    # add logger
    if not os.path.exists("logs"):
        os.mkdir("logs")
    # dt_string = datetime.now().strftime("%m-%d-%H-%M")
    writer = SummaryWriter("logs")

    for cur_epoch in range(cfg.num_epoch):
        train(model, optimizer, losses, train_meter, cur_epoch, writer, cfg)
        scheduler.step()
        if (cur_epoch+1)%cfg.val_freq == 0:
            val(model, cur_epoch, val_meter, writer, cfg)
        
def train(model, train_loader, optimizer, losses, train_meter, epoch, writer, cfg):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs) 

    


def val(model, val_loader, val_meter, epoch, writer, cfg):
    model.eval()
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs) 

