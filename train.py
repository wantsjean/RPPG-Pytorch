import config
import os
from datetime import datetime
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models import RPPG
from dataset import RPPG_Dataset
from utils import AverageMeter


def main():
    cfg = config.Cfg()

    # choose model
    model = RPPG(losses=cfg.losses, num_classes=cfg.num_classes,drop_rate=0.25).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # construct dataloader
    train_dataset = RPPG_Dataset(mode = 'train')
    val_dataset = RPPG_Dataset(mode = 'val')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )

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
        train(model, train_loader, optimizer, losses, train_meter, cur_epoch, writer, cfg)
        scheduler.step()
        if (cur_epoch+1)%cfg.val_freq == 0:
            val(model, val_loader, cur_epoch, val_meter, writer, cfg)
        
def train(model, train_loader, optimizer, losses, train_meter, epoch, writer, cfg):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs) 

        # add to meter
        train_meter.update()

        # calculate loss
        loss = losses(outputs,labels)
        
        # perform backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    


def val(model, val_loader, val_meter, epoch, writer, cfg):
    model.eval()
    for inputs, labels in val_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs) 

