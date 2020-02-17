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
    cfg = config.Config()

    # choose model
    model = RPPG(losses=cfg.losses, num_classes=cfg.num_classes,drop_rate=0.25).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # construct dataloader
    train_dataset = RPPG_Dataset(
        label_path = cfg.label_path, 
        mode = 'train', 
        clip_len=cfg.clip_len, 
        cur=0,
        k_fold=cfg.k_fold
    )
    val_dataset = RPPG_Dataset(
        label_path = cfg.label_path, 
        mode = 'val', 
        clip_len=cfg.clip_len, 
        cur=0,
        k_fold=cfg.k_fold
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
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
    train_meter = {"ce_loss":AverageMeter(),"mse_loss":AverageMeter(), "mse":AverageMeter()}
    val_meter = {"ce_loss":AverageMeter(),"mse_loss":AverageMeter(), "mse":AverageMeter()}

    # add logger
    if not os.path.exists("logs"):
        os.mkdir("logs")
    logdir = os.path.join("logs",datetime.now().strftime("%m-%d-%H-%M"))
    if not os.path.exists("logs"):
        os.mkdir(logdir)
    writer = SummaryWriter(logdir) 

    # set global min_mse
    min_mse = float("inf")

    # start training
    for cur_epoch in range(cfg.num_epoch):
        train(model, train_loader, optimizer, losses, train_meter, cur_epoch, writer, cfg)
        scheduler.step()
        if (cur_epoch+1)%cfg.val_freq == 0:
            val(model, val_loader, val_meter, min_mse, cur_epoch, writer, cfg)
        
def train(model, train_loader, optimizer, losses, train_meter, epoch, writer, cfg):
    model.train()
    # reset the meters
    for key in train_meter.keys():
        train_meter[key].reset()
    
    for inputs, classes, hrs, scale_rate in train_loader:
        inputs = inputs.cuda()
        classes = classes.cuda()
        hrs =  hrs.cuda()
        outputs = model(inputs) 

        # calculate loss and mse
        loss = 0
        if "ce" in losses:
            ce_loss = losses["ce"](outputs["ce"], classes)
            loss+=ce_loss
            train_meter["ce_loss"].update(ce_loss.cpu().detach().item(), inputs.size(0))
            _, predicted = torch.max(outputs["ce"].detach(), 1)
            train_meter["mse"].update((predicted+30-hrs).pow(2).sqrt().mean())
        

        if "mse" in losses:
            mse_loss = losses["mse"](outputs["mse"], (hrs-80)/40)
            loss+=mse_loss
            train_meter["mse_loss"].update(mse_loss.cpu().detach().item(), inputs.size(0))
            train_meter["mse"].update((outputs["mse"].detach()*40+80-hrs).pow(2).sqrt().mean())
            

        # perform backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    writer.add_scalar("train_ce_loss", train_meter["ce_loss"].avg, epoch)
    writer.add_scalar("train_mse_loss", train_meter["mse_loss"].avg, epoch)
    writer.add_scalar("train_mse", train_meter["mse"].avg, epoch)

    print("epoch:{}/{}, ce_loss:{}, mse_loss:{}, mse:{}".format(
        epoch,
        cfg.num_epoch,
        train_meter["ce_loss"].avg,
        train_meter["mse_loss"].avg,
        train_meter["mse"].avg
    ))

def val(model, val_loader, val_meter, min_mse, epoch, writer, cfg):
    model.eval()
    # reset the meters
    for key in val_meter.keys():
        val_meter[key].reset()
    
    for inputs, classes, hrs, scale_rate in val_loader:
        inputs = inputs.cuda()
        classes = classes.cuda()
        outputs = {"ce":0,"mse":0}
        hrs =  hrs.cuda()
        sample_len = inputs.size(3)
        step = int((sample_len-cfg.clip_len)/cfg.num_samples)
        for i in range(cfg.num_samples):
            tmp = model(inputs[:,:,:,i*step:i*step+cfg.clip_len])
            for key in tmp.keys():
                outputs[key] += tmp[key] 

        # calculate mse
        if "ce" in cfg.losses:
            _, predicted = torch.max(outputs["ce"].detach(), 1)
            val_meter["mse"].update((predicted+30-hrs).pow(2).sqrt().mean())
        

        if "mse" in cfg.losses:
            val_meter["mse"].update((outputs["mse"]/cfg.num_samples*40+80-hrs).pow(2).sqrt().mean())


    writer.add_scalar("test_mse", val_meter["mse"].avg, epoch)

    min_mse = min(min_mse, val_meter["mse"].avg)

    print("epoch:{}/{}, min_mse:{}, mse:{}".format(
        epoch, 
        cfg.num_epoch,
        min_mse, 
        val_meter["mse"].avg
    ))


if __name__ == "__main__":
    main()

