import argparse
import os
import random
import math
import time
import sys

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms,datasets
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger

from networks.MTC import SupConResNet
from utils.tools import TwoCropTransform, AverageMeter, warmup_learning_rate, set_optimizer, \
    adjust_learning_rate, save_model
from utils.losses import SupConLoss


# Set random seeds for reproducibility.
def init_seed(fixed_seed):
    random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    torch.cuda.manual_seed(fixed_seed)

# Arguments for training.
def parse_option():
    parser = argparse.ArgumentParser('Argument for training.')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--dataset_path', type=str, default='./datasets/train', help='path for datasets')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--encoder', type=str, default='resnet18', help='encoder')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='70,80,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for lr')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')

    # schedulers / tricks
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true', help='warm up for large batch training')

    # bookkeeping
    parser.add_argument('--trial', type=str, default=0, help='id for recording multiple runs')

    opt = parser.parse_args()

    # save dirs
    opt.model_path = "./save/models"
    os.makedirs(opt.model_path, exist_ok=True)
    opt.tb_path = "./save/tensorboard"
    os.makedirs(opt.tb_path, exist_ok=True)

    opt.model_name = (f"encoder_{opt.encoder}_lr_{opt.learning_rate}_decay_{opt.weight_decay}"
                      f"_bsz_{opt.batch_size}_temp_{opt.temp}_trial_{opt.trial}")

    # parse milestones
    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]

    if opt.cosine:
        opt.model_name = f"{opt.model_name}_cosine"

    # warmup logic
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = f"{opt.model_name}_warm"
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            # cosine warmup target
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.model_folder, exist_ok=True)
    opt.tb_folder =os.path.join(opt.tb_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)

    return opt


def set_loader(opt):

    mean, std = ((0.9920479655265808, 0.9398094415664673, 0.9398094415664673),
                 (0.08487337082624435, 0.21652734279632568, 0.21652734279632568))

    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 亮度，对比度，饱和度，色调
        ], p=0.8),
        transforms.ToTensor(),
        # light Gaussian noise to encourage robustness
        transforms.Lambda(lambda x: torch.clamp(x + torch.randn(x.shape, device=x.device) * 0.05, 0, 1)),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.ImageFolder(root=opt.dataset_path, transform=TwoCropTransform(train_transform))

    train_sampler = None
    train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                   num_workers=opt.num_workers,
                                   # prefetch data into pinned (page-locked) memory for faster GPU transfer
                                   pin_memory=True,
                                   sampler=train_sampler)
    return train_loader


def set_model(opt):
    model = SupConResNet(encoder=opt.encoder)
    criterion = SupConLoss(temperature=opt.temp)

    if opt.syncBN:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        # enable cudnn autotuner for better perf
        cudnn.benchmark = True

    return model, criterion


# Train for one epoch.
def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()

    # metric
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # concatenate two augmented views along batch dim
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # per-iteration warmup update
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        # shape -> [bsz, 2, feat_dim]
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():

    # 1、set fixed random seed
    fixed_seed = 2025
    init_seed(fixed_seed)

    # 2. parse command-line arguments
    opt = parse_option()

    # 3. build data loader
    train_loader = set_loader(opt)

    # 4. build model and loss function
    model, criterion = set_model(opt)
    print(next(model.parameters()).device)

    # 5. build optimizer
    optimizer = set_optimizer(opt, model)

    # 6、tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # 7. training loop
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print(f'epoch {epoch}, total time {(time2 - time1):.2f}')

        # log metrics to tensorboard
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # save checkpoint
        if epoch % opt.save_freq ==0:
            save_file = os.path.join(opt.model_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    # save final model
    save_file = os.path.join(opt.model_folder, f'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()