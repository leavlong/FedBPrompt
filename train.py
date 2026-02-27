import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import yaml
from types import SimpleNamespace

from models import BAPM
from utils import get_logger, AverageMeter, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train BAPM')
    parser.add_argument('--config', default='configs/bapm_default.yaml',
                        help='path to config file')
    parser.add_argument('--distributed', action='store_true',
                        help='enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    return parser.parse_args()


def load_config(path):
    try:
        with open(path) as f:
            cfg_dict = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'Config file not found: {path}')
    except yaml.YAMLError as e:
        raise ValueError(f'Failed to parse config file {path}: {e}')

    def dict_to_ns(d):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_ns(v) if isinstance(v, dict) else v)
        return ns

    return dict_to_ns(cfg_dict)


def build_transform(cfg, is_train=True):
    mean = cfg.DATA.MEAN
    std = cfg.DATA.STD
    size = cfg.DATA.IMG_SIZE
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.makedirs(cfg.OUTPUT.DIR, exist_ok=True)
    logger = get_logger('bapm', log_file=os.path.join(cfg.OUTPUT.DIR, 'train.log'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Build model
    model = BAPM(cfg).to(device)
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Build dataset
    train_transform = build_transform(cfg, is_train=True)
    val_transform = build_transform(cfg, is_train=False)

    # NOTE: Replace with your actual dataset class, e.g.:
    # from datasets import YourDataset
    # train_dataset = YourDataset(cfg.DATA.ROOT, split='train', transform=train_transform)
    # val_dataset   = YourDataset(cfg.DATA.ROOT, split='val',   transform=val_transform)
    raise NotImplementedError(
        'No dataset configured. Please replace the dataset placeholder in train.py.'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.EPOCHS)

    best_acc = 0.0
    for epoch in range(cfg.TRAIN.EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg, logger)
        acc = validate(model, val_loader, criterion, device, logger)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            ckpt_path = os.path.join(cfg.OUTPUT.DIR, 'bapm_best.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'best_acc': best_acc}, ckpt_path)
            logger.info(f'Saved best checkpoint to {ckpt_path}')

        if (epoch + 1) % cfg.OUTPUT.SAVE_FREQ == 0:
            ckpt_path = os.path.join(cfg.OUTPUT.DIR, f'bapm_epoch{epoch + 1}.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, ckpt_path)

    logger.info(f'Training finished. Best accuracy: {best_acc:.2f}%')


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, cfg, logger):
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    for i, (img_q, img_k, target) in enumerate(loader):
        img_q = img_q.to(device, non_blocking=True)
        img_k = img_k.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(img_q, img_k)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        if cfg.TRAIN.CLIP_GRAD > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRAD)
        optimizer.step()

        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), img_q.size(0))
        top1.update(acc1.item(), img_q.size(0))

        if (i + 1) % cfg.OUTPUT.LOG_INTERVAL == 0:
            logger.info(f'Epoch [{epoch}][{i + 1}/{len(loader)}] {losses} {top1}')


def validate(model, loader, criterion, device, logger):
    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    with torch.no_grad():
        for img_q, img_k, target in loader:
            img_q = img_q.to(device, non_blocking=True)
            img_k = img_k.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(img_q, img_k)
            loss = criterion(output, target)

            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), img_q.size(0))
            top1.update(acc1.item(), img_q.size(0))

    logger.info(f'Validation: {losses} {top1}')
    return top1.avg


if __name__ == '__main__':
    main()
