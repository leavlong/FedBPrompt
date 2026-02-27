import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from types import SimpleNamespace

from models import BAPM
from utils import get_logger, AverageMeter, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BAPM')
    parser.add_argument('--config', default='configs/bapm_default.yaml',
                        help='path to config file')
    parser.add_argument('--checkpoint', required=True,
                        help='path to model checkpoint')
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


def build_transform(cfg):
    mean = cfg.DATA.MEAN
    std = cfg.DATA.STD
    size = cfg.DATA.IMG_SIZE
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def main():
    args = parse_args()
    cfg = load_config(args.config)

    logger = get_logger('bapm_test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Build model and load checkpoint
    model = BAPM(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict)
    logger.info(f'Loaded checkpoint from {args.checkpoint}')

    # Build test dataset
    test_transform = build_transform(cfg)

    # NOTE: Replace with your actual dataset class, e.g.:
    # from datasets import YourDataset
    # test_dataset = YourDataset(cfg.DATA.ROOT, split='test', transform=test_transform)
    raise NotImplementedError(
        'No dataset configured. Please replace the dataset placeholder in test.py.'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        for img_q, img_k, target in test_loader:
            img_q = img_q.to(device, non_blocking=True)
            img_k = img_k.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(img_q, img_k)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), img_q.size(0))
            top5.update(acc5.item(), img_q.size(0))

    logger.info(f'Test results: {top1} {top5}')


if __name__ == '__main__':
    main()
