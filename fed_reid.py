import argparse
import os
import os.path as osp
import random
import numpy as np
import time
from datetime import timedelta

import torch
from torch import nn
from torch.backends import cudnn

from reid import models
from reid.server import FedDomainMemoTrainer
from reid.evaluators import Evaluator
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.tools import get_test_loader, get_data
from reid import datasets
import torch.nn.functional as F
from reid.utils.tools import get_test_loader, get_data, get_train_loader
from reid import datasets
from reid.utils.logger import setup_logger
import collections
from PIL import Image
from visualize.tsne import tsne_main

start_epoch = best_mAP = 0


def create_model(args, num_cls=0):
    # we only use triplet loss, remember to turn off 'norm'
    # model = models.create(
    #     args.arch, num_features=args.features, norm=False,
    #     dropout=args.dropout, num_classes=num_cls
    # )
    model = models.make_model(
        args=args, num_class=num_cls, 
        index_num=None
    )
    # use CUDA
    model = model.cuda()
    model = nn.DataParallel(model) if args.is_parallel else model
    return model


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True
    all_datasets = datasets.names()
    test_set_name = args.test_dataset
    all_datasets.remove(test_set_name)
    
    if args.exclude_dataset is not '':
        exclude_set_name = args.exclude_dataset.split(',')
        [all_datasets.remove(name) for name in exclude_set_name]
    train_sets_name = sorted(all_datasets)

    # Create datasets
    print("==> Building Datasets")
    test_set = get_data(args)
    test_loader = get_test_loader(test_set, args.height, args.width, 
                                  args.batch_size, args.workers)
    train_sets = get_data(args, train_sets_name)
    num_users = len(train_sets)

    # Create model
    model = create_model(args)  
    # sub local models
    sub_models = [create_model(args) for key in range(num_users)]
    
    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        from collections import OrderedDict

        # 1. 加载权重文件
        checkpoint_path = osp.join(args.logs_dir, "model_best.pth.tar")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # 使用 map_location 避免GPU内存问题
        original_state_dict = checkpoint['state_dict']

        # 2. 创建一个新的、没有 'module.' 前缀的 state_dict
        new_state_dict = OrderedDict()
        for k, v in original_state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # 移除 'module.' 前缀 (7个字符)
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v  # 如果有些键没有module前缀，也一并保留

        # 3. 使用修正后的 state_dict 加载权重，并坚持使用 strict=True
        print("正在加载修正后的权重...")
        model.load_state_dict(new_state_dict, strict=True)
        print("权重加载成功！")

        test_photo = "photo/market1501/0018_c2s1_001126_00.jpg"
        base_name = os.path.basename(test_photo)
        image = Image.open(test_photo).convert('RGB')
        model.eval()
        evaluate_test_loader = get_test_loader([train_sets[2], test_set], args.height, args.width, args.batch_size,
                                               args.workers)
        # visualize_prompt_attention(model,image)
        target = [4, 12, 13, 24, 41]
        # visualize_attention_map(model, image, "image_to_image_avg" , base_name , args.test_dataset)
        tsne_main(model.base, evaluate_test_loader, base_name, args.test_dataset)
        # GradCAM(model , test_photo , base_name , args.test_dataset)
        evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
        return

    if args.only_update_prompt:
        model.module.base.set_training_mode("vpt")
        for key in range(len(sub_models)):
            sub_models[key].module.base.set_training_mode("vpt")
            
    
    if args.resume_checkpoint:
        checkpoint = torch.load(osp.join(args.logs_dir, "model_best.pth.tar"), weights_only=False)
        print("正在加载修正后的权重...")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("权重加载成功！")
        
    #Tune
    logger = setup_logger("transreid", args.logs_dir, if_train=True)
    if args.load_checkpoint:
        checkpoint = torch.load(osp.join(args.logs_dir, "official_model_best.pth.tar"), weights_only=False)
        # 直接不加载pos_embed
        if 'module.base.pos_embed' in checkpoint['state_dict']:
            del checkpoint['state_dict']['module.base.pos_embed']
        print("正在加载修正后的权重...")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("权重加载成功！")
        
        logger.info("加入Prompt后的初始性能:")
        cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
        logger.info('Mean AP: {:4.1%}'.format(cur_map))
        for k in (1, 5, 10):
            logger.info('  top-{:<4}{:12.1%}'.format(k, rank1[k - 1]))
        # evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)

    # trainer = FedDomainMemoTrainer(args, train_sets, model)
    trainer = FedDomainMemoTrainer(args, train_sets, model, feature_dim=768)
    
    # if args.resume:
    #     checkpoint = load_checkpoint(args.resume)
    #     start_epoch = checkpoint['epoch'] - 1
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
    #     pass
        
    # start training

    
    prompt_global = {}
    for k, v in model.named_parameters():
        if "prompt_embeddings" in k:
            prompt_global[k] = v

    eval_start_time = start_time
    for epoch in range(args.epochs):  # number of epochs
        w_locals = []
        prompt_locals = []
        torch.cuda.empty_cache()
        for index in range(num_users):  # client index
            w , prompt_local = trainer.train_reid(sub_models[index], model, epoch, index) #Fed_reid
            # w , prompt_local = trainer.train_crossstyle(sub_models[index], epoch, index)
            w_locals.append(w)
            prompt_locals.append(prompt_local)
        # update global weight
        w_global = trainer.fed_avg(w_locals)
        if args.only_update_prompt:
            prompt_global = trainer.fed_avg(prompt_locals)
            model.load_state_dict(prompt_global , strict = False)
        else:
            model.load_state_dict(w_global)  # the averaged model



        if epoch % args.eval_step == 0 and epoch != 0:
            cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)

            # print('rank1:',rank1)
            # save
            if cur_map > best_mAP:
                print('best model saved!')
                save_checkpoint({
                    'state_dict': w_global,
                    'epoch': epoch + 1, 'best_mAP': best_mAP,
                }, 1, fpath=osp.join(args.logs_dir, f'checkpoint_{epoch}.pth.tar'))
            logger.info('Mean AP: {:4.1%}'.format(cur_map))
            for k in (1, 5, 10):
                logger.info('  top-{:<4}{:12.1%}'.format(k, rank1[k - 1]))
            end_time = time.monotonic()
            logger.info(f'finishing this checkpoint running time: {timedelta(seconds=end_time - eval_start_time)}')
            eval_start_time = end_time
    end_time = time.monotonic()
    logger.info(f'Total running time: {timedelta(seconds=end_time - start_time)}')
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Domain-level Fed Learning")
    # data
    parser.add_argument('-td', '--test-dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-ed', '--exclude-dataset', type=str, default='')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    # parser.add_argument('-a', '--arch', type=str, default='resnet50',
    #                     choices=models.names())
    # parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--only_update_prompt', type=bool, default=False)
    parser.add_argument('-a', '--arch', type=str, default='transformer',
                        choices=models.names())
    parser.add_argument('--pretrain-choice', type=str, default='imagenet')
    parser.add_argument('--neck-feat', type=str, default='before')
    parser.add_argument('--neck', type=str, default='bnneck')
    parser.add_argument('--transformer-type', type=str, default='vit_base_patch16_224_TransReID')
    parser.add_argument('--stride-size', type=int, default=12)
    parser.add_argument('--shift-num', type=int, default=5)
    parser.add_argument('--shuffle-group', type=int, default=2)
    parser.add_argument('--devide-length', type=int, default=4)
    parser.add_argument('--re-arrange', type=bool, default=True)
    parser.add_argument('--last-stride', type=bool, default=True)
    parser.add_argument('--resume', type=str, default='./checkpoints/jx_vit_base_p16_224-80ecf9dd.pth')
    
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer

    parser.add_argument('--temp', type=float, default=3, help="temperature")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--milestones', nargs='+', type=int, 
                        default=[20, 30], help='milestones for the learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate")

    parser.add_argument('--epochs', type=int, default=81)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--resume_checkpoint', type=bool, default=False)
    
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default="/mnt/data/lh/yzx/reiddatasets/")
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'fed_crossstyle/cuhk03_official'))
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--is_parallel', type=int, default=1)
    main()
