import argparse
import os
import os.path as osp
import random
import numpy as np
import time
from datetime import timedelta
import torch
from sympy import false
from torch import nn
from torch.backends import cudnn
from reid import models
from reid.server import FedDomainMemoTrainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.serialization import save_checkpoint
import torch.nn.functional as F
from reid.utils.tools import get_test_loader, get_data , plotTSNE
from reid import datasets
from reid.utils.logger import setup_logger
import collections
from PIL import Image
from visualize.prompt_tsne import run_tsne_and_plot_prompts
from test_attention import visualize_attention_map

start_epoch = best_mAP = 0


def create_model(args, num_classes=0, index_num=None):
    model = models.make_model(
        args=args, num_class=num_classes,
        index_num=index_num
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

    print("==========\nArgs:{}\n==========".format(args))
    # Create datasets
    print("==> Building Datasets")
    test_set = get_data(args)
    test_loader = get_test_loader(test_set, args.height, args.width,
                                  args.batch_size, args.workers)
    train_sets = get_data(args, train_sets_name)
    num_users = len(train_sets)

    # Create model
    model = create_model(args)
    # sub models on different servers
    sub_models = [create_model(args) for key in range(num_users)]
    aug_mods = [
        models.create('aug', num_features=3, width=args.width, height=args.height).cuda()
        for idx in range(num_users)
    ]

    if args.only_update_prompt:
        model.module.base.set_training_mode("vpt")
        for key in range(len(sub_models)):
            sub_models[key].module.base.set_training_mode("vpt")

    # Evaluator
    evaluator = Evaluator(model)
    trainer = FedDomainMemoTrainer(args, train_sets, model, feature_dim=768)

    if args.prompt_tsne:
        checkpoint_path = osp.join(args.logs_dir, "model_best.pth.tar")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)  # 使用 map_location 避免GPU内存问题
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict , strict=True)

            # 只获取最终训练完成的浅层 prompt
            # 它的形状是 [1, num_tokens, embed_dim]
        final_shallow_prompts = model.module.prompt_embeddings.clone().detach().cpu()
            
            # 我们需要的是一个包含 num_tokens 个向量的列表
            # final_shallow_prompts[0] 的形状是 [num_tokens, embed_dim]
            # 我们将其转换为一个Python列表
        prompt_vectors_list = [p for p in final_shallow_prompts[0]]
            
        
        all_prompts = []
        all_client_ids = []
        for index in range(num_users):  # client index
            prompt = trainer.train_dacs(
                sub_models[index], model, aug_mods[index],
                1, index, op_type='sgd' , tsne = True
            )
            all_prompts.extend(prompt)
            all_client_ids.extend([index] * len(prompt))
        run_tsne_and_plot_prompts(
            all_prompts,
            all_client_ids,
            title="",
            save_path="/mnt/data/lh/lwl/mainwork/visualize/tsne/DACS/prompt_tsne.png"
        )
        
        return

    if args.evaluate:
        from collections import OrderedDict

        # 1. 加载权重文件
        checkpoint_path = osp.join(args.logs_dir, "official_model_best.pth.tar")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)  # 使用 map_location 避免GPU内存问题
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
        model.eval()
        test_photo = "photo/msmt17/0000_c14_0030.jpg"
        pn = os.path.basename(test_photo)
        image = Image.open(test_photo).convert('RGB')
        # visualize_attention_map(model, image, "image_to_image_avg", pn, args.test_dataset)
        evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)

        return

    if args.resume_checkpoint:
        checkpoint = torch.load(osp.join(args.logs_dir, "model_best.pth.tar"), weights_only=False)
        print("正在加载修正后的权重...")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("权重加载成功！")
    
    if args.load_checkpoint:  
        checkpoint = torch.load(osp.join(args.logs_dir, "official_model_best.pth.tar"), weights_only=False)
        # 直接不加载pos_embed
        if 'module.base.pos_embed' in checkpoint['state_dict']:
            del checkpoint['state_dict']['module.base.pos_embed']

        print("正在加载修正后的权重...")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("权重加载成功！")
        
        cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
        logger.info("加入Prompt后的初始性能:")
        logger.info('Mean AP: {:4.1%}'.format(cur_map))
        for k in (1, 5, 10):
            logger.info('  top-{:<4}{:12.1%}'.format(k, rank1[k - 1]))
        # 直接不加载pos_embed

        # evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)


    # start training
    logger = setup_logger("transreid", args.logs_dir, if_train=True)
    eval_start_time = start_time
    pretrain_end = False
    for epoch in range(start_epoch, args.epochs):  # number of epochs
        w_locals = []
        prompt_locals = []
        sub_prompts =[]
        all_client_ids = []
        torch.cuda.empty_cache()
        for index in range(num_users):  # client index
            w , prompt_local , sub_prompt = trainer.train_dacs(
                sub_models[index], model, aug_mods[index],
                epoch, index, op_type='sgd'
            )
            w_locals.append(w)
            prompt_locals.append(prompt_local)
            prompt_vectors_list = [p for p in sub_prompt[0]]

            sub_prompts.extend(prompt_vectors_list)
            all_client_ids.extend([index] * len(prompt_vectors_list))
        # update global weight
        w_global = trainer.fed_avg(w_locals)
        if args.only_update_prompt:
            prompt_global = trainer.fed_avg(prompt_locals)
            model.load_state_dict(prompt_global, strict=False)
        else:
            model.load_state_dict(w_global)

        # if args.only_update_prompt:
        #     prompt_global = trainer.fed_avg(prompt_locals)
        #     model.load_state_dict(prompt_global, strict=False)
        # else:
        #     model.load_state_dict(w_global)
        # cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query,test_set.gallery, cmc_flag=True)

        if epoch % args.eval_step == 0 and epoch != 0:
            cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
            print(f"原始 sub_prompts 类型: {type(sub_prompts)}, 长度: {len(sub_prompts)}")
            
            # --- 核心修复代码 ---
            # 1. 使用 torch.stack 将张量列表堆叠成一个大的张量
            #    dim=0 表示在新的第0维进行堆叠
            features_tensor = torch.stack(sub_prompts, dim=0)
            
            # 2. 将这个大的张量转换为 NumPy 数组
            #    .cpu() 是一个好习惯，确保数据在CPU上，以防万一
            features_numpy = features_tensor.cpu().numpy()
            
            print(f"转换后的 features 类型: {type(features_numpy)}, 形状: {features_numpy.shape}")
            print("绘制Prompt本地分布")
            plotTSNE(
                features_numpy,
                all_client_ids,
                save_path=f"{args.logs_dir}/prompt_tsne/epoch.jpg",
                epoch = epoch
            )
            print("绘制完毕！")
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
    parser.add_argument('-td', '--test-dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-ed', '--exclude-dataset', type=str, default='')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    parser.add_argument('--only_update_prompt', type=bool, default=False)
    parser.add_argument('--patches', nargs='+', default=(16, 16))
    parser.add_argument('--num-prompts', type=int, default=1)
    parser.add_argument('--num-tokens', type=int, default=50)
    parser.add_argument('--location', type=str, default='prepend')
    parser.add_argument('--deep', type=bool, default=False)

    parser.add_argument('--pretrain-choice', type=str, default='imagenet')
    parser.add_argument('--stride-size', type=int, default=12)
    parser.add_argument('--shift-num', type=int, default=5)
    parser.add_argument('--shuffle-group', type=int, default=2)
    parser.add_argument('--devide-length', type=int, default=4)

    # model
    parser.add_argument('-a', '--arch', type=str, default='transformer')
    parser.add_argument('--re-arrange', type=bool, default=True)
    parser.add_argument('--last-stride', type=bool, default=True)
    parser.add_argument('--transformer-type', type=str, default='vit_base_patch16_224_TransReID_Prompt_PAT')
    parser.add_argument('--neck-feat', type=str, default='before')
    parser.add_argument('--neck', type=str, default='bnneck')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lam', type=float, default=5)
    # optimizer
    parser.add_argument('--rho', type=float, default=0.05, help="rho")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="momentum to update model")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)

    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[20, 30], help='milestones for the learning rate decay')
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
    parser.add_argument("--resume_checkpoint", type=bool, default=False)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default="/mnt/data/lh/yzx/reiddatasets/") #/home/user/Documents/lwl/fed-clip-reid #root/autodl-tmp/Fed-CLIP-ReID
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'dacs_logs/market1501_PATtune'))
    parser.add_argument('--resume', type=str, default='./checkpoints/jx_vit_base_p16_224-80ecf9dd.pth')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--prompt_tsne', action='store_true',
                        help="evaluation only")
    parser.add_argument('--is_parallel', type=int, default=1)
    main()

