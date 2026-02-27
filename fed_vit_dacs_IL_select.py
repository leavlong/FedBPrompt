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
from torchvision import transforms
from reid import models
from reid.models.memory import MemoryClassifier
from reid.server import FedDomainMemoTrainer
from reid.evaluators import Evaluator, extract_features , extract_features_tsne
from reid.utils.serialization import save_checkpoint , load_checkpoint
import torch.nn.functional as F
from reid.utils.tools import get_test_loader, get_data, get_train_loader , plotTSNE
from reid import datasets
from reid.utils.logger import setup_logger
import collections
from PIL import Image

# from test_prompt import visualize_average_prompt_attention, visualize_all_prompt_attentions_grid
from test_attention import visualize_attention_map


start_epoch = best_mAP = best_R1 = former_mAP = former_R1 = 0


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
    global start_epoch, best_mAP, best_R1, former_mAP, former_R1
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
    test_set = get_data(args)  # 获取指定目标域数据
    test_loader = get_test_loader(test_set, args.height, args.width,  # 创建目标域数据加载器
                                  args.batch_size, args.workers)

    train_sets = get_data(args, train_sets_name)  # 获取训练数据集
    num_classes1 = train_sets[0].num_train_pids
    num_classes2 = train_sets[1].num_train_pids
    num_classes3 = train_sets[2].num_train_pids
    num_classes = [num_classes1, num_classes2, num_classes3]
    print(' number classes = ', num_classes)
    num_users = len(train_sets)  # 计算训练集数量（即联邦学习的用户数）

    # 训练数据加载器
    train_set1_loader = get_train_loader(args, train_sets[0], args.height, args.width,
                                         args.batch_size, args.workers, args.num_instances, args.max_iter)
    test_set1_loader = get_test_loader(train_sets[0], args.height, args.width, args.batch_size, args.workers)

    train_set2_loader = get_train_loader(args, train_sets[1], args.height, args.width,
                                         args.batch_size, args.workers, args.num_instances, args.max_iter)
    test_set2_loader = get_test_loader(train_sets[1], args.height, args.width, args.batch_size, args.workers)

    train_set3_loader = get_train_loader(args, train_sets[2], args.height, args.width,
                                         args.batch_size, args.workers, args.num_instances, args.max_iter)
    test_set3_loader = get_test_loader(train_sets[2], args.height, args.width, args.batch_size, args.workers)

    train_set_loader = [train_set1_loader, train_set2_loader, train_set3_loader]
    test_set_loader = [test_set1_loader, test_set2_loader, test_set3_loader]

    # Create model
    model = create_model(args)
    # sub models on different servers
    sub_models = [create_model(args) for key in range(num_users)]
    aug_mods = [
        models.create('aug', num_features=3, width=args.width, height=args.height).cuda()
        for idx in range(num_users)
    ]

    evaluator = Evaluator(model)
    # Evaluator
    if args.tsne:
        MAX_SAMPLES_PER_DATASET = 2
    
        all_features_list = []
        all_labels_list = []
        model = model.eval()
        
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
        # evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
        
        # --- 1. 定义图像变换 (只需一次) ---
        for idx in range(len(test_set_loader)):
            features , _ = extract_features_tsne(model, test_set_loader[idx] , max_iter=MAX_SAMPLES_PER_DATASET)
            if features.shape[0] > 0:
                all_features_list.append(features)
                # 这里的标签是数据集的名字，用于区分颜色
                all_labels_list.extend([idx] * features.shape[0])
                print(f"Extracted {features.shape[0]} features from dataset: {train_sets_name[idx]}")
            else:
                print(f"Warning: No features extracted for {name}.")
                
        if not all_features_list:
            print("Error: No features were extracted. Aborting t-SNE.")
            return
        
        all_features = np.concatenate(all_features_list, axis=0)
        print(f"\n--- Finalizing ---")
        print(f"Total features for t-SNE: {len(all_features)}")
        print(f"Total domains for t-SNE: {len(all_labels_list)}")
        print(f"Dataset names for plot legend: {train_sets_name}")
        
        plotTSNE(all_features, all_labels_list, "/mnt/data/lh/lwl/mainwork/visualize/tsne/SSCU/sscu+prompt", 80)
        return 
    logger = setup_logger("transreid", args.logs_dir, if_train=True)
    if args.evaluate:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
        return

    if args.use_prompt:
        model.module.base.set_training_mode("vpt")
        for key in range(len(sub_models)):
            sub_models[key].module.base.set_training_mode("vpt")
    
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, weights_only=False)
        #直接不加载pos_embed
        if 'module.base.pos_embed' in checkpoint['state_dict']:
            del checkpoint['state_dict']['module.base.pos_embed']
        
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("权重加载成功！")
        logger.info("加入Prompt后的初始性能:")
        cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)

        logger.info('Mean AP: {:4.1%}'.format(cur_map))
        for k in (1, 5, 10):
            logger.info('  top-{:<4}{:12.1%}'.format(k, rank1[k - 1]))
        # evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)

        # model = model.module
        # checkpoint = osp.join(args.logs_dir, "official_model_best.pth.tar")
        # model.prompt_load_param(checkpoint)
        # test_photo = "photo/market1501/0001_c1s1_001051_00.jpg"
        # image = Image.open(test_photo).convert('RGB')
        # #visualize_prompt_attention(model,image)
        # # visualize_all_prompt_attentions_grid(model , image)
        # # visualize_average_prompt_attention(model, image)
        evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)

    print("==> Initialize source-domain class centroids and memorys ")
    source_centers_all = []  # 存储源域类别中心
    memories = []
    for dataset_i in range(len(train_sets)):  # 遍历每个源域数据集
        dataset_source = train_sets[dataset_i]
        # 获取数据集对应的测试数据
        sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                              args.batch_size, args.workers,
                                              testset=sorted(dataset_source.train))  # mixdata还没设计
        source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
        sour_fea_dict = collections.defaultdict(list)  # 默认字典 用于存储每个类别的特征

        for f, pid, _ in sorted(dataset_source.train):
            sour_fea_dict[pid].append(source_features[f].unsqueeze(0))  # 将同一身份的特征聚合在一起

        source_centers = [torch.cat(sour_fea_dict[pid], 0).mean(0) for pid in
                          sorted(sour_fea_dict.keys())]  # 计算每个身份标识的类中心
        source_centers = torch.stack(source_centers, 0)  ## pid,2048 将所有类中心堆叠起来)
        print(source_centers.shape)
        source_centers = F.normalize(source_centers, dim=1).cuda()  # 沿着第一个维度就进行归一化处理
        print('the num of source centers is:', source_centers.shape[0])
        # 自定义的内存分类器
        curMemo = MemoryClassifier(768, source_centers.shape[0],  # 类中心的数量
                                   temp=args.temp, momentum=args.momentum).cuda()
        curMemo.features = source_centers
        curMemo.labels = torch.arange(num_classes[dataset_i]).cuda()
        curMemo = nn.DataParallel(curMemo)  # 并行模式

        memories.append(curMemo)  # 将配置好的分类器添加到其中

        del source_centers, sour_cluster_loader, sour_fea_dict

    trainer = FedDomainMemoTrainer(args, train_sets, model, feature_dim=768, memory=memories)

    # start training

    prompt_global = {}
    for k, v in model.named_parameters():
        if "prompt_embeddings" in k:
            prompt_global[k] = v

    eval_start_time = start_time
    for epoch in range(start_epoch, args.epochs):  # number of epochs
        w_locals = []
        prompt_locals = []
        torch.cuda.empty_cache()
        F_news = [[], [], []]
        Labels = [[], [], []]
        for index in range(num_users):  # client index
            w, prompt_state = trainer.train_dacs_IL_VIT(
                sub_models[index], model, aug_mods[index],
                epoch, index, op_type='sgd', F_news=F_news, Labels=Labels, logger=logger
            )
            w_locals.append(w)
            if args.use_prompt:
                prompt_locals.append(prompt_state)
        # update global weight
        w_global = trainer.fed_avg(w_locals)
        if args.use_prompt:
            prompt_global = trainer.fed_avg(prompt_locals)
            model.load_state_dict(prompt_global, strict=False)
        else:
            model.load_state_dict(w_global, strict=True)
        # cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query,test_set.gallery, cmc_flag=True)

        if epoch % args.eval_step == 0 and epoch != 0:
            cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)

            # print('rank1:',rank1)

            if rank1[0] >= former_R1:
                print('severy Performance better with this epoch of style images,update the memory!')
                for index in range(num_users):
                    for lenth in range(len(Labels[index])):
                        memories[index].module.MomentumUpdate(F_news[index][lenth], Labels[index][lenth])
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
    parser.add_argument('-b', '--batch-size', type=int, default=16)  # 如有必要，减小
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    parser.add_argument('--use_prompt', type=bool, default=False, help="use prompt embedding")
    parser.add_argument('--patches', nargs='+', default=(16, 16))
    parser.add_argument('--num-prompts', type=int, default=1)
    parser.add_argument('--num-tokens', type=int, default=50)
    parser.add_argument('--location', type=str, default='prepend')
    parser.add_argument('--deep', type=bool, default=True)

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
    parser.add_argument('--temp', type=float, default=0.05, help="temperature")
    parser.add_argument('--rho', type=float, default=0.05, help="rho")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="momentum to update model")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--mu', type=float, default=0.3)

    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[20, 30], help='milestones for the learning rate decay')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate")

    parser.add_argument('--epochs', type=int, default=101)
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
    parser.add_argument('--load_checkpoint', type=str, default='')
    

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default="/mnt/data/lh/yzx/reiddatasets/") #/home/user/Documents/lwl/fed-clip-reid #/root/autodl-tmp/Fed-CLIP-ReID
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'sscu_logs/market1501_PAT'))
    parser.add_argument('--resume', type=str, default='./checkpoints/jx_vit_base_p16_224-80ecf9dd.pth')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--is_parallel', type=int, default=1)
    parser.add_argument('--tsne', action='store_true',
                        help="tsne only")
    main()
