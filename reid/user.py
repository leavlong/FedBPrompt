from torch import nn
from torch.utils.data import DataLoader
from .utils.data import IterLoader, Preprocessor
import torch
from torch.cuda import amp
from .models.resnet import UBS
from .utils.data.sampler import RandomMultipleGallerySampler
from .utils.tools import get_entropy, get_auth_loss, ScaffoldOptimizer, cn_op_2ins_space_chan, freeze_model, \
    inception_score
from .loss.triplet import TripletLoss
from .loss.triplet_loss import TripletLoss as Tri_clip
from .loss.softmax_loss import CrossEntropyLabelSmooth
from .loss.make_loss import make_loss
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import os
import copy
from torchvision.utils import save_image
# from pytorch_msssim import ssim
import numpy as np
import time
from .loss.supcontrast import SupConLoss
from reid.lr_scheduler import WarmupMultiStepLR

#scaffold

class UnifiedClientModel(nn.Module):
    """一个简单的包装器，将主干网络和分类头合并成一个单一的 nn.Module"""
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        # 这个前向传播逻辑与您在训练循环中的逻辑完全一致
        features = self.backbone(x)[0]
        scores = self.classifier(features)
        # 您的损失函数同时需要 features 和 scores
        return features, scores

# trainers in user side
class DomainLocalUpdate(object):
    def __init__(self, args, dataset=None, trans=None, memory=None, client_id=None, logger=None):
        self.args = args
        self.trans = trans
        self.memory = memory
        self.client_id = client_id
        self.logger = logger
        # only for non-qaconv algos
        if dataset is not None:
            if not isinstance(dataset, list):
                self.local_train = IterLoader(DataLoader(
                    Preprocessor(dataset.train, transform=trans, root=None),
                    batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                    sampler=RandomMultipleGallerySampler(
                        dataset.train, args.num_instances),
                    pin_memory=False, num_workers=self.args.num_workers
                ), length=None)
                self.set_name = dataset.__class__.__name__
            else:
                self.local_train = [IterLoader(DataLoader(
                    Preprocessor(cur_set.train, transform=trans, root=None),
                    batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                    sampler=RandomMultipleGallerySampler(
                        cur_set.train, args.num_instances),
                    pin_memory=False, num_workers=self.args.num_workers
                ), length=None) for cur_set in dataset]
                pid_list = [user.num_train_pids for user in dataset]
                self.padding = np.cumsum([0, ] + pid_list)
        self.max_iter = args.max_iter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tri_loss = TripletLoss(margin=0.5, is_avg=True)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduce='mean')
        self.dataset = dataset

    def fire_backbone(self, net):
        for param in net.patch_embed.parameters():
            param.requires_grad = True
        net.cls_token.requires_grad = True
        net.pos_embed.requires_grad = True
        for block in net.blocks:
            for param in block.parameters():
                # 默认先冻结所有Block内部参数
                param.requires_grad = True
            # 解冻Attention模块中的Prompt参数 (如果存在)

        net.prompt_embeddings.requires_grad = False
        net.deep_prompt_embeddings.requires_grad = False

    def print_model_parameters_in_million(self, models):
        a , b = 0 , 0
        for model in models:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params_all = sum(p.numel() for p in model.parameters())
            a += total_params
            b += total_params_all
            print(total_params, total_params_all)
        # 将参数数量转换为以百万（Million）为单位
        total_params_in_million = a / 1e6
        total_params_all_in_million = b / 1e6

        print(f"Total number of trainable parameters: {total_params_in_million:.2f}M")
        print(f"Total number of parameters (including non-trainable): {total_params_all_in_million:.2f}M")

    def count_parameters(self, model):
        """
        计算 PyTorch 模型中全部参数和可训练参数的数量。
        """
        # 计算全部参数
        total_params = sum(p.numel() for p in model.parameters())
        
        # 计算可训练参数 (通过 requires_grad 属性判断)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 格式化输出 (使用 M 或 K 单位)
        # def format_count(count):
        #     if count >= 1e6:
        #         return f"{count / 1e6:.2f} M"
        #     elif count >= 1e3:
        #         return f"{count / 1e3:.2f} K"
        #     return str(count)

        # print(f"--- {model.__class__.__name__} 模型参数统计 ---")
        # print(f"全部参数量 (Total parameters):   {total_params} ({format_count(total_params)})")
        # print(f"可训练参数量 (Trainable parameters): {trainable_params} ({format_count(trainable_params)})")
        # print(f"非训练参数量 (Non-trainable parameters): {total_params - trainable_params} ({format_count(total_params - trainable_params)})")
        
        return total_params, trainable_params
    
    def count_more_parameters(self, models):
        total = 0
        trainable = 0
        for model in models:
            a , b = self.count_parameters(model)
            total += a
            trainable += b
        def format_count(count):
            if count >= 1e6:
                return f"{count / 1e6:.2f} M"
            elif count >= 1e3:
                return f"{count / 1e3:.2f} K"
            return str(count)

        print(f"--- {model.__class__.__name__} 模型参数统计 ---")
        print(f"全部参数量 (Total parameters):   {total} ({format_count(total)})")
        print(f"可训练参数量 (Trainable parameters): {trainable} ({format_count(trainable)})")
        print(f"非训练参数量 (Non-trainable parameters): {total - trainable} ({format_count(total - trainable)})")
            
    def handle_set(self, dataset):
        cur_loader = IterLoader(DataLoader(
            Preprocessor(dataset.train, transform=self.trans, root=None),
            batch_size=self.args.batch_size, shuffle=False, drop_last=True,
            sampler=RandomMultipleGallerySampler(
                dataset.train, self.args.num_instances),
            pin_memory=True, num_workers=self.args.num_workers
        ), length=None)
        return cur_loader
    def get_optimizer(self, nets, epoch, optimizer_type='sgd'):
        # 假设分类器是传入的nets列表中的最后一个元素
        classifier = nets[-1]
        # 其他网络部分（例如特征提取器）
        feature_extractors = nets[:-1]
            
        self.count_more_parameters(nets)
        if optimizer_type.lower() == 'sgd':
            # 为特征提取器和分类器创建不同的参数组
            if len(nets) == 1:
                optimizer = torch.optim.SGD(
                    [{'params': sub_net.parameters()} for sub_net in nets],
                    lr=self.args.lr, weight_decay=self.args.weight_decay,
                    momentum=self.args.momentum
                )
            else:
                optimizer = torch.optim.SGD(
                    [
                        {'params': sub_net.parameters(), 'lr': self.args.lr} for sub_net in feature_extractors
                    ] + [
                        # 为分类器设置双倍的学习率
                        {'params': classifier.parameters(), 'lr': self.args.lr * 2}
                    ],
                    weight_decay=self.args.weight_decay,
                    momentum=self.args.momentum
                )
            lr_scheduler = MultiStepLR(
                optimizer, milestones=self.args.milestones, gamma=0.5)
        elif optimizer_type.lower() == 'scaffold':
            optimizer = ScaffoldOptimizer(
                [{'params': sub_net.parameters()} for sub_net in nets],
                lr=self.args.lr, weight_decay=self.args.weight_decay
            )
            lr_scheduler = MultiStepLR(optimizer,
                                       milestones=self.args.milestones, gamma=0.5)
        lr_scheduler.step(epoch)
        return optimizer

    def get_new_optimizer(self, epoch, nets=None, param_list=None, optimizer_type='sgd'):
        if param_list is None and nets is None:
            raise ValueError("Either 'nets' or 'param_list' must be provided.")
        if param_list is not None and nets is not None:
            print("Warning: Both 'nets' and 'param_list' were provided. 'param_list' will be used.")
        
        if nets:
            self.count_more_parameters(nets)

        optimizer_params = []

        if param_list is not None:
            # 如果传入的 param_list 已经是参数组的格式 (list of dicts)
            if isinstance(param_list, list) and all(isinstance(i, dict) for i in param_list):
                optimizer_params = param_list
            # 如果传入的是一个扁平的参数列表
            else:
                optimizer_params = [{'params': param_list}]
        
        else: # 保持原有的 nets 逻辑
            if optimizer_type.lower() == 'sgd':
                if len(nets) == 1:
                    optimizer_params = [{'params': nets[0].parameters()}]
                else:
                    classifier = nets[-1]
                    feature_extractors = nets[:-1]
                    optimizer_params = [
                        {'params': sub_net.parameters(), 'lr': self.args.lr} for sub_net in feature_extractors
                    ] + [
                        {'params': classifier.parameters(), 'lr': self.args.lr * 2}
                    ]
            else:
                optimizer_params = [{'params': sub_net.parameters()} for sub_net in nets]
        
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                optimizer_params,
                lr=self.args.lr,  # 默认lr，会被参数组内的lr覆盖
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum
            )
        elif optimizer_type.lower() == 'scaffold':
            # 注意：ScaffoldOptimizer 可能需要特定的参数格式，请根据其文档确认
            optimizer = ScaffoldOptimizer(
                optimizer_params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer type '{optimizer_type}' is not implemented.")

        lr_scheduler = MultiStepLR(optimizer, milestones=self.args.milestones, gamma=0.5)
        lr_scheduler.step(epoch)
        return optimizer

    def get_optimizer_clip_s2(self, nets, epoch, optimizer_type='Adam'):
        params = []
        keys = []
        for model in nets:
            for key, value in model.named_parameters():
                if "text_encoder" in key:
                    value.requires_grad_(False)
                    continue
                if "prompt_learner" in key:
                    value.requires_grad_(False)
                    continue
                if not value.requires_grad:
                    continue
                lr = 0.00035
                weight_decay = 0.0005
                if "bias" in key:
                    lr = 0.00035 * 2
                    weight_decay = 0.0005
                if False:
                    if "classifier" in key or "arcface" in key:
                        lr = cfg.SOLVER.BASE_LR * 2
                        print('Using two times learning rate for fc ')

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                keys += [key]
        if optimizer_type == 'SGD':
            optimizer = getattr(torch.optim, optimizer_type)(params, momentum=0.9)
        elif optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=0.00035, weight_decay=0.0005)
        else:
            optimizer = getattr(torch.optim, optimizer_type)(params)
        lr_scheduler = MultiStepLR(
            optimizer, milestones=self.args.milestones, gamma=0.1)
        lr_scheduler.step(epoch)
        return optimizer

    def make_optimizer_1stage(self, model, op_type):
        params = []  # 存储了参数组的信息
        keys = []  # 记录哪些参数被分到了这个组中
        for key, value in model.named_parameters():
            if "prompt_learner" in key:
                # lr = cfg.SOLVER.STAGE1.BASE_LR
                lr = 0.00035
                # weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
                weight_decay = 0.0005
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                keys += [key]
        if op_type == 'SGD':
            # optimizer = getattr(torch.optim, op_type)(params, momentum=cfg.SOLVER.STAGE1.MOMENTUM)
            optimizer = getattr(torch.optim, op_type)(params, momentum=0.9)
        elif op_type == 'AdamW':
            # optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
            optimizer = torch.optim.AdamW(params, lr=0.00035, weight_decay=0.0005)
        else:
            optimizer = getattr(torch.optim, op_type)(params)
        return optimizer

    def make_optimizer_2stage(self, models, epoch, op_type):
        params = []
        keys = []
        LR = self.args.lrs2
        WD = self.args.weight_decay

        # print('lr:',LR)

        for model in models:
            if hasattr(model, "text_encoder"):
                for param in model.text_encoder.parameters():
                    param.requires_grad_(False)

                if self.args.use_prompt:
                    for param in model.image_encoder.parameters():
                        param.requires_grad_(False)

                for name, param in model.named_parameters():
                    if "prompt" in name:
                        param.requires_grad = True

            for key, value in model.named_parameters():
                if "text_encoder" in key:
                    value.requires_grad_(False)
                    continue
                if "prompt_learner" in key:
                    value.requires_grad_(False)
                    continue
                if not value.requires_grad:
                    continue
                lr = LR
                weight_decay = WD
                if "bias" in key:
                    lr = LR * 2
                    weight_decay = WD
                if False:
                    if "classifier" in key or "arcface" in key:
                        lr = cfg.SOLVER.BASE_LR * 2
                        print('Using two times learning rate for fc ')

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                keys += [key]
            for key, value in model.named_parameters():
                if value.requires_grad:
                    print(key)
        if op_type == 'SGD':
            optimizer = getattr(torch.optim, op_type)(params, momentum=self.args.momentum)
        elif op_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
        else:
            optimizer = getattr(torch.optim, op_type)(params)
        # lr_scheduler = MultiStepLR(
        #         optimizer, milestones=self.args.milestones, gamma=0.1)
        # lr_scheduler.step(epoch)
        # print(f'lr: {lr_scheduler.get_lr()[0]:.7f}')
        return optimizer

    # resnet50, mAP=26.7
    def train_cls(self, net, global_epoch,
                  client_id, cls_layer, op_type='sgd'):
        net.train(True)
        optimizer = self.get_optimizer([net, cls_layer], global_epoch,
                                       optimizer_type=op_type)
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            feature = net(images)[0]
            feature_norm = F.normalize(feature)
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss_id = self.memory[client_id](feature_norm, labels).mean()  # 使用分类器计算损失

            loss = loss_ce + loss_tri + loss_id
            # loss = loss_id
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # f_new = net(images, style='store_true')[0] #提取风格化特征
                f_new = net(images, style='store_true')[1]  # 提取风格化特征
                self.memory[client_id].module.MomentumUpdate(f_new, labels)  # 更新特征原型
            if batch_idx % self.args.print_freq == 0:
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                      f'LossTri: {loss_tri.item():.2f}, LossId: {loss_id.item():.2f})')
            # print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
            #       f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}]  LossId: {loss_id.item():.2f})')

        return net.state_dict()

    def train_cls2(self, net, global_epoch,
                   client_id, cls_layer, op_type='sgd'):
        net.train(True)
        optimizer1 = self.get_optimizer([net, ], global_epoch,
                                        optimizer_type=op_type)
        optimizer2 = self.get_optimizer([net, cls_layer], global_epoch,
                                        optimizer_type=op_type)
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            feature, feature_norm = net(images)
            # score = cls_layer(feature)
            # loss_ce = self.ce_loss(score, labels)
            # loss_tri = self.tri_loss(feature, labels)
            loss_id = self.memory[client_id](feature_norm, labels).mean()  # 使用分类器计算损失
            # loss = loss_id
            optimizer1.zero_grad()
            loss_id.backward()
            optimizer1.step()

            f_new, f_new_norm = net(images, style='store_true')  # 提取风格化特征
            self.memory[client_id].module.MomentumUpdate(f_new_norm, labels)  # 更新特征原型
            score = cls_layer(f_new)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(f_new, labels)
            loss2 = loss_ce + loss_tri
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f}, LossId: {loss_id.item():.2f})')
            # print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
            #       f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}]  LossId: {loss_id.item():.2f})')

        return net.state_dict()

    # resnet50, style
    def train_mixstyle(self, net, global_epoch,
                       client_id, cls_layer, op_type='sgd'):
        net.train(True)
        optimizer = self.get_optimizer([net, cls_layer], global_epoch,
                                       optimizer_type=op_type)
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        start_time = time.time()
        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                      f'LossTri: {loss_tri.item():.2f})'
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')
        end_time = time.time()
        time_per_batch = (end_time - start_time) / self.max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))
        prompt_state = {}
        for key, value in net.named_parameters():
            if "prompt_embeddings" in key:  # 新Visual_prompt
                prompt_state[key] = value

        return net.state_dict() , prompt_state

    def train_fedprox(self, net, global_net, global_epoch,
                      client_id, cls_layer, op_type='sgd'):
        """
        使用FedProx进行本地训练的函数。

        Args:
          net: 本地模型 (nn.Module)，它将在本地数据上进行训练。
          global_net: 从服务器接收到的、本轮开始时的全局模型 (nn.Module)。
                      它的参数将被用作近端项的基准，并且不会被训练。
          ... (其他参数与您原来的函数相同)
        """
        net.train(True)
        # 将全局模型的参数设置为不需要梯度，以防意外更新
        global_net.eval()

        # 您的优化器只优化本地模型 net 和 cls_layer 的参数
        optimizer = self.get_optimizer([net, cls_layer], global_epoch,
                                       optimizer_type=op_type)
        self.local_train.new_epoch()
        start_time = time.time()

        # --- FedProx 关键步骤 1: 存储全局模型参数 ---
        # 将全局模型的参数提取出来，存成一个列表，方便后续计算
        global_params = list(global_net.parameters())
        # 如果 cls_layer 也是全局聚合的一部分，也需要加进来
        # global_cls_params = list(global_cls_layer.parameters())

        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # 前向传播 (与之前完全相同)
            feature = net(images)[0]
            score = cls_layer(feature)

            # 计算原始损失 (与之前完全相同)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss_original = loss_ce + loss_tri

            # --- FedProx 關鍵步驟 2: 計算近端項損失 ---
            prox_loss = 0.0
            # 1. 遍历本地 backbone (net) 的参数
            for local_param, global_param in zip(net.parameters(), global_params):
                # 计算本地参数和全局参数之间的L2距离的平方
                prox_loss += torch.pow(torch.norm(local_param - global_param), 2)

            # 2. [重要] 如果你的 cls_layer 也是全局聚合的，也需要对它计算prox_loss
            # for local_cls_param, global_cls_param in zip(cls_layer.parameters(), global_cls_params):
            #     prox_loss += torch.pow(torch.norm(local_cls_param - global_cls_param), 2)

            # --- FedProx 關鍵步驟 3: 將近端項添加到總損失中 ---
            # self.args.mu 是您需要设置的超参数
            loss = loss_original + (self.args.mu / 2) * prox_loss

            # 反向传播和优化 (与之前完全相同)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                # 在日志中可以加上prox_loss，方便调试
                prox_loss_item = (self.args.mu / 2) * prox_loss.item()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] '
                      f'(LossCE: {loss_ce.item():.2f}, LossTri: {loss_tri.item():.2f}, LossProx: {prox_loss_item:.2f})'  # 添加了Prox Loss
                      f' time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')

        end_time = time.time()
        time_per_batch = (end_time - start_time) / self.max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))

        # 您原有的 prompt_state 逻辑保持不变
        prompt_state = {}
        for key, value in net.named_parameters():
            if "prompt_embeddings" in key:
                prompt_state[key] = value

        return net.state_dict(), prompt_state


    def train_fedpav(self, net, global_epoch,
                       client_id, cls_layer, op_type='sgd'):
        net.train(True)
        optimizer = self.get_optimizer([net, cls_layer], global_epoch,
                                       optimizer_type=op_type)
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        start_time = time.time()
        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                      f'LossTri: {loss_tri.item():.2f})'
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')
        end_time = time.time()
        time_per_batch = (end_time - start_time) / self.max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))
        prompt_state = {}
        for key, value in net.named_parameters():
            if "prompt_embeddings" in key:  # 新Visual_prompt
                prompt_state[key] = value

        return net.state_dict() , prompt_state

    def train_crossstyle(self, net, global_epoch,
                         client_id, cls_layer, op_type='sgd'):
        net.train(True)
        optimizer = self.get_optimizer([net, cls_layer], global_epoch,
                                       optimizer_type=op_type)
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        start_time = time.time()
        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # images = cn_op_2ins_space_chan(images, beta=0.5, crop='both')
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                      f'LossTri: {loss_tri.item():.2f})'
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')
        end_time = time.time()
        time_per_batch = (end_time - start_time) / self.max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))
        prompt_state = {}
        for key, value in net.named_parameters():
            if "prompt_embeddings" in key:  # 新Visual_prompt
                prompt_state[key] = value
        return net.state_dict() , prompt_state

    # vanilla aug, mAP=34.2
    def train_dacs(self, net, avg_net, aug_mod, global_epoch, client_id,
                   cls_layer, op_type='sgd' , tsne = False):
        net.train(True)
        avg_net.train(True)

        self.local_train.new_epoch()
        # avg optimizer #本地侧全局
        optimizer = self.get_optimizer(
            nets=[avg_net, aug_mod, cls_layer], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # ssim_scores, is_epoch = [], 0
        start_time = time.time()
        max_iter = self.max_iter
        trained_prompts = []
        for batch_idx in range(max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            b_size = images.shape[0]
            cur_mean, cur_var = images.mean((2, 3)).view(
                b_size, -1, 1, 1), images.var((2, 3)).view(b_size, -1, 1, 1)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)  # 计算均值和方差就行归一化

            # stage1: expert train #专家训练 使用局部模型提取原始图像，计算损失，优化本地模型
            feature, feature_avg = net(images)[0], avg_net(images)[0]
            score, score_avg = cls_layer(feature), cls_layer(feature_avg)
            loss_erm = self.ce_loss(score, labels) + self.tri_loss(feature, labels)
            optimizer_local.zero_grad()
            loss_erm.backward()
            optimizer_local.step()

            # stage2: joint train #联合训练 使用全局模型提取原始图像，计算损失，优化本地侧全局模型
            # basic cls loss with ori images and avg_net
            loss_ce = self.ce_loss(score_avg, labels)
            loss_tri = self.tri_loss(feature_avg, labels)
            loss_aux, loss_aug, loss_wd = 0, 0, 0

            # aug avg model 如果不是第一轮全局训练
            if global_epoch > 0:
                # generate freezed global model to detach grad, training=False version
                freeze_avg = freeze_model(copy.deepcopy(avg_net))  # 冻结全局模型
                # transformed image
                aug_image = aug_mod(norm_image)  # 使用数据增强模块增强图像

                # obtain H(fG(x')), use a frozen avg_net to avoid updating avg_net model
                aug_feature_avg_freeze = freeze_avg(aug_image)[0]  # 全局模型对增强图像的分数
                aug_score_avg_freeze = cls_layer(aug_feature_avg_freeze)
                # obtain H(fL(x')), optimizer does not contain net.params(), so we do not need to use 'freeze_model'
                aug_feature_local = net(aug_image)[0]  # 本地模型对增强图像的分数
                aug_score_local = cls_layer(aug_feature_local)
                # generate H(fG(x)), use a frozen avg_net
                score_avg_freeze = cls_layer(freeze_avg(images)[0])  # 全局模型对原始图像的分数
                # au loss,  H(fG(x)) < H(fG(x')) < H(fL(x'))
                loss_aux = get_auth_loss(  # 增强损失  优化图像增强模型
                    get_entropy(F.softmax(aug_score_avg_freeze)),
                    get_entropy(F.softmax(score_avg_freeze)),
                    get_entropy(F.softmax(aug_score_local))
                )

                # aug images to update avg_net 计算增强图像在全局模型的损失来更新
                aug_feature_avg = avg_net(aug_image)[0]
                aug_score_avg = cls_layer(aug_feature_avg)
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels)

                # div loss 训练增强模型
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                          F.mse_loss(cur_var, shift_var)

            # optimize avg model, share across domains
            loss = loss_ce + loss_tri + loss_aug + loss_wd + self.args.lam * loss_aux
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{max_iter}] (LossCE: {loss_ce.item():.2f}, '
                      f'LossTri: {loss_tri.item():.2f}, LossAux:{float(loss_aux):.2f})'
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')
        end_time = time.time()
        time_per_batch = (end_time - start_time) / max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))
        prompt_state = {}

        for key, value in avg_net.named_parameters():
            if "prompt_embeddings" in key:  # 新Visual_prompt
                prompt_state[key] = value
        # # print ssim
        # ssim_epoch = max(ssim_scores) if len(ssim_scores) else 0
        # is_epoch = inception_score(aug_image.detach()) if global_epoch>0 else 0

        # if global_epoch % 5 == 0:
        #     print(f'Dataset {self.set_name}. SSIM: {ssim_epoch:4.3f}, IS: {is_epoch:4.3f}.')

        return avg_net.state_dict() , prompt_state , net.module.base.prompt_embeddings.clone().detach().cpu()
        
    def train_dacs_IL(self, net, avg_net, aug_mod, global_epoch, client_id,
                      cls_layer, F_news, Labels, op_type='sgd', bad_iamges=None):
        net.train(True)
        avg_net.train(True)
        memory = copy.deepcopy(self.memory)
        self.local_train.new_epoch()
        # avg optimizer #本地侧全局
        optimizer = self.get_optimizer(
            nets=[avg_net, cls_layer, aug_mod, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )

        # ssim_scores, is_epoch = [], 0
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            b_size = images.shape[0]
            cur_mean, cur_var = images.mean((2, 3)).view(b_size, -1, 1, 1), images.var((2, 3)).view(b_size, -1, 1, 1)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)  # 计算均值和方差进行图像归一化

            # stage1: expert train #专家训练
            # feature, feature_avg = net(images, style = None)[0], avg_net(images, style = None)[0]
            feature, feature_norm = net(images, style=False)
            feature_avg, feature_avg_norm = avg_net(images, style=False)
            # score, score_avg = cls_layer(feature), cls_layer(feature_avg)
            # loss_id_local = memory[client_id](feature_norm, labels).mean() + self.ce_loss(score, labels) + self.tri_loss(feature, labels)
            loss_id_local = memory[client_id](feature_norm, labels).mean()
            optimizer_local.zero_grad()
            loss_id_local.backward()
            optimizer_local.step()

            # stage2: joint train #联合训练
            # basic cls loss with ori images and avg_net
            # loss_id = memory[client_id](feature_avg_norm, labels).mean() + self.ce_loss(score_avg, labels) + self.tri_loss(feature_avg, labels)
            loss_id = memory[client_id](feature_avg_norm, labels).mean()
            aug_image = aug_mod(norm_image)  # 使用数据增强模块增强图像

            # aug_image_to_save = (aug_image + 1) / 2
            image_to_save = images
            aug_image_to_save = aug_image

            # 保存每个图像，假设批量大小为B
            aug_save_dir = '/mnt/data/rcy/DACS_IL-mainwork/image/aug_img'
            save_dir = '/mnt/data/rcy/DACS_IL-mainwork/image/img'
            for i in range(aug_image_to_save.size(0)):
                # 生成文件名，例如 'augmented_0.png', 'augmented_1.png', ...
                filename = f'Client_{client_id}_{global_epoch}_{labels[i]}_{i}.png'
                aug_filename = f'augmented_Client_{global_epoch}_{client_id}_{labels[i]}_{i}.png'
                save_path = os.path.join(save_dir, filename)
                save_path_aug = os.path.join(aug_save_dir, aug_filename)

                # 保存单个图像
                save_image(image_to_save[i], save_path)
                save_image(aug_image_to_save[i], save_path_aug)
            print('finished!')
            del image_to_save, aug_image_to_save

            with torch.no_grad():
                # f_new = avg_net(images,style='store_true')[1] #提取风格化特征
                f_new = avg_net(aug_image, style=False)[1]
                # if bad_iamges is not None:
                #     bad_iamges.append(aug_image)
                if F_news is not None:
                    F_news[client_id].append(f_new)
                if Labels is not None:
                    Labels[client_id].append(labels)
                memory[client_id].module.MomentumUpdate(f_new, labels)

            loss_aux, loss_aug, loss_wd = 0, 0, 0

            # aug avg model 如果不是第一轮全局训练
            if global_epoch > 0:
                # generate freezed global model to detach grad, training=False version
                freeze_avg = freeze_model(copy.deepcopy(avg_net))  # 冻结全局模型
                # transformed image
                # aug_image = aug_mod(norm_image) #使用数据增强模块增强图像
                # ssim_scores.append(ssim(aug_image.detach(), images.detach(), data_range=1, size_average=True).item())

                # obtain H(fG(x')), use a frozen avg_net to avoid updating avg_net model
                aug_feature_avg_freeze = freeze_avg(aug_image, style=False)  # 全局模型对增强图像的分数
                aug_score_avg_freeze = cls_layer(aug_feature_avg_freeze)
                # obtain H(fL(x')), optimizer does not contain net.params(), so we do not need to use 'freeze_model'
                aug_feature_local = net(aug_image, style=False)[0]  # 本地模型对增强图像的分数
                aug_score_local = cls_layer(aug_feature_local)
                # generate H(fG(x)), use a frozen avg_net
                score_avg_freeze = cls_layer(freeze_avg(images, style=False))  # 全局模型对原始图像的分数
                # au loss,  H(fG(x)) < H(fG(x')) < H(fL(x'))
                loss_aux = get_auth_loss(  # 增强损失
                    get_entropy(F.softmax(aug_score_avg_freeze)),
                    get_entropy(F.softmax(score_avg_freeze)),
                    get_entropy(F.softmax(aug_score_local))
                )

                # aug images to update avg_net 计算增强图像在全局模型的损失来更新
                aug_feature_avg, aug_feature_avg_norm = avg_net(aug_image, style=False)
                aug_score_avg = cls_layer(aug_feature_avg)
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels)

                # div loss 训练增强模型
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                          F.mse_loss(cur_var, shift_var)

            # optimize avg model, share across domains
            # loss = loss_ce + loss_tri + loss_aug + loss_wd + self.args.lam * loss_aux
            loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux
            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
            #       f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
            #       f'LossTri: {loss_tri.item():.2f}, LossAux:{float(loss_aux):.2f})')
            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                  f'LossID: {loss_id.item():.2f}, LossAux:{float(loss_aux):.2f})')
        # # print ssim
        # ssim_epoch = max(ssim_scores) if len(ssim_scores) else 0
        # is_epoch = inception_score(aug_image.detach()) if global_epoch>0 else 0

        # if global_epoch % 5 == 0:
        #     print(f'Dataset {self.set_name}. SSIM: {ssim_epoch:4.3f}, IS: {is_epoch:4.3f}.')

        return avg_net.state_dict()

    def train_dacs_IL_VIT(self, net, avg_net, aug_mod, global_epoch, client_id,
                          cls_layer, F_news, Labels, op_type='sgd'):
        net.train(True)
        avg_net.train(True)

        memory = copy.deepcopy(self.memory)

        self.local_train.new_epoch()
        # avg optimizer #本地侧全局
        if not self.args.use_prompt:
            optimizer = self.get_optimizer(
                nets=[avg_net, aug_mod, cls_layer, ], epoch=global_epoch,
                optimizer_type=op_type, 
            )
            # local optimizer
            optimizer_local = self.get_optimizer(
                nets=[net, ], epoch=global_epoch,
                optimizer_type=op_type
            )
        else:
            prompt_params_avg = [p for n, p in avg_net.named_parameters() if "prompt_embeddings" in n and p.requires_grad]
            prompt_params_local = [p for n, p in net.named_parameters() if "prompt_embeddings" in n and p.requires_grad]

            # 2. 将所有需要优化的参数合并到一个列表
            # 对于 optimizer，我们需要训练 prompt_avg, aug_mod, 和 cls_layer
            # 并且我们想给 cls_layer 一个不同的学习率
            # 所以这里我们需要构建一个更复杂的参数组列表
            params_to_optimize_global = [
                {'params': prompt_params_avg, 'lr': self.args.lr},
                {'params': aug_mod.parameters(), 'lr': self.args.lr},
                {'params': cls_layer.parameters(), 'lr': self.args.lr * 2} # 保持特殊学习率
            ]

            # 对于 optimizer_local，我们只需要训练 prompt_local
            params_to_optimize_local = prompt_params_local # 这是一个简单的列表
            optimizer = self.get_new_optimizer(
                epoch=global_epoch, 
                param_list=params_to_optimize_global, # <--- 传入参数列表
                optimizer_type=op_type
            )
        # 优化器2
            optimizer_local = self.get_new_optimizer(
                epoch=global_epoch,
                param_list=params_to_optimize_local, # <--- 传入参数列表
                optimizer_type=op_type
            )
        
        global_params = list(avg_net.parameters())
        # ssim_scores, is_epoch = [], 0
        start_time = time.time()
        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            b_size = images.shape[0]
            cur_mean, cur_var = images.mean((2, 3)).view(b_size, -1, 1, 1), images.var((2, 3)).view(b_size, -1, 1, 1)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)  # 计算均值和方差进行图像归一化

            # stage1: expert train #专家训练
            # feature, feature_avg = net(images, style = None)[0], avg_net(images, style = None)[0]
            feature = net(images)[0]
            feature_norm = F.normalize(feature)
            feature_avg = avg_net(images)[0]
            feature_avg_norm = F.normalize(feature_avg)
            # score, score_avg = cls_layer(feature), cls_layer(feature_avg)
            loss_tri = self.tri_loss(feature, labels)
            loss_id_local = memory[client_id](feature_norm, labels).mean() + loss_tri
            # loss_id_local = memory[client_id](feature_norm, labels).mean()


            # 将 prox_loss 加入本地损失函数
            # 注意：这里的超参数 mu 应该从 args 传入，而不是写死
            optimizer_local.zero_grad()
            loss_id_local.backward()
            optimizer_local.step()

            # stage2: joint train #联合训练
            # basic cls loss with ori images and avg_net
            loss_id = memory[client_id](feature_avg_norm, labels).mean() + self.tri_loss(feature_avg, labels)
            # loss_id = memory[client_id](feature_avg_norm, labels).mean()
            with torch.no_grad():
                aug_image = aug_mod(norm_image)  # 使用数据增强模块增强图像
                f_new = avg_net(aug_image)[0]
                f_new_norm = F.normalize(f_new)
                F_news[client_id].append(f_new_norm)
                Labels[client_id].append(labels)
                memory[client_id].module.MomentumUpdate(f_new_norm, labels)

            loss_aux, loss_aug, loss_wd = 0, 0, 0

            # aug avg model 如果不是第一轮全局训练
            if global_epoch > 0:
                # generate freezed global model to detach grad, training=False version
                freeze_avg = freeze_model(copy.deepcopy(avg_net))  # 冻结全局模型
                # transformed image
                aug_image = aug_mod(norm_image)  # 使用数据增强模块增强图像
                # ssim_scores.append(ssim(aug_image.detach(), images.detach(), data_range=1, size_average=True).item())

                # obtain H(fG(x')), use a frozen avg_net to avoid updating avg_net model
                aug_feature_avg_freeze = freeze_avg(aug_image)[0]  # 全局模型对增强图像的分数
                aug_score_avg_freeze = cls_layer(aug_feature_avg_freeze)
 
                # obtain H(fL(x')), optimizer does not contain net.params(), so we do not need to use 'freeze_model'
                aug_feature_local = net(aug_image)[0]  # 本地模型对增强图像的分数
                aug_score_local = cls_layer(aug_feature_local)
                # generate H(fG(x)), use a frozen avg_net
                score_avg_freeze = cls_layer(freeze_avg(images)[0])  # 全局模型对原始图像的分数
                # au loss,  H(fG(x)) < H(fG(x')) < H(fL(x'))
                loss_aux = get_auth_loss(  # 增强损失
                    get_entropy(F.softmax(aug_score_avg_freeze)),
                    get_entropy(F.softmax(score_avg_freeze)),
                    get_entropy(F.softmax(aug_score_local))
                )

                # aug images to update avg_net 计算增强图像在全局模型的损失来更新
                aug_feature_avg = avg_net(aug_image)[0]
                aug_score_avg = cls_layer(aug_feature_avg)
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels)

                # div loss 训练增强模型
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                          F.mse_loss(cur_var, shift_var)

            
            # optimize avg model, share across domains
            # loss = loss_ce + loss_tri + loss_aug + loss_wd + self.args.lam * loss_aux
            loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux
            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
            #       f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
            #       f'LossTri: {loss_tri.item():.2f}, LossAux:{float(loss_aux):.2f})')'

            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                      f'LossID: {loss_id.item():.2f}, LossAux:{float(loss_aux):.2f} ,'
                      f'LossTri: {loss_tri.item():.2f}, '
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')

        end_time = time.time()
        time_per_batch = (end_time - start_time) / self.max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))
        prompt_state = {}
        for key, value in avg_net.named_parameters():
            if "prompt_embeddings" in key:  # 新Visual_prompt
                prompt_state[key] = value

        print(prompt_state.keys())

        # # print ssim
        # ssim_epoch = max(ssim_scores) if len(ssim_scores) else 0
        # is_epoch = inception_score(aug_image.detach()) if global_epoch>0 else 0

        # if global_epoch % 5 == 0:
        #     print(f'Dataset {self.set_name}. SSIM: {ssim_epoch:4.3f}, IS: {is_epoch:4.3f}.')

        return avg_net.state_dict(), prompt_state

    
    # vanilla aug, mAP=26
    def train_moon(self, net, prev_net, avg_net, global_epoch, client_id,
                   cls_layer, op_type='sgd'):
        net.train(True)
        avg_net.eval()
        prev_net.eval()
        self.local_train.new_epoch()

        cos_func = torch.nn.CosineSimilarity(dim=-1)
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, cls_layer], epoch=global_epoch,
            optimizer_type=op_type
        )

        start_time = time.time()
        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # basic cls loss for local model
            feature = net(images)
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss_con = 0

            # aug avg model
            if global_epoch > 0:
                with torch.no_grad():
                    avg_feat = avg_net(images)
                    prev_feat = prev_net(images)


                score_pos = cos_func(feature, avg_feat).reshape(-1, 1)
                score_neg = cos_func(feature, prev_feat).reshape(-1, 1)

                denominator_score = torch.cat(
                    [score_pos, score_neg], dim=1) / self.args.temp
                con_labels = torch.zeros(
                    score_pos.shape[0]).to(self.device).long()
                loss_con = F.cross_entropy(denominator_score, con_labels)

            # optimize avg model, sahre across domains
            loss = loss_ce + loss_tri + self.args.lam * loss_con
            optimizer_local.zero_grad()
            loss.backward()
            optimizer_local.step()
            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                      f'LossTri: {loss_tri.item():.2f}, LossCon:{float(loss_con):.2f})'
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')

        end_time = time.time()
        time_per_batch = (end_time - start_time) / self.max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))
        prompt_state = {}
        for key, value in net.named_parameters():
            if "prompt_embeddings" in key:  # 新Visual_prompt
                prompt_state[key] = value
        return net.state_dict() , prompt_state

    def train_dacs_snr(self, net, avg_net, aug_mod,
                       global_epoch, client_id,
                       fc, fc1, fc2, fc3, op_type='sgd'):
        net.train(True)
        avg_net.train(True)

        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[avg_net, fc, fc1, fc2, fc3, aug_mod],
            epoch=global_epoch, optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            cur_mean, cur_var = images.mean(0), images.var(0)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)

            # fine tune local snr
            local_features, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = net(images)
            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            local_score = fc(local_features)
            loss_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob),
                                                  get_entropy(x_1_useless_prob)) + \
                             0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob),
                                                  get_entropy(x_2_useless_prob)) + \
                             0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                x_3_useful_prob), get_entropy(x_3_useless_prob))
            loss = loss_causality + \
                   self.tri_loss(local_features, labels) + \
                   self.ce_loss(local_score, labels)
            optimizer_local.zero_grad()
            loss.backward()
            optimizer_local.step()

            # basic cls loss for avg model
            feature_avg, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = avg_net(
                images)
            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            score_avg = fc(feature_avg)
            # Causality loss for avg model:
            loss_causality_avg = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob),
                                                      get_entropy(x_1_useless_prob)) + \
                                 0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob),
                                                      get_entropy(x_2_useless_prob)) + \
                                 0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                x_3_useful_prob), get_entropy(x_3_useless_prob))
            score_avg = fc(feature_avg)
            loss_ce_avg = self.ce_loss(score_avg, labels)
            loss_tri_avg = self.tri_loss(feature_avg, labels)
            loss_aug, loss_aux_avg = 0, 0

            # aug avg model
            if global_epoch > 0:
                aug_image = aug_mod(norm_image)

                aug_feature_avg, x_IN_1_pool_aug, x_1_useful_pool_aug, x_1_useless_pool_aug, \
                    x_IN_2_pool_aug, x_2_useful_pool_aug, x_2_useless_pool_aug, \
                    x_IN_3_pool_aug, x_3_useful_pool_aug, x_3_useless_pool_aug = avg_net(
                    aug_image)
                x_IN_1_prob_aug = F.softmax(fc1(x_IN_1_pool_aug))
                x_1_useful_prob_aug = F.softmax(fc1(x_1_useful_pool_aug))
                x_1_useless_prob_aug = F.softmax(fc1(x_1_useless_pool_aug))
                x_IN_2_prob_aug = F.softmax(fc2(x_IN_2_pool_aug))
                x_2_useful_prob_aug = F.softmax(fc2(x_2_useful_pool_aug))
                x_2_useless_prob_aug = F.softmax(fc2(x_2_useless_pool_aug))
                x_IN_3_prob_aug = F.softmax(fc3(x_IN_3_pool_aug))
                x_3_useful_prob_aug = F.softmax(fc3(x_3_useful_pool_aug))
                x_3_useless_prob_aug = F.softmax(fc3(x_3_useless_pool_aug))
                loss_aug_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob_aug),
                                                          get_entropy(x_1_useful_prob_aug),
                                                          get_entropy(x_1_useless_prob_aug)) + \
                                     0.01 * get_auth_loss(get_entropy(x_IN_2_prob_aug),
                                                          get_entropy(x_2_useful_prob_aug),
                                                          get_entropy(x_2_useless_prob_aug)) + \
                                     0.01 * get_auth_loss(get_entropy(x_IN_3_prob_aug), get_entropy(
                    x_3_useful_prob_aug), get_entropy(x_3_useless_prob_aug))

                aug_feature_local = net(aug_image)[0]
                aug_score_avg, aug_score_local = fc(
                    aug_feature_avg), fc(aug_feature_local)
                # loss to disentangle, fL(I) < fA(I) < fA(I') < fL(I')
                loss_aux_avg = get_auth_loss(
                    get_entropy(F.softmax(aug_score_avg)),
                    get_entropy(F.softmax(score_avg)),
                    get_entropy(F.softmax(aug_score_local))
                )
                loss_aug = self.ce_loss(
                    aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels) + loss_aug_causality

            # optimize avg model, sahre across domains
            loss = loss_ce_avg + loss_tri_avg + loss_causality_avg + \
                   loss_aug + self.args.lam * loss_aux_avg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce_avg.item():.2f}, '
                  f'LossTri: {loss_tri_avg.item():.2f}, LossAux:{float(loss_aux_avg):.2f})')

        return avg_net.state_dict()

    def train_dacs_IL_snr(self, net, avg_net, aug_mod,
                          global_epoch, client_id,
                          fc, fc1, fc2, fc3, F_news, Labels, op_type='sgd'):
        net.train(True)
        avg_net.train(True)

        memory = copy.deepcopy(self.memory)

        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[avg_net, fc, fc1, fc2, fc3, aug_mod],
            epoch=global_epoch, optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            cur_mean, cur_var = images.mean(0), images.var(0)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)

            # 原始图像训练本地模型
            local_features, local_features_norm, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = net(images)

            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            loss_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob),
                                                  get_entropy(x_1_useless_prob)) + \
                             0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob),
                                                  get_entropy(x_2_useless_prob)) + \
                             0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                x_3_useful_prob), get_entropy(x_3_useless_prob))
            # score_avg = fc(local_features) #消融
            # loss_id_local = self.ce_loss(score_avg, labels) #消融

            loss_id_local = memory[client_id](local_features_norm, labels).mean()
            loss = loss_causality + loss_id_local
            optimizer_local.zero_grad()
            loss.backward()
            optimizer_local.step()

            # 原始图像训练本地侧全局模型
            feature_avg, feature_avg_norm, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = avg_net(
                images)

            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            score_avg = fc(feature_avg)
            # Causality loss for avg model:
            loss_causality_avg = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob),
                                                      get_entropy(x_1_useless_prob)) + \
                                 0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob),
                                                      get_entropy(x_2_useless_prob)) + \
                                 0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                x_3_useful_prob), get_entropy(x_3_useless_prob))

            # score_avg = fc(feature_avg) #消融
            # loss_id_avg = self.ce_loss(score_avg, labels) #消融

            loss_id_avg = memory[client_id](feature_avg_norm, labels).mean()
            loss_aug, loss_aux_avg = 0, 0
            aug_image = aug_mod(norm_image)

            # aug_image_to_save = (aug_image + 1) / 2
            image_to_save = images
            aug_image_to_save = aug_image

            # 保存每个图像，假设批量大小为B
            aug_save_dir = '/mnt/data/rcy/DACS_IL-mainwork/image/aug_img'
            save_dir = '/mnt/data/rcy/DACS_IL-mainwork/image/img'
            # for i in range(8):
            #     # 生成文件名，例如 'augmented_0.png', 'augmented_1.png', ...
            #     filename = f'Client_{client_id}_epoch_{global_epoch}_label_{labels[i]}.png'
            #     aug_filename = f'augmented_Client_{client_id}_epoch_{global_epoch}_label_{labels[i]}.png'
            #     save_path = os.path.join(save_dir, filename)
            #     save_path_aug = os.path.join(aug_save_dir, aug_filename)

            #     # 保存单个图像
            #     save_image(image_to_save[i], save_path)
            #     save_image(aug_image_to_save[i], save_path_aug)
            # print('finished!')
            del image_to_save, aug_image_to_save

            with torch.no_grad():
                f_new = avg_net(aug_image)[1]
                F_news[client_id].append(f_new)
                Labels[client_id].append(labels)
                memory[client_id].module.MomentumUpdate(f_new, labels)

            # aug avg model
            if global_epoch > 0:
                aug_feature_avg, aug_feature_avg_norm, x_IN_1_pool_aug, x_1_useful_pool_aug, x_1_useless_pool_aug, \
                    x_IN_2_pool_aug, x_2_useful_pool_aug, x_2_useless_pool_aug, \
                    x_IN_3_pool_aug, x_3_useful_pool_aug, x_3_useless_pool_aug = avg_net(
                    aug_image)
                x_IN_1_prob_aug = F.softmax(fc1(x_IN_1_pool_aug))
                x_1_useful_prob_aug = F.softmax(fc1(x_1_useful_pool_aug))
                x_1_useless_prob_aug = F.softmax(fc1(x_1_useless_pool_aug))
                x_IN_2_prob_aug = F.softmax(fc2(x_IN_2_pool_aug))
                x_2_useful_prob_aug = F.softmax(fc2(x_2_useful_pool_aug))
                x_2_useless_prob_aug = F.softmax(fc2(x_2_useless_pool_aug))
                x_IN_3_prob_aug = F.softmax(fc3(x_IN_3_pool_aug))
                x_3_useful_prob_aug = F.softmax(fc3(x_3_useful_pool_aug))
                x_3_useless_prob_aug = F.softmax(fc3(x_3_useless_pool_aug))
                loss_aug_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob_aug),
                                                          get_entropy(x_1_useful_prob_aug),
                                                          get_entropy(x_1_useless_prob_aug)) + \
                                     0.01 * get_auth_loss(get_entropy(x_IN_2_prob_aug),
                                                          get_entropy(x_2_useful_prob_aug),
                                                          get_entropy(x_2_useless_prob_aug)) + \
                                     0.01 * get_auth_loss(get_entropy(x_IN_3_prob_aug), get_entropy(
                    x_3_useful_prob_aug), get_entropy(x_3_useless_prob_aug))

                aug_feature_local = net(aug_image)[0]
                aug_score_avg, aug_score_local = fc(
                    aug_feature_avg), fc(aug_feature_local)
                # loss to disentangle, fL(I) < fA(I) < fA(I') < fL(I')
                loss_aux_avg = get_auth_loss(
                    get_entropy(F.softmax(aug_score_avg)),
                    get_entropy(F.softmax(score_avg)),
                    get_entropy(F.softmax(aug_score_local))
                )
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg,
                                                                               labels) + loss_aug_causality

            # optimize avg model, sahre across domains
            loss = loss_id_avg + loss_causality_avg + loss_aug + self.args.lam * loss_aux_avg  # 消融

            # loss = loss_id_avg + loss_causality_avg + self.args.lam * loss_aux_avg #消融

            # loss = loss_id_avg + loss_causality_avg #消融
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                  f'LossID: {loss_id_avg.item():.2f}, LossAux:{float(loss_aux_avg):.2f})')

        return avg_net.state_dict()

    def train_dacs_IL_snr_ablation(self, net, avg_net, aug_mod,
                                   global_epoch, client_id,
                                   fc, fc1, fc2, fc3, F_news, Labels, op_type='sgd'):
        net.train(True)
        avg_net.train(True)

        memory = copy.deepcopy(self.memory)

        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[avg_net, fc, fc1, fc2, fc3, aug_mod],
            epoch=global_epoch, optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            cur_mean, cur_var = images.mean(0), images.var(0)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)

            # 原始图像训练本地模型
            local_features, local_features_norm, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = net(images)

            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            loss_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob),
                                                  get_entropy(x_1_useless_prob)) + \
                             0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob),
                                                  get_entropy(x_2_useless_prob)) + \
                             0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                x_3_useful_prob), get_entropy(x_3_useless_prob))

            # score_avg = fc(local_features) #消融
            # loss_id_local = self.ce_loss(score_avg, labels) #消融

            loss_id_local = memory[client_id](local_features_norm, labels).mean()
            loss = loss_causality + loss_id_local
            optimizer_local.zero_grad()
            loss.backward()
            optimizer_local.step()

            # 原始图像训练本地侧全局模型
            feature_avg, feature_avg_norm, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = avg_net(
                images)

            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            score_avg = fc(feature_avg)
            # Causality loss for avg model:
            loss_causality_avg = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob),
                                                      get_entropy(x_1_useless_prob)) + \
                                 0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob),
                                                      get_entropy(x_2_useless_prob)) + \
                                 0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                x_3_useful_prob), get_entropy(x_3_useless_prob))

            # score_avg = fc(feature_avg) #消融
            # loss_id_avg = self.ce_loss(score_avg, labels) #消融

            loss_id_avg = memory[client_id](feature_avg_norm, labels).mean()
            loss_aug, loss_aux_avg = 0, 0
            aug_image = aug_mod(norm_image)

            # aug_image_to_save = (aug_image + 1) / 2
            image_to_save = images
            aug_image_to_save = aug_image

            # 保存每个图像，假设批量大小为B
            aug_save_dir = '/mnt/data/rcy/DACS_IL-mainwork/image/aug_img'
            save_dir = '/mnt/data/rcy/DACS_IL-mainwork/image/img'
            # for i in range(8):
            #     # 生成文件名，例如 'augmented_0.png', 'augmented_1.png', ...
            #     filename = f'Client_{client_id}_epoch_{global_epoch}_label_{labels[i]}.png'
            #     aug_filename = f'augmented_Client_{client_id}_epoch_{global_epoch}_label_{labels[i]}.png'
            #     save_path = os.path.join(save_dir, filename)
            #     save_path_aug = os.path.join(aug_save_dir, aug_filename)

            #     # 保存单个图像
            #     save_image(image_to_save[i], save_path)
            #     save_image(aug_image_to_save[i], save_path_aug)
            # print('finished!')
            del image_to_save, aug_image_to_save

            with torch.no_grad():
                f_new = avg_net(aug_image)[1]
                F_news[client_id].append(f_new)
                Labels[client_id].append(labels)
                memory[client_id].module.MomentumUpdate(f_new, labels)

            # aug avg model
            if global_epoch > 0:
                aug_feature_avg, aug_feature_avg_norm, x_IN_1_pool_aug, x_1_useful_pool_aug, x_1_useless_pool_aug, \
                    x_IN_2_pool_aug, x_2_useful_pool_aug, x_2_useless_pool_aug, \
                    x_IN_3_pool_aug, x_3_useful_pool_aug, x_3_useless_pool_aug = avg_net(
                    aug_image)
                x_IN_1_prob_aug = F.softmax(fc1(x_IN_1_pool_aug))
                x_1_useful_prob_aug = F.softmax(fc1(x_1_useful_pool_aug))
                x_1_useless_prob_aug = F.softmax(fc1(x_1_useless_pool_aug))
                x_IN_2_prob_aug = F.softmax(fc2(x_IN_2_pool_aug))
                x_2_useful_prob_aug = F.softmax(fc2(x_2_useful_pool_aug))
                x_2_useless_prob_aug = F.softmax(fc2(x_2_useless_pool_aug))
                x_IN_3_prob_aug = F.softmax(fc3(x_IN_3_pool_aug))
                x_3_useful_prob_aug = F.softmax(fc3(x_3_useful_pool_aug))
                x_3_useless_prob_aug = F.softmax(fc3(x_3_useless_pool_aug))
                loss_aug_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob_aug),
                                                          get_entropy(x_1_useful_prob_aug),
                                                          get_entropy(x_1_useless_prob_aug)) + \
                                     0.01 * get_auth_loss(get_entropy(x_IN_2_prob_aug),
                                                          get_entropy(x_2_useful_prob_aug),
                                                          get_entropy(x_2_useless_prob_aug)) + \
                                     0.01 * get_auth_loss(get_entropy(x_IN_3_prob_aug), get_entropy(
                    x_3_useful_prob_aug), get_entropy(x_3_useless_prob_aug))

                aug_feature_local = net(aug_image)[0]
                aug_score_avg, aug_score_local = fc(
                    aug_feature_avg), fc(aug_feature_local)
                # loss to disentangle, fL(I) < fA(I) < fA(I') < fL(I')
                loss_aux_avg = get_auth_loss(
                    get_entropy(F.softmax(aug_score_avg)),
                    get_entropy(F.softmax(score_avg)),
                    get_entropy(F.softmax(aug_score_local))
                )
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg,
                                                                               labels) + loss_aug_causality

            # optimize avg model, sahre across domains
            loss = loss_id_avg + loss_causality_avg + loss_aug + self.args.lam * loss_aux_avg  # 消融

            # loss = loss_id_avg + loss_causality_avg + self.args.lam * loss_aux_avg #消融

            # loss = loss_id_avg + loss_causality_avg #消融
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                  f'LossID: {loss_id_avg.item():.2f}, LossAux:{float(loss_aux_avg):.2f})')

        return avg_net.state_dict()

    # vanilla aug, mAP=31.9
    def train_free_dacs(self, net, aug_mod, global_epoch, client_id,
                        cls_layer, op_type='sgd'):
        net.train(True)
        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[net, cls_layer, aug_mod], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            cur_mean, cur_var = images.mean(0), images.var(0)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)

            # basic cls loss for local model
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_tri = self.tri_loss(feature, labels)
            loss_ce = self.ce_loss(score, labels)
            loss = loss_ce + loss_tri
            loss_aux, loss_aug = 0, 0

            # aug avg model
            if global_epoch > 0:
                aug_image = aug_mod(norm_image)
                aug_feature_avg = net(aug_image)[0]
                aug_score_avg = cls_layer(aug_feature_avg)
                aug_mean, aug_var = aug_mod.get_mean_var()
                loss_aux = -(F.mse_loss(cur_mean, aug_mean) +
                             F.mse_loss(cur_var, aug_var))
                loss_aug = self.ce_loss(
                    aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels)

            loss = loss + loss_aug + self.args.lam * loss_aux
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f}, LossAux:{float(loss_aux):.2f})')

        return net.state_dict()

    # nofed resnet50, mAP=33
    def train_cls_nofed_sepcls(self, net, global_epoch,
                               cls_layer, op_type='sgd'):
        net.train(True)
        optimizer = self.get_optimizer(
            [net, cls_layer, ], global_epoch, optimizer_type=op_type
        )
        [daset.new_epoch() for daset in self.local_train]
        num_domains = len(self.local_train)

        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            loss = 0
            for client_id in range(num_domains):
                padding = self.padding[client_id]
                (images, _, labels, _, _) = self.local_train[client_id].next()
                images, labels = images.cuda(), labels.cuda() + padding
                feature = net(images)[0]
                score = cls_layer(feature)
                loss_tri = self.tri_loss(feature, labels)
                loss_ce = self.ce_loss(score, labels)

                loss += loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}].'
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f})')
        return net.state_dict()

    # others, distill + scarfold
    def train_fedreid(self, net, avg_net, global_epoch, client_id, cls_layer):
        net.train(True)
        avg_net.train(True)
        optimizer = self.get_optimizer([net, ], global_epoch)
        optim_avg = self.get_optimizer([avg_net, cls_layer], global_epoch)

        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        start_time = time.time()
        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # extract features from avg model and vanilla model
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss = loss_ce + loss_tri
            loss_kl = 0
            # update local model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_epoch > 0:
                feat_avg = avg_net(images)[0]
                score_avg = cls_layer(feat_avg)
                loss_ce = self.ce_loss(score_avg, labels)
                loss_tri = self.tri_loss(feat_avg, labels)
                score = score.detach()  # optim avg only
                loss_kl = (F.softmax(score, 1) * F.log_softmax(score, 1)).sum(1).mean() - \
                          (F.softmax(score, 1) * F.log_softmax(score_avg, 1)).sum(1).mean()
                # update local model
                loss_consist = loss_ce + loss_tri + self.args.temp ** 2 * loss_kl
                optim_avg.zero_grad()
                loss_consist.backward()
                optim_avg.step()
            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                      f'LossTri: {loss_tri.item():.2f}, LossKL: {float(loss_kl):.2f})'
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')
        end_time = time.time()
        time_per_batch = (end_time - start_time) / self.max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))
        prompt_state = {}
        if global_epoch > 0:
            for key, value in avg_net.named_parameters():
                if "prompt_embeddings" in key:  # 新Visual_prompt
                    prompt_state[key] = value
        else:
            for key, value in net.named_parameters():
                if "prompt_embeddings" in key:  # 新Visual_prompt
                    prompt_state[key] = value
        return avg_net.state_dict() if global_epoch > 0 else net.state_dict() , prompt_state

    def train_scar(self, net, avg_net, global_epoch, client_id,cls_layer,
                   client_control, server_control):
        # net 是客户端本地的模型，它带有 'module.' 前缀
        net.train(True)
        avg_net.train(False)

        # 1. 使用标准的SGD优化器
        optimizer = self.get_optimizer(
            [net, cls_layer, ], global_epoch, optimizer_type='sgd'
        )

        # 辅助函数，用于剥离 'module.' 前缀，得到干净的参数名
        def get_clean_name(name):
            return name.replace('module.', '', 1)

        # 2. 保存训练前的初始权重 (x)，用于后续计算
        # 我们保存的 initial_weights 使用的是干净的 key
        initial_weights = {get_clean_name(k): v.clone().cpu() for k, v in net.state_dict().items()}

        self.local_train.new_epoch()
        start_time = time.time()
        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            feature = net(images)[0]
            # 注意：这里需要确认您的 loss 计算是否依赖一个外部的 cls_layer
            # 如果是，需要做相应调整。这里假设 net 已经包含了分类头
            score = cls_layer(feature)  # 假设分类头仍然是外部的

            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss = loss_tri + loss_ce

            optimizer.zero_grad()
            loss.backward()

            # --- 3. Scaffold 核心：手动修正梯度 ---
            with torch.no_grad():
                # 遍历模型中所有带梯度的参数
                for param_name, param in net.named_parameters():
                    if param.grad is None:
                        continue

                    # 获取干净的参数名，用于在控制变量字典中查找
                    clean_name = get_clean_name(param_name)

                    # 将服务器和客户端控制变量移动到与参数相同的设备上
                    c = server_control[clean_name].to(param.device)
                    ci = client_control[clean_name].to(param.device)

                    # 核心公式: g_i = g_i + c - c_i
                    param.grad.data += c - ci

            # 4. 优化器执行一步更新 (此时使用的是修正后的梯度)
                        # --- 新增：梯度裁剪 ---
            # 设置一个合理的裁剪阈值，例如 1.0 或 5.0
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            
            optimizer.step()

            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                      f'LossTri: {loss_tri.item():.2f})'
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')

        # --- 5. 训练结束后，计算新的控制变量和模型更新增量 ---
        with torch.no_grad():
            final_weights = {get_clean_name(k): v.cpu() for k, v in net.state_dict().items()}

            new_client_control = {}
            model_update_delta = {}

            local_lr = optimizer.param_groups[0]['lr']
            K = self.max_iter  # 本地训练的总步数

            # --- 关键修改：只遍历 server_control 中存在的键 ---
            # server_control 定义了哪些参数是需要被管理的。我们只处理这些参数。
            for name in server_control.keys():
                # 确保 initial_weights 和 final_weights 中确实有这个键
                if name not in initial_weights or name not in final_weights:
                    continue

                # 计算模型更新增量: delta_x = x - y_i (初始权重 - 最终权重)
                delta_x = initial_weights[name] - final_weights[name]
                model_update_delta[name] = delta_x

                # 获取 c 和 c_i (它们都在CPU上)
                c = server_control[name]
                ci = client_control[name]

                # 计算新的客户端控制变量: c_i^+ = c_i - c + (1/(K*lr)) * (x - y_i)
                coef = 1 / (K * local_lr)
                new_client_control[name] = ci - c + delta_x * coef

        end_time = time.time()
        time_per_batch = (end_time - start_time) / self.max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))

        prompt_state = {}
        for key, value in net.named_parameters():
            if "prompt_embeddings" in key:
                prompt_state[key] = value
        return final_weights, prompt_state , new_client_control, model_update_delta

    # snr version
    def train_snr(self, net, global_epoch, client_id,
                  fc, fc1, fc2, fc3, op_type='sgd'):
        # preparation
        net.train(True)
        optimizer = self.get_optimizer(
            [net, fc, fc1, fc2, fc3], global_epoch,
            optimizer_type=op_type
        )
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        start_time = time.time()
        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # forward with the original parameters
            features, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = net.module.base(images)

            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))

            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))

            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))

            # Causality loss:
            loss_causality = get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob),
                                           get_entropy(x_1_useless_prob)) + \
                             get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob),
                                           get_entropy(x_2_useless_prob)) + \
                             get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                                 x_3_useful_prob), get_entropy(x_3_useless_prob))
            # common loss
            score = fc(features)
            loss = self.ce_loss(score, labels) + loss_causality

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (Loss: {loss.item():.4f}. '
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')
        prompt_state = {}
        for key, value in net.named_parameters():
            if "prompt_embeddings" in key:
                prompt_state[key] = value
        return net.state_dict() , prompt_state

    # 当前只为局部侧全局模型应用CLIP更新
    def train_dacs_IL_CLIP_stage1(self, net, global_epoch, client_id, optimizer_1stage, scheduler_1stage, scaler):
        net.train(True)
        scheduler_1stage.step(global_epoch)
        self.local_train.new_epoch()
        xent = SupConLoss()
        for batch_idx in range(self.max_iter):
            optimizer_1stage.zero_grad()
            (images, _, labels, _, _) = self.local_train.next()
            images = images.cuda()
            labels = labels.cuda()
            with amp.autocast(enabled=True):
                with torch.no_grad():
                    image_features = net(images, labels, get_image=True)  # 获得图像特征
                text_features = net(label=labels, get_text=True)
            loss_i2t = xent(image_features, text_features, labels, labels)
            loss_t2i = xent(text_features, image_features, labels, labels)
            loss = loss_i2t + loss_t2i
            scaler.scale(loss).backward()
            scaler.step(optimizer_1stage)
            scaler.update()

            if (batch_idx + 1) % 100 == 0:
                print(f'Training Stage-1.'
                      f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs1}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (Loss_i2t: {loss_i2t.item():.2f}, '
                      f'Loss_t2i: {loss_t2i.item():.2f}, lr: {scheduler_1stage._get_lr(global_epoch)[0]:.7f})')

        return net.state_dict()

    def train_dacs_IL_CLIP_stage2(self, net, avg_net, aug_mod, num_classes, text_features, optimizer_2stage_local,
                                  scheduler_2stage_loacl, global_epoch, client_id,
                                  cls_layer, cls_layer_proj, F_news, Labels, op_type='SGD', scaler=None):
        net.train(True)
        avg_net.train(True)
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        memory = copy.deepcopy(self.memory)
        self.local_train.new_epoch()
        scheduler_2stage_loacl.step()
        # optimizer_2stage_local = self.make_optimizer_2stage(
        #     models=[net, ], epoch=global_epoch,
        #     op_type='SGD')
        optimizer_2stage = self.make_optimizer_2stage(
            models=[avg_net, cls_layer, cls_layer_proj, aug_mod, ], epoch=global_epoch,
            op_type=op_type)
        scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, [20, 40], 0.1, 0.01, 10, 'linear')
        scheduler_2stage.step()
        for batch_idx in range(self.max_iter):
            optimizer_2stage_local.zero_grad()
            # 获取训练数据
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # 图像归一化
            b_size = images.shape[0]
            cur_mean, cur_var = images.mean((2, 3)).view(b_size, -1, 1, 1), images.var((2, 3)).view(b_size, -1, 1, 1)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)  # 计算均值和方差进行图像归一化

            with amp.autocast(enabled=True):
                f_bn, feat, image_features = net(x=images, label=labels, cam_label=0,
                                                 view_label=0)  # 客户端本地模型  原始图像   识别损失
                feature_norm = F.normalize(f_bn[0])
                f_bn_avg, feat_avg, image_features_avg = avg_net(x=images, label=labels, cam_label=0,
                                                                 view_label=0)  # 客户端全局模型  原始图像   识别损失
                feature_avg_norm = F.normalize(f_bn_avg[0])
            loss_id_local = memory[client_id](feature_norm, labels).mean()
            scaler.scale(loss_id_local).backward()
            scaler.step(optimizer_2stage_local)
            scaler.update()

            # loss_id = memory[client_id](feature_avg_norm, labels).mean() + self.ce_loss(score_avg, labels) + self.tri_loss(feature_avg, labels)
            loss_id = memory[client_id](feature_avg_norm, labels).mean()
            # loss_i2t = xent(logits,labels)
            with amp.autocast(enabled=True):
                aug_image = aug_mod(norm_image)  # 使用数据增强模块增强图像
                with torch.no_grad():
                    f_bn_avg, _, _ = avg_net(x=aug_image, label=labels, cam_label=0, view_label=0)
                    f_new = F.normalize(f_bn_avg[0])
                    if F_news is not None:
                        F_news[client_id].append(f_new)
                    if Labels is not None:
                        Labels[client_id].append(labels)
                    memory[client_id].module.MomentumUpdate(f_new, labels)

            loss_aux, loss_aug, loss_wd = 0, 0, 0

            # aug avg model 如果不是第一轮全局训练
            if global_epoch >= 0:
                freeze_avg = freeze_model(copy.deepcopy(avg_net))  # 冻结全局模型
                with amp.autocast(enabled=True):
                    feat_bn_avg_freeze, _, _ = freeze_avg(x=aug_image, label=labels, cam_label=0, view_label=0)
                    feat_bn_freeze, _, _ = freeze_avg(x=images, label=labels, cam_label=0, view_label=0)
                    aug_score_avg_freeze = cls_layer(feat_bn_avg_freeze[0])
                    # obtain H(fL(x')), optimizer does not contain net.params(), so we do not need to use 'freeze_model'
                    feat, _, _ = net(x=aug_image, label=labels, cam_label=0, view_label=0)
                    aug_score_local = cls_layer(feat[0])
                    # generate H(fG(x)), use a frozen avg_net
                    score_avg_freeze = cls_layer(feat_bn_freeze[0])  # 全局模型对原始图像的分数
                # au loss,  H(fG(x)) < H(fG(x')) < H(fL(x'))
                loss_aux = get_auth_loss(  # 增强损失
                    get_entropy(F.softmax(aug_score_avg_freeze)),
                    get_entropy(F.softmax(score_avg_freeze)),
                    get_entropy(F.softmax(aug_score_local))
                )

                # aug images to update avg_net 计算增强图像在全局模型的损失来更新
                # with amp.autocast(enabled=True):
                with amp.autocast(enabled=True):
                    feat_avg, _, image_feature_avg = avg_net(x=aug_image, label=labels, cam_label=0, view_label=0)
                    aug_feature_avg = feat_avg[0]
                    aug_score_avg = cls_layer(aug_feature_avg)
                logits = image_feature_avg @ text_features.t()
                loss_aug = xent(logits, labels) + self.tri_loss(aug_feature_avg, labels)
                # loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels)

                # div loss 训练增强模型
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                          F.mse_loss(cur_var, shift_var)

            # optimize avg model, share across domains
            # loss = loss_ce + loss_tri + loss_aug + loss_wd + self.args.lam * loss_aux
            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux + loss_i2t
            loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux
            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux
            optimizer_2stage.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_2stage)
            scaler.update()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
            #       f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
            #       f'LossTri: {loss_tri.item():.2f}, LossAux:{float(loss_aux):.2f})')
            # print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
            #       f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
            #       f'Lossi2t_local: {loss_i2t_local.item():.2f}, LossID: {loss_id.item():.2f}, Lossi2t: {loss_i2t.item():.2f}, '
            #       f'LossAux:{float(loss_aux):.2f})')
            if batch_idx % 100 == 0:
                print(f'Training Stage-2.'
                      f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs2}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                      f'LossID: {loss_id.item():.2f}, LossAux:{float(loss_aux):.2f}, '
                      f'LossAUG: {loss_aug:.2f}), '
                      f'lr_local: {scheduler_2stage_loacl.get_lr()[0]:.7f}, lr: {scheduler_2stage.get_lr()[0]:.7f})')
        # # print ssim
        # ssim_epoch = max(ssim_scores) if len(ssim_scores) else 0
        # is_epoch = inception_score(aug_image.detach()) if global_epoch>0 else 0

        # if global_epoch % 5 == 0:
        #     print(f'Dataset {self.set_name}. SSIM: {ssim_epoch:4.3f}, IS: {is_epoch:4.3f}.')

        return avg_net.module.image_encoder.state_dict()

    def train_dacs_IL_CLIP_stage2_1(self, net, avg_net, aug_mod, num_classes, text_features, optimizer_2stage_local,
                                    scheduler_2stage_loacl, global_epoch, client_id,
                                    cls_layer, cls_layer_proj, F_news, Labels, op_type='sgd', scaler=None):
        net.train(True)
        avg_net.train(True)
        memory = copy.deepcopy(self.memory)
        self.local_train.new_epoch()
        scheduler_2stage_loacl.step()
        # avg optimizer #本地侧全局
        # optimizer_2stage = self.get_optimizer_clip_s2(
        #     nets=[avg_net, cls_layer, cls_layer_proj, aug_mod, ], epoch=global_epoch,
        #     optimizer_type=op_type)
        optimizer_2stage = self.make_optimizer_2stage(
            models=[avg_net, cls_layer, cls_layer_proj, aug_mod, ], epoch=global_epoch,
            op_type=op_type)
        scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, [20, 40], 0.1, 0.01, 10, 'linear')
        scheduler_2stage.step()
        # ssim_scores, is_epoch = [], 0
        for batch_idx in range(self.max_iter):
            optimizer_2stage_local.zero_grad()
            # 获取训练数据
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # 图像归一化
            b_size = images.shape[0]
            cur_mean, cur_var = images.mean((2, 3)).view(b_size, -1, 1, 1), images.var((2, 3)).view(b_size, -1, 1, 1)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)  # 计算均值和方差进行图像归一化

            with amp.autocast(enabled=True):
                f_bn, feat, image_features = net(x=images, label=labels, cam_label=0,
                                                 view_label=0)  # 客户端本地模型  原始图像   识别损失
                feature_norm = F.normalize(f_bn[0])
                f_bn_avg, feat_avg, image_features_avg = avg_net(x=images, label=labels, cam_label=0,
                                                                 view_label=0)  # 客户端全局模型  原始图像   识别损失
                feature_avg_norm = F.normalize(f_bn_avg[0])
            # logits = image_features_avg @ text_features.t()
            loss_id_local = memory[client_id](feature_norm, labels).mean()
            scaler.scale(loss_id_local).backward()
            scaler.step(optimizer_2stage_local)
            scaler.update()

            # loss_id = memory[client_id](feature_avg_norm, labels).mean() + self.ce_loss(score_avg, labels) + self.tri_loss(feature_avg, labels)
            loss_id = memory[client_id](feature_avg_norm, labels).mean()
            # loss_i2t = xent(logits,labels)
            with amp.autocast(enabled=True):
                aug_image = aug_mod(norm_image)  # 使用数据增强模块增强图像

            # 保存每个图像，假设批量大小为B
            # aug_image_to_save = (aug_image + 1) / 2
            # image_to_save = images
            # aug_image_to_save = aug_image

            # aug_save_dir = '/mnt/data/rcy/DACS_IL-mainwork/image/aug_img'
            # save_dir = '/mnt/data/rcy/DACS_IL-mainwork/image/img'
            # for i in range(aug_image_to_save.size(0)):
            #     # 生成文件名，例如 'augmented_0.png', 'augmented_1.png', ...
            #     filename = f'Client_{client_id}_{global_epoch}_{labels[i]}_{i}.png'
            #     aug_filename = f'augmented_Client_{global_epoch}_{client_id}_{labels[i]}_{i}.png'
            #     save_path = os.path.join(save_dir, filename)
            #     save_path_aug = os.path.join(aug_save_dir, aug_filename)

            #     # 保存单个图像
            #     save_image(image_to_save[i], save_path)
            #     save_image(aug_image_to_save[i], save_path_aug)
            # print('finished!')
            # del image_to_save,aug_image_to_save

            with torch.no_grad():
                # f_new = avg_net(images,style='store_true')[1] #提取风格化特征
                with amp.autocast(enabled=True):
                    f_bn_avg, feat_avg, image_features_avg = avg_net(x=aug_image, label=labels, cam_label=0,
                                                                     view_label=0)
                    f_new = F.normalize(f_bn_avg[0])
                    # f_new = avg_net(aug_image,style=False)[1]
                    # if bad_iamges is not None:
                    #     bad_iamges.append(aug_image)
                    if F_news is not None:
                        F_news[client_id].append(f_new)
                    if Labels is not None:
                        Labels[client_id].append(labels)
                    memory[client_id].module.MomentumUpdate(f_new, labels)

            loss_aux, loss_aug, loss_wd = 0, 0, 0

            # aug avg model 如果不是第一轮全局训练
            if global_epoch > 0:
                # generate freezed global model to detach grad, training=False version
                freeze_avg = freeze_model(copy.deepcopy(avg_net))  # 冻结全局模型
                # for name, param in freeze_avg.named_parameters():
                #     print(f"Layer name: {name}, Gradient required: {param.requires_grad}")

                freeze_avg.train()
                # freeze_avg = copy.deepcopy(avg_net)
                # transformed image
                # obtain H(fG(x')), use a frozen avg_net to avoid updating avg_net model
                with amp.autocast(enabled=True):
                    # with torch.no_grad():
                    feat_bn_avg_freeze, _, _ = freeze_avg(x=aug_image, label=labels, cam_label=0, view_label=0)
                    feat_bn_freeze, _, _ = freeze_avg(x=images, label=labels, cam_label=0, view_label=0)
                    aug_score_avg_freeze = cls_layer(feat_bn_avg_freeze[0])
                    # obtain H(fL(x')), optimizer does not contain net.params(), so we do not need to use 'freeze_model'
                    feat, _, _ = net(x=aug_image, label=labels, cam_label=0, view_label=0)
                    aug_score_local = cls_layer(feat[0])
                    # generate H(fG(x)), use a frozen avg_net
                    score_avg_freeze = cls_layer(feat_bn_freeze[0])  # 全局模型对原始图像的分数
                # au loss,  H(fG(x)) < H(fG(x')) < H(fL(x'))
                loss_aux = get_auth_loss(  # 增强损失
                    get_entropy(F.softmax(aug_score_avg_freeze)),
                    get_entropy(F.softmax(score_avg_freeze)),
                    get_entropy(F.softmax(aug_score_local))
                )

                # aug images to update avg_net 计算增强图像在全局模型的损失来更新
                # with amp.autocast(enabled=True):
                feat_avg, _, _ = avg_net(x=aug_image, label=labels, cam_label=0, view_label=0)
                aug_feature_avg = feat_avg[0]
                aug_score_avg = cls_layer(aug_feature_avg)
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels)

                # div loss 训练增强模型
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                          F.mse_loss(cur_var, shift_var)

            # optimize avg model, share across domains
            # loss = loss_ce + loss_tri + loss_aug + loss_wd + self.args.lam * loss_aux
            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux + loss_i2t
            loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux
            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux
            optimizer_2stage.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_2stage)
            scaler.update()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
            #       f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
            #       f'LossTri: {loss_tri.item():.2f}, LossAux:{float(loss_aux):.2f})')
            # print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
            #       f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
            #       f'Lossi2t_local: {loss_i2t_local.item():.2f}, LossID: {loss_id.item():.2f}, Lossi2t: {loss_i2t.item():.2f}, '
            #       f'LossAux:{float(loss_aux):.2f})')
            if batch_idx % 100 == 0:
                print(f'Training Stage-2.'
                      f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs2}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                      f'LossID: {loss_id.item():.2f}, LossAux:{float(loss_aux):.2f}, '
                      f'lr_local: {scheduler_2stage_loacl.get_lr()[0]:.7f}, lr: {scheduler_2stage.get_lr()[0]:.7f})')
        # # print ssim
        # ssim_epoch = max(ssim_scores) if len(ssim_scores) else 0
        # is_epoch = inception_score(aug_image.detach()) if global_epoch>0 else 0

        # if global_epoch % 5 == 0:
        #     print(f'Dataset {self.set_name}. SSIM: {ssim_epoch:4.3f}, IS: {is_epoch:4.3f}.')

        return avg_net.module.image_encoder.state_dict()

    def train_dacs_IL_CLIP_stage2_2(self, net, avg_net, aug_mod, num_classes, text_features, optimizer_2stage_local,
                                    scheduler_2stage_loacl, global_epoch, client_id,
                                    cls_layer, cls_layer_proj, F_news, Labels, op_type='SGD', scaler=None):
        net.train(True)
        avg_net.train(True)
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        memory = copy.deepcopy(self.memory)
        self.local_train.new_epoch()
        scheduler_2stage_loacl.step()
        optimizer_2stage = self.make_optimizer_2stage(
            models=[avg_net, cls_layer, cls_layer_proj, aug_mod, ], epoch=global_epoch,
            op_type=op_type)
        scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, [20, 40], 0.1, 0.01, 10, 'linear')
        scheduler_2stage.step()
        for batch_idx in range(self.max_iter):
            # 获取训练数据
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # 图像归一化
            b_size = images.shape[0]
            cur_mean, cur_var = images.mean((2, 3)).view(b_size, -1, 1, 1), images.var((2, 3)).view(b_size, -1, 1, 1)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)  # 计算均值和方差进行图像归一化
            # 优化客户端本地模型
            with amp.autocast(enabled=True):
                f_bn, _, image_features = net(x=images, label=labels, cam_label=0, view_label=0)  # 客户端本地模型  原始图像   识别损失
                f_bn_norm = F.normalize(f_bn[0])
                loss_id_local = memory[client_id](f_bn_norm, labels).mean()

            # optimizer_2stage_local.zero_grad()
            # loss_id_local.backward()
            # optimizer_2stage_local.step()
            # scaler.scale(loss_id_local).backward()
            # scaler.step(optimizer_2stage_local)
            # scaler.update()

            with amp.autocast(enabled=True):
                loss_aux, loss_aug, loss_wd = 0, 0, 0
                aug_image = aug_mod(norm_image)  # 使用数据增强模块增强图像
                f_bn_avg, _, _ = avg_net(x=images, label=labels, cam_label=0, view_label=0)  # 客户端全局模型  原始图像   识别损失
                f_bn_avg_norm = F.normalize(f_bn_avg[0])
                loss_id = memory[client_id](f_bn_avg_norm, labels).mean()
                if global_epoch > 2:
                    freeze_avg = freeze_model(copy.deepcopy(avg_net))  # 冻结全局模型
                    # freeze_avg.train()
                    aug_avg_freeze, _, _ = freeze_avg(x=aug_image, label=labels, cam_label=0, view_label=0)
                    avg_freeze, _, _ = freeze_avg(x=images, label=labels, cam_label=0, view_label=0)
                    aug_score_avg_freeze = aug_avg_freeze
                    aug_score_avg_freeze[0] = cls_layer(aug_score_avg_freeze[0])
                    aug_score_avg_freeze[1] = cls_layer_proj(aug_score_avg_freeze[1])
                    # generate H(fG(x)), use a frozen avg_net
                    score_avg_freeze = avg_freeze
                    score_avg_freeze[0] = cls_layer(avg_freeze[0])  # 全局模型对原始图像的分数
                    score_avg_freeze[1] = cls_layer_proj(avg_freeze[1])
                    # obtain H(fL(x')), optimizer does not contain net.params(), so we do not need to use 'freeze_model'
                    feat, _, _ = net(x=aug_image, label=labels, cam_label=0, view_label=0)
                    aug_score_local = feat
                    aug_score_local[0] = cls_layer(aug_score_local[0])
                    aug_score_local[1] = cls_layer_proj(aug_score_local[1])

                    # 计算增强图像在全局模型的损失来更新
                    feat_avg, _, image_feature_avg = avg_net(x=aug_image, label=labels, cam_label=0, view_label=0)
                    # aug_feature_avg = feat_avg[0]
                    # aug_score_avg = cls_layer(aug_feature_avg)
                    logits = image_feature_avg @ text_features.t()

                    # if global_epoch > 2:
                    loss_aux = get_auth_loss(  # 增强损失
                        get_entropy(F.softmax(aug_score_avg_freeze[0])),
                        get_entropy(F.softmax(score_avg_freeze[0])),
                        get_entropy(F.softmax(aug_score_local[0]))
                    ) + get_auth_loss(  # 增强损失
                        get_entropy(F.softmax(aug_score_avg_freeze[1])),
                        get_entropy(F.softmax(score_avg_freeze[1])),
                        get_entropy(F.softmax(aug_score_local[1]))
                    )
                    if isinstance(feat_avg, list):
                        tri_loss = [self.tri_loss(feat, labels) for feat in feat_avg[0:]]
                        tri_loss = sum(tri_loss)
                    else:
                        tri_loss = self.tri_loss(feat_avg, labels)
                    I2T_loss = xent(logits, labels)
                    loss_aug = I2T_loss + tri_loss

                    shift_mean, shift_var = aug_mod.get_mean_var()
                    loss_wd = -F.mse_loss(shift_mean, cur_mean) - F.mse_loss(cur_var, shift_var)
            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux + loss_i2t
            loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux

            optimizer_2stage.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_2stage)
            scaler.update()

            with torch.no_grad():
                f_bn_avg, _, _ = avg_net(x=aug_image, label=labels, cam_label=0, view_label=0)
                f_new = F.normalize(f_bn_avg[0])
                if F_news is not None:
                    F_news[client_id].append(f_new)
                if Labels is not None:
                    Labels[client_id].append(labels)
                memory[client_id].module.MomentumUpdate(f_new, labels)

            if batch_idx % 100 == 0:
                print(f'Training Stage-2.'
                      f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs2}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                      f'LossID: {loss_id.item():.2f}, LossAux:{float(loss_aux):.2f}, '
                      f'LossAUG: {loss_aug:.2f}), '
                      f'lr_local: {scheduler_2stage_loacl.get_lr()[0]:.7f}, lr: {scheduler_2stage.get_lr()[0]:.7f})')

        return avg_net.module.image_encoder.state_dict()

    def train_dacs_IL_CLIP_stage2_3(self, net, avg_net, aug_mod, num_classes, text_features, optimizer_2stage_local,
                                    scheduler_2stage_loacl, global_epoch, client_id,
                                    cls_layer, cls_layer_proj, F_news, Labels, op_type='SGD', scaler=None):
        net.train(True)
        avg_net.train(True)
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        triplet = Tri_clip(0.3)
        memory = copy.deepcopy(self.memory)
        self.local_train.new_epoch()
        scheduler_2stage_loacl.step()
        optimizer_2stage = self.make_optimizer_2stage(
            models=[avg_net.module, cls_layer, cls_layer_proj, aug_mod, ], epoch=global_epoch,
            op_type=op_type)
        scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, [20, 40], 0.1, 0.01, 10, 'linear')
        scheduler_2stage.step(global_epoch)
        start_time = time.time()
        for batch_idx in range(self.max_iter):
            eval_start_time = time.time()
            optimizer_2stage_local.zero_grad()
            # 获取训练数据
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # 图像归一化
            b_size = images.shape[0]
            cur_mean, cur_var = images.mean((2, 3)).view(b_size, -1, 1, 1), images.var((2, 3)).view(b_size, -1, 1, 1)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)  # 计算均值和方差进行图像归一化

            with amp.autocast(enabled=True):
                f_bn, feat, image_features = net(x=images, label=labels)  # 客户端本地模型  原始图像   识别损失
                feature_norm = F.normalize(f_bn[0])
                f_bn_avg, feat_avg, image_features_avg = avg_net(x=images, label=labels)  # 客户端全局模型  原始图像   识别损失
                feature_avg_norm = F.normalize(f_bn_avg[0])

            loss_id_local = memory[client_id](feature_norm, labels).mean()
            scaler.scale(loss_id_local).backward()
            scaler.step(optimizer_2stage_local)
            scaler.update()

            loss_id = memory[client_id](feature_avg_norm, labels).mean()
            with amp.autocast(enabled=True):
                aug_image = aug_mod(norm_image)  # 使用数据增强模块增强图像
                with torch.no_grad():
                    f_bn_avg, _, _ = avg_net(x=aug_image, label=labels, cam_label=None, view_label=None)
                    f_new = F.normalize(f_bn_avg[0])
                    if F_news is not None:
                        F_news[client_id].append(f_new)
                    if Labels is not None:
                        Labels[client_id].append(labels)
                    memory[client_id].module.MomentumUpdate(f_new, labels)

            loss_aux, loss_aug, loss_wd = 0, 0, 0
            loss_i2t = 0
            TRI_LOSS, ID_LOSS = 0, 0

            # aug avg model 如果不是第一轮全局训练
            if global_epoch > 0:
                freeze_avg = freeze_model(copy.deepcopy(avg_net))  # 冻结全局模型
                with amp.autocast(enabled=True):
                    feat_bn_avg_freeze, _, _ = freeze_avg(x=aug_image, label=labels, cam_label=None, view_label=None)
                    feat_bn_freeze, _, _ = freeze_avg(x=images, label=labels, cam_label=None, view_label=None)
                    aug_score_avg_freeze = cls_layer(feat_bn_avg_freeze[0])
                    feat, _, _ = net(x=aug_image, label=labels, cam_label=None, view_label=None)
                    aug_score_local = cls_layer(feat[0])
                    score_avg_freeze = cls_layer(feat_bn_freeze[0])  # 全局模型对原始图像的分数
                loss_aux = get_auth_loss(  # 增强损失
                    get_entropy(F.softmax(aug_score_avg_freeze)),
                    get_entropy(F.softmax(score_avg_freeze)),
                    get_entropy(F.softmax(aug_score_local))
                )

                # 计算增强图像在全局模型的损失来更新
                with amp.autocast(enabled=True):
                    feat_avg, f, image_feature_avg = avg_net(x=aug_image, label=labels, cam_label=None, view_label=None)
                    aug_score_avg = feat_avg
                    aug_score_avg[0] = cls_layer(feat_avg[0])
                    aug_score_avg[1] = cls_layer_proj(feat_avg[1])
                logits = image_feature_avg @ text_features.t()
                if isinstance(f, list):
                    TRI_LOSS = [triplet(feat, labels)[0] for feat in f[0:]]
                    TRI_LOSS = sum(TRI_LOSS)
                else:
                    TRI_LOSS = triplet(f, labels)[0]
                if isinstance(aug_score_avg, list):
                    ID_LOSS = [xent(score, labels) for score in aug_score_avg[0:]]
                    ID_LOSS = sum(ID_LOSS)
                else:
                    ID_LOSS = xent(aug_score_avg, labels)

                # loss_aug = xent(logits,labels) + self.tri_loss(aug_feature_avg, labels)
                # loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels)
                # loss_aug = self.tri_loss(aug_feature_avg, labels)
                # loss_aug = xent(logits,labels)
                # loss_aug = xent(logits,labels) + ID_LOSS + TRI_LOSS
                loss_aug = xent(logits, labels) + ID_LOSS
                loss_i2t = xent(logits, labels)
                # div loss 训练增强模型
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                          F.mse_loss(cur_var, shift_var)

            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux + loss_i2t
            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux + TRI_LOSS
            loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux
            optimizer_2stage.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_2stage)
            scaler.update()
            if batch_idx % self.args.print_freq == 0:
                eval_end_time = time.time()
                print(f'Training Stage-2.'
                      f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs2}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                      f'LossID: {loss_id.item():.2f}, LossAux:{float(loss_aux):.2f}, '
                      f'LossAUG: {loss_aug:.2f}, LossWD: {loss_wd:.2f}, LossTri_clip: {TRI_LOSS:.2f}, '
                      # f'lr_local: {scheduler_2stage_loacl.get_lr()[0]:.7f}, lr: {scheduler_2stage.get_lr()[0]:.7f})')
                      f'lossi2t: {loss_i2t:.2f}, '
                      f'lr_local: {scheduler_2stage_loacl.get_lr()[0]:.7f}, '
                      f'time per batch: {(eval_end_time - eval_start_time) / self.args.print_freq:.2f} s)')
        end_time = time.time()
        time_per_batch = (end_time - start_time) / self.max_iter
        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
              .format(global_epoch, time_per_batch, self.args.batch_size / time_per_batch))
        prompt_state = {}
        for key, value in avg_net.named_parameters():
            if "prompt_embeddings" in key:  # 新Visual_prompt
                prompt_state[key] = value

        print(prompt_state.keys())
        return avg_net.module.image_encoder.state_dict(), prompt_state

    def CLIP_stage0(self, net, global_epoch, client_id, optimizer_0stage, scheduler_0stage, scaler, classifier,
                    classifier_proj, num_classes):
        net.train(True)
        scheduler_0stage.step()
        self.local_train.new_epoch()
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        triplet = TripletLoss()
        for batch_idx in range(self.max_iter):
            optimizer_0stage.zero_grad()
            (images, _, labels, _, _) = self.local_train.next()
            images = images.cuda()
            labels = labels.cuda()
            target_cam = None
            target_view = None
            with amp.autocast(enabled=True):
                feat_bn, feat, image_features = net(x=images, label=labels, cam_label=target_cam,
                                                    view_label=target_view)
                score = feat_bn
                score[0] = classifier(feat_bn[0])
                score[1] = classifier_proj(feat_bn[1])
            if isinstance(score, list):
                ID_LOSS = [xent(scor, labels) for scor in score[0:]]
                ID_LOSS = sum(ID_LOSS)
            else:
                ID_LOSS = xent(score, labels)
            if isinstance(feat, list):
                TRI_LOSS = [triplet(feats, labels).item() for feats in feat[0:]]
                TRI_LOSS = sum(TRI_LOSS)
            else:
                # TRI_LOSS = triplet(feat, labels)[0]
                TRI_LOSS = triplet(feat, labels).item()
            loss = 1.0 * TRI_LOSS + 1.0 * ID_LOSS
            scaler.scale(loss).backward()
            scaler.step(optimizer_0stage)
            scaler.update()

            print(f'Training Stage-0.'
                  f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs0}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (Loss_ID: {ID_LOSS.item():.2f}, '
                  f'Loss_TRI: {TRI_LOSS:.2f}, lr: {scheduler_0stage.get_lr()[0]:.7f})')

        return net.state_dict()

    def CLIP_stage1_1(self, net, global_epoch, client_id, optimizer_1stage, scheduler_1stage, scaler):
        net.train(True)
        scheduler_1stage.step(global_epoch)
        self.local_train.new_epoch()
        xent = SupConLoss()
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images = images.cuda()
            labels = labels.cuda()
            with amp.autocast(enabled=True):
                with torch.no_grad():
                    image_features = net(images, labels, get_image=True)  # 获得图像特征
                # del Labels, Image_features
                text_features = net(label=labels, get_text=True)
            loss_i2t = xent(image_features, text_features, labels, labels)
            loss_t2i = xent(text_features, image_features, labels, labels)
            loss = loss_i2t + loss_t2i
            optimizer_1stage.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_1stage)
            scaler.update()
            if batch_idx % 100 == 0:
                print(f'Training Stage-1.'
                      f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs1}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (Loss_i2t: {loss_i2t.item():.2f}, '
                      f'Loss_t2i: {loss_t2i.item():.2f}, lr: {scheduler_1stage._get_lr(global_epoch)[0]:.7f})')

        return net.state_dict()

    def CLIP_stage1_2(self, net, global_epoch, client_id, optimizer_1stage, scheduler_1stage, scaler):
        net.train(True)
        scheduler_1stage.step(global_epoch)
        self.local_train.new_epoch()
        xent = SupConLoss()
        for batch_idx in range(self.max_iter):
            optimizer_1stage.zero_grad()
            (images, _, labels, _, _) = self.local_train.next()
            images = images.cuda()
            labels = labels.cuda()
            with amp.autocast(enabled=True):
                with torch.no_grad():
                    image_features = net(images, labels, get_image=True)  # 获得图像特征
                text_features = net(label=labels, get_text=True)
            loss_i2t = xent(image_features, text_features, labels, labels)
            loss_t2i = xent(text_features, image_features, labels, labels)
            loss = loss_i2t + loss_t2i
            scaler.scale(loss).backward()
            scaler.step(optimizer_1stage)
            scaler.update()

            if batch_idx % 100 == 0:
                print(f'Training Stage-1.'
                      f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs1}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (Loss_i2t: {loss_i2t.item():.2f}, '
                      f'Loss_t2i: {loss_t2i.item():.2f}, lr: {scheduler_1stage._get_lr(global_epoch)[0]:.7f})')

        return net.state_dict()

    def CLIP_stage2(self, avg_net, num_classes, total_text_features, optimizer_2stage, optimizer_center_2stage,
                    loss_func, scheduler_2stage,
                    global_epoch, client_id, scaler, cls_layer, cls_layer_proj):
        avg_net.train(True)
        self.local_train.new_epoch()
        scheduler_2stage.step()
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            optimizer_2stage.zero_grad()
            images = images.cuda()
            labels = labels.cuda()
            target_cam = None
            target_view = None
            with amp.autocast(enabled=True):
                feat_bn, feat, image_features = avg_net(x=images, label=labels, cam_label=target_cam,
                                                        view_label=target_view)
                score = feat_bn
                score[0] = cls_layer(feat_bn[0])
                score[1] = cls_layer_proj(feat_bn[1])
                logits = image_features @ total_text_features.t()
                loss = loss_func(score, feat, labels, target_cam, logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer_2stage)
            scaler.update()
            if batch_idx % 100 == 0:
                print(f'Training Stage-2.'
                      f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs2}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (Loss: {loss.item():.2f}, lr: {scheduler_2stage.get_lr()[0]:.7f})')

        return avg_net.module.image_encoder.state_dict()

    def train_dacs_CLIP_stage2(self, net, avg_net, aug_mod, num_classes, text_features, optimizer_2stage_local,
                               scheduler_2stage_loacl, global_epoch, client_id,
                               cls_layer, cls_layer_proj, F_news, Labels, op_type='SGD', scaler=None):
        net.train(True)
        avg_net.train(True)
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        triplet = Tri_clip(0.3)
        memory = copy.deepcopy(self.memory)
        self.local_train.new_epoch()
        scheduler_2stage_loacl.step()
        optimizer_2stage = self.make_optimizer_2stage(
            models=[avg_net, cls_layer, cls_layer_proj, aug_mod, ], epoch=global_epoch,
            op_type=op_type)
        scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, [20, 40], 0.1, 0.01, 10, 'linear')
        scheduler_2stage.step(global_epoch)
        for batch_idx in range(self.max_iter):
            optimizer_2stage_local.zero_grad()
            # 获取训练数据
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # 图像归一化
            b_size = images.shape[0]
            cur_mean, cur_var = images.mean((2, 3)).view(b_size, -1, 1, 1), images.var((2, 3)).view(b_size, -1, 1, 1)
            norm_image = (images - cur_mean).div(cur_var.sqrt() + 1e-8)  # 计算均值和方差进行图像归一化

            with amp.autocast(enabled=True):
                f_bn, feat, image_features_local = net(x=images, label=labels, cam_label=0, view_label=0)
                score = f_bn
                score[0] = cls_layer(score[0])
                score[1] = cls_layer_proj(score[1])
                f_bn_avg, feat_avg, image_features_avg = avg_net(x=images, label=labels, cam_label=0,
                                                                 view_label=0)  # 客户端全局模型  原始图像   识别损失
                score_avg = f_bn_avg
                score_avg[0] = cls_layer(score_avg[0])
                score_avg[1] = cls_layer_proj(score_avg[1])

            if isinstance(score, list):
                ID_LOSS_LOCAL = [xent(s, labels) for s in score[0:]]
                ID_LOSS_LOCAL = sum(ID_LOSS_LOCAL)
            else:
                ID_LOSS_LOCAL = xent(score, labels)

            if isinstance(feat, list):
                TRI_LOSS_LOACL = [triplet(f, labels)[0] for f in feat[0:]]
                TRI_LOSS_LOACL = sum(TRI_LOSS_LOACL)
            else:
                TRI_LOSS_LOACL = triplet(feat, labels)[0]

            logits_loacl = image_features_local @ text_features.t()
            I2T_LOSS_LOCAL = xent(logits_loacl, labels)

            loss_loacl = ID_LOSS_LOCAL + TRI_LOSS_LOACL + I2T_LOSS_LOCAL
            scaler.scale(loss_loacl).backward()
            scaler.step(optimizer_2stage_local)
            scaler.update()

            if isinstance(score_avg, list):
                ID_LOSS_AVG = [xent(score, labels) for score in score_avg[0:]]
                ID_LOSS_AVG = sum(ID_LOSS_AVG)
            else:
                ID_LOSS_AVG = xent(score_avg, labels)

            if isinstance(feat_avg, list):
                TRI_LOSS_AVG = [triplet(feat, labels)[0] for feat in feat_avg[0:]]
                TRI_LOSS_AVG = sum(TRI_LOSS_AVG)
            else:
                TRI_LOSS_AVG = triplet(feat_avg, labels)[0]

            logits_avg = image_features_avg @ text_features.t()
            I2T_LOSS_AVG = xent(logits_avg, labels)

            loss_avg = I2T_LOSS_AVG + TRI_LOSS_AVG + ID_LOSS_AVG

            with amp.autocast(enabled=True):
                aug_image = aug_mod(norm_image)  # 使用数据增强模块增强图像

            loss_aux, loss_aug, loss_wd, TRI_LOSS_AUG_AVG, ID_LOSS_AUG_AVG = 0, 0, 0, 0, 0

            # aug avg model 如果不是第一轮全局训练
            if global_epoch > 0:
                freeze_avg = freeze_model(copy.deepcopy(avg_net))  # 冻结全局模型
                with amp.autocast(enabled=True):
                    feat_bn_avg_freeze, _, _ = freeze_avg(x=aug_image, label=labels, cam_label=0, view_label=0)
                    feat_bn_freeze, _, _ = freeze_avg(x=images, label=labels, cam_label=0, view_label=0)
                    aug_score_avg_freeze = cls_layer(feat_bn_avg_freeze[0])
                    feat, _, _ = net(x=aug_image, label=labels, cam_label=0, view_label=0)
                    aug_score_local = cls_layer(feat[0])
                    score_avg_freeze = cls_layer(feat_bn_freeze[0])  # 全局模型对原始图像的分数
                loss_aux = get_auth_loss(  # 增强损失
                    get_entropy(F.softmax(aug_score_avg_freeze)),
                    get_entropy(F.softmax(score_avg_freeze)),
                    get_entropy(F.softmax(aug_score_local))
                )

                # 计算增强图像在全局模型的损失来更新
                with amp.autocast(enabled=True):
                    feat_avg, f, image_feature_avg = avg_net(x=aug_image, label=labels, cam_label=0, view_label=0)
                    aug_score_avg = feat_avg
                    aug_score_avg[0] = cls_layer(feat_avg[0])
                    aug_score_avg[1] = cls_layer_proj(feat_avg[1])
                if isinstance(f, list):
                    TRI_LOSS_AUG_AVG = [triplet(feat, labels)[0] for feat in f[0:]]
                    TRI_LOSS_AUG_AVG = sum(TRI_LOSS_AUG_AVG)
                else:
                    TRI_LOSS_AUG_AVG = triplet(f, labels)[0]
                if isinstance(aug_score_avg, list):
                    ID_LOSS_AUG_AVG = [xent(score, labels) for score in aug_score_avg[0:]]
                    ID_LOSS_AUG_AVG = sum(ID_LOSS_AUG_AVG)
                else:
                    ID_LOSS_AUG_AVG = xent(aug_score_avg, labels)
                logits_aug_avg = image_feature_avg @ text_features.t()
                I2T_LOSS_AUG_AVG = xent(logits_aug_avg, labels)

                loss_aug_avg = I2T_LOSS_AUG_AVG + ID_LOSS_AUG_AVG + TRI_LOSS_AUG_AVG

                # div loss 训练增强模型
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                          F.mse_loss(cur_var, shift_var)

            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux + loss_i2t
            # loss = loss_id + loss_aug + loss_wd + self.args.lam * loss_aux + TRI_LOSS
            loss = loss_avg + loss_aug_avg + loss_wd + self.args.lam * loss_aux
            optimizer_2stage.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_2stage)
            scaler.update()
            if batch_idx % 100 == 0:
                print(f'Training Stage-2.'
                      f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs2}]. Net Client: {client_id}. '
                      f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (Loss_local: {loss_loacl.item():.2f}, '
                      f'Loss_avg: {loss_avg.item():.2f}, Loss_aug_avg: {loss_aug_avg.item():.2f}, LossAux:{float(loss_aux):.2f}, '
                      f'LossWD: {loss_wd:.2f}'
                      # f'lr_local: {scheduler_2stage_loacl.get_lr()[0]:.7f}, lr: {scheduler_2stage.get_lr()[0]:.7f})')
                      f'lr_local: {scheduler_2stage_loacl.get_lr()[0]:.7f})')

        return avg_net.module.image_encoder.state_dict()
