import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
from .data import transforms as T
from .data.preprocessor import Preprocessor
from torch.utils.data import DataLoader, ConcatDataset
import torch
from reid import datasets
import os.path as osp
from .data import IterLoader, Preprocessor
from .data.preprocessor import ModernPreprocessor
from .data.sampler import RandomMultipleGallerySampler
import matplotlib as mpl
from torchvision.models.inception import inception_v3
import torch.nn.functional as F
from scipy.stats import entropy
mpl.use("Agg")
sns.set()

def inception_score(imgs, cuda=True, resize=False):
    N = len(imgs)
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor
    # Load inception model
    inception_model = inception_v3(
        pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    imgs = imgs.type(dtype)
    preds = get_pred(imgs)
    py = np.mean(preds, axis=0)
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(entropy(pyx, py))

    return np.exp(np.mean(scores))


class ScaffoldOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, unified_model, server_controls, client_controls, closure=None):
        # 遍历整个统一模型的所有命名参数
        # 这样可以确保参数 p 和它的名字 name 是正确对应的
        for name, p in unified_model.named_parameters():
            if p.grad is None:
                continue

            # 为当前参数 p 找到正确的学习率
            lr = self.defaults['lr']
            for group in self.param_groups:
                if any(p is param_in_group for param_in_group in group['params']):
                    lr = group['lr']
                    break

            p_device = p.device

            # --- 通过名字安全地查找控制变量 ---
            # .get(name) 如果找不到键会返回 None，这里我们假设一定能找到
            c = server_controls[name].to(p_device)
            ci = client_controls[name].to(p_device)

            # --- 执行Scaffold更新 ---
            # 修正梯度: dp = g_i + c - c_i
            dp = p.grad.data + c - ci

            # 更新参数: p = p - lr * dp
            p.data.add_(dp, alpha=-lr)

def freeze_model(cur_model):
    # cur_model.eval()
    for param in cur_model.parameters():
        param.requires_grad = False
    return cur_model


# 假设 T 和 Preprocessor 已经定义好了
def get_test_loader_tsne(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    final_testset = []
    root_dir = None

    if isinstance(dataset, list):
        print("Input is a list of datasets. Activating combined loading mode...")
        pid_offset = 0
        
        for domain_id, dset in enumerate(dataset):
            # 合并query和gallery
            domain_data = dset.query + dset.gallery
            
            # 获取数据集中唯一ID的数量
            num_pids_in_dataset = len(set(p for _, p, _ in domain_data))

            print(f"  - Processing domain {domain_id} ({dset.__class__.__name__}):")
            print(f"    - Applying PID offset: {pid_offset}")

            # 遍历样本，创建新的、包含正确信息的元组
            for img_path, original_pid, original_camid in domain_data:
                new_pid = original_pid + pid_offset
                
                # “偷梁换柱”：将domain_id放入元组的第三个位置
                # 这是为了让下游的camid变量接收到域信息
                final_testset.append((img_path, new_pid, domain_id))

            pid_offset += num_pids_in_dataset
            
        # CUHK03可能需要一个 root 目录
        if dataset[0].__class__.__name__ == 'CUHK03':
             root_dir = dataset[0].images_dir

    else: # 处理单个数据集的情况
        print("Input is a single dataset. Using standard loading mode...")
        if dataset.__class__.__name__ == 'CUHK03':
            root_dir = dataset.images_dir
        
        # 你的原始逻辑是合并query和gallery
        data_to_process = dataset.query + dataset.gallery
        # 注意：这里我们不转换成set，因为set会打乱顺序并丢失重复项
        # 我们假设Preprocessor的输入就是一个列表
        
        # 对于单个数据集，我们没有domain_id，但下游代码需要5个返回值
        # 所以我们把原始的camid传下去
        for img_path, pid, camid in data_to_process:
            final_testset.append((img_path, pid, camid))
            
    # 核心修改：使用我们新的、可靠的 ModernPreprocessor
    test_loader = DataLoader(
        ModernPreprocessor(final_testset, root=root_dir, transform=test_transformer),
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)
        
    return test_loader


def get_gallery_loader(dataset, height, width, batch_size, workers, galleryset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], #归一化参数
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([ #数据预处理
        T.Resize((height, width), interpolation=3),
        T.ToTensor(), normalizer
    ])
    root_dir = dataset.images_dir if dataset.__class__.__name__ == 'CUHK03' else None #如果数据集是 CUHK03，则需要设置 root_dir 为图像文件夹路径。

    if galleryset is None: #如果没有指定 galleryset，则将查询图像和图库图像合并，并去除重复图像
        galleryset = list(set(dataset.gallery))
    gallery_loader = DataLoader( #创建数据加载器
        Preprocessor(galleryset, root=root_dir, transform=test_transformer),
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    return gallery_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], #归一化参数
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([ #数据预处理
        T.Resize((height, width), interpolation=3),
        T.ToTensor(), normalizer
    ])
    root_dir = dataset.images_dir if dataset.__class__.__name__ == 'CUHK03' else None #如果数据集是 CUHK03，则需要设置 root_dir 为图像文件夹路径。

    if isinstance(dataset.query[0], list): #如果数据集的查询图像和图库图像是列表形式，则将它们合并为一个列表
        testset = dataset.query + dataset.gallery
    elif testset is None: #如果没有指定 testset，则将查询图像和图库图像合并，并去除重复图像
        testset = list(set(dataset.query) | set(dataset.gallery))
    test_loader = DataLoader( #创建数据加载器
        Preprocessor(testset, root=root_dir, transform=test_transformer),
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    return test_loader

def get_train_loaders(dataset_lists, args, is_shuffle=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((args.height, args.width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10), T.RandomCrop((args.height, args.width)),
        T.ToTensor(), normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    train_loaders = []
    for dataset in dataset_lists:
        if is_shuffle:
            temp_loader = DataLoader(
                Preprocessor(dataset.train, root=None, transform=transformer),
                batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True
            )
        else:
            temp_loader = IterLoader(DataLoader(
                Preprocessor(dataset.train, transform=transformer, root=None),
                batch_size=args.batch_size, shuffle=False, drop_last=True,
                sampler=RandomMultipleGallerySampler(
                    dataset.train, args.num_instances),
                pin_memory=True, num_workers=args.workers
            ), length=None)
        train_loaders.append(temp_loader)

    return train_loaders

def get_train_loader(args, dataset, height, width, batch_size, workers,
                         num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer])
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    sampler = RandomMultipleGallerySampler(train_set, num_instances)
    rmgs_flag = False
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=rmgs_flag, pin_memory=True, drop_last=True), length=None)
    return train_loader

def get_entropy(p_softmax):
    mask = p_softmax.ge(1e-6)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return (entropy / float(p_softmax.size(0)))


def get_auth_loss(ent_aug_global, ent_ori_global, ent_aug_local):
    # HG(x'), HG(x), HL(x'); should be HG(x) < HG(x') < HL(x')
    ranking_loss = torch.nn.SoftMarginLoss()
    y = torch.ones_like(ent_aug_global)
    # HG(x) < HG(x'), HG(x') < HL(x')
    return ranking_loss(ent_aug_global - ent_ori_global, y) +\
        ranking_loss(ent_aug_local - ent_aug_global, y)


def get_data(args, set_names=None, sub_split=1):
    data_dir = args.data_dir
    if set_names is not None:
        dataset = []
        for name in set_names:
            root = osp.join(data_dir)
            if sub_split==1:
                dataset.append(datasets.create(name, root))
            else:
                for idx in range(sub_split):
                    cur_set = datasets.create(name, root)
                    cur_set.split_clients(sub_split, idx)
                    dataset.append(cur_set)
    else:
        name = args.test_dataset
        root = osp.join(data_dir)
        dataset = datasets.create(name, root)
    return dataset

def get_aug_data(args):
    set_names = 'unreal_v1.1,unreal_v2.1,unreal_v3.1,unreal_v4.1,unreal_v1.2,unreal_v2.2,unreal_v3.2,unreal_v4.2,unreal_v1.3,unreal_v2.3,unreal_v3.3,unreal_v4.3'
    return datasets.create(
        name='unreal', root=args.data_dir,
        dataset=set_names.split(','), data=args.data_dir
    )

plt.rcParams['figure.facecolor'] = 'white'

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import torch
import os.path as osp 

def plotTSNE(features, domains, save_path, epoch):
    func = TSNE()

    domain_to_color = {
        0: '#AC97F5',  # 淡紫色
        1: '#099C9A',  # 青色
        2: '#FFA460',  # 天蓝色
        3: '#FFA460',  # 浅橙色
        # ... 添加更多域标签和对应的颜色
    }
    colors = [domain_to_color[domain] for domain in domains]

    def map_label(val):
        if val == 0:
            return 'C2'
        elif val == 1:
            return 'C3'
        elif val == 2:
            return 'MS'
        else:
            return 'MS'

    embFeat = func.fit_transform(features)
    embFeat = pd.DataFrame(embFeat, columns=["x", "y"])
    embFeat["Domain"] = pd.Series(domains).apply(map_label)

    plt.figure(facecolor='white')
    # 使用Seaborn的scatterplot根据Domain列上色
    fig = sns.scatterplot(x=embFeat["x"], y=embFeat["y"],
                          hue=embFeat["Domain"], palette=domain_to_color.values(),legend=False)
    fig.xaxis.set_ticklabels([])
    fig.yaxis.set_ticklabels([])
    fig.xaxis.set_label_text(None)
    fig.yaxis.set_label_text(None)
    fig.set_facecolor('white')
    
    # 移除plt.scatter调用，以避免覆盖Seaborn的颜色设置
    # plt.scatter(features[:, 0], features[:, 1], c=colors)  # 这行代码不再需要

    import os
    if not osp.exists(osp.dirname(save_path)):
        os.makedirs(osp.dirname(save_path))
    plt.savefig(f"{osp.dirname(save_path)}/t_sne_{epoch}epoch")

    
    # torch.save(embFeat, osp.join(osp.dirname(save_path), f"tsne_{epoch}.pth"))
    plt.close()

# 假设features和domains已经定义
# plotTSNE(features, domains, 'path_to_save_image.png', epoch)


# 假设features和domains已经定义
# plotTSNE(features, domains, 'path_to_save_image.png', epoch)

# 假设features和domains已经定义
# plotTSNE(features, domains, 'path_to_save_image.png', epoch)



# def plotTSNE(features, domains, save_path, epoch):
#     func = TSNE()

#     domain_to_color = {
#     0: '#F49FAC',  # 粉红色
#     1: '#099C9A',  # 青色
#     2: '#7AB2D3',  # 天蓝色
#     3: '#FFA460',  # 浅橙色
#     # ... 添加更多域标签和对应的颜色
# }
#     colors = [domain_to_color[domain] for domain in domains]

#     def map_label(val):
#         if val == 0:
#             return 'C2'
#         elif val == 1:
#             return 'C3'
#         elif val == 2:
#             return 'C3-Trans'
#         else:
#             return 'MS'

#     embFeat = func.fit_transform(features)
#     embFeat = pd.DataFrame(embFeat, columns=["x", "y"])
#     embFeat["Domain"] = pd.Series(domains).apply(map_label)

#     pl.figure(facecolor='white')
#     fig = sns.scatterplot(x=embFeat["x"], y=embFeat["y"],
#                           hue=embFeat["Domain"], palette="tab10")
#     fig.xaxis.set_ticklabels([])
#     fig.yaxis.set_ticklabels([])
#     fig.xaxis.set_label_text(None)
#     fig.yaxis.set_label_text(None)
#     fig.set_facecolor('white')
#     plt.scatter(features[:, 0], features[:, 1], c=colors)
#     pl.savefig(save_path)
#     torch.save(embFeat, osp.join(osp.dirname(save_path), f"tsne_{epoch}.pth"))
#     pl.close()


# instance norm mix, ref: https://github.com/amazon-science/crossnorm-selfnorm
def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def instance_norm_mix(content_feat, style_feat):
    """replace content statistics with style statistics"""
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_ins_mean_std(style_feat)
    content_mean, content_std = calc_ins_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def cn_rand_bbox(size, beta, bbx_thres):
    """sample a bounding box for cropping."""
    W = size[2]
    H = size[3]
    while True:
        ratio = np.random.beta(beta, beta)
        cut_rat = np.sqrt(ratio)
        cut_w = np.int64(W * cut_rat)
        cut_h = np.int64(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        if ratio > bbx_thres:
            break

    return bbx1, bby1, bbx2, bby2


def cn_op_2ins_space_chan(x, crop='neither', beta=1, bbx_thres=0.1, lam=None, chan=False):
    """2-instance crossnorm with cropping."""
    assert crop in ['neither', 'style', 'content', 'both']
    ins_idxs = torch.randperm(x.size()[0]).to(x.device)

    if crop in ['style', 'both']:
        bbx3, bby3, bbx4, bby4 = cn_rand_bbox(
            x.size(), beta=beta, bbx_thres=bbx_thres)
        x2 = x[ins_idxs, :, bbx3:bbx4, bby3:bby4]
    else:
        x2 = x[ins_idxs]

    if chan:
        chan_idxs = torch.randperm(x.size()[1]).to(x.device)
        x2 = x2[:, chan_idxs, :, :]

    if crop in ['content', 'both']:
        x_aug = torch.zeros_like(x)
        bbx1, bby1, bbx2, bby2 = cn_rand_bbox(
            x.size(), beta=beta, bbx_thres=bbx_thres)
        x_aug[:, :, bbx1:bbx2, bby1:bby2] = instance_norm_mix(content_feat=x[:, :, bbx1:bbx2, bby1:bby2],
                                                              style_feat=x2)
        mask = torch.ones_like(x, requires_grad=False)
        mask[:, :, bbx1:bbx2, bby1:bby2] = 0.
        x_aug = x * mask + x_aug
    else:
        x_aug = instance_norm_mix(content_feat=x, style_feat=x2)

    if lam is not None:
        x = x * lam + x_aug * (1-lam)
    else:
        x = x_aug
    return x

from torch.utils.data import Dataset, Subset
from PIL import Image
from torchvision import transforms
import os.path as osp
class RuntimeImageDataset(Dataset):
    """
    一个通用的Dataset类，接收一个包含(路径, pid, camid)的数据列表，
    并在 __getitem__ 中实时加载和转换图像。
    """
    def __init__(self, dataset_list, transform, root=None):
        self.dataset_list = dataset_list
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        # 从列表中获取元数据
        data_tuple = self.dataset_list[index]
        img_path, pid, camid = data_tuple[0], data_tuple[1], data_tuple[2]
        
        # 构建完整路径
        fpath = img_path
        if self.root is not None:
            fpath = osp.join(self.root, img_path)
            
        # 实时加载和转换图像
        try:
            img = Image.open(fpath).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"\nError loading image: {fpath}. Returning a black tensor. Error: {e}")
            img = torch.zeros((3, 256, 128)) # 假设您的图像尺寸

        # 返回下游代码期望的格式
        # 注意：这里我们返回 img, pid, camid，以匹配您现有的extract_features_tsne函数
        return img, pid, camid