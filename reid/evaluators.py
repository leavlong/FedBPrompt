from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def compute_accuracy(predictions, labels):
    if np.ndim(labels) == 2:
        y_true = np.argmax(labels, axis=-1)
    else:
        y_true = labels
    accuracy = accuracy_score(
        y_true=y_true, y_pred=np.argmax(predictions, axis=-1))
    return accuracy

def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)  # pool5 for eva 评估模式下返回值只有一个
    # print('shape of outputs:',outputs.shape)#([64, 2048])
    if isinstance(outputs, list) and len(outputs) > 1:
        outputs = outputs[-1][0]
    outputs = outputs.data.cpu()
    # print('shape of outputs:',outputs.shape)#([64, 2048])
    return outputs


def extract_cnn_feature_clip(model, inputs):
    inputs = to_torch(inputs).cuda()#转为tensor张量并放到gpu上
    outputs = model(inputs)  # pool5 for eva #获得结果
    avp = nn.AdaptiveAvgPool2d(1)
    bn = nn.BatchNorm1d(outputs[1].size()[1], affine=False).cuda()
    # print('shape of outputs[1]:',outputs[1].shape)#([64, 2048, 14, 14])
    output = bn(avp(outputs[1]).view(outputs[1].shape[0],-1)).data.cpu()
    # print('shape of output:',output.shape)#([64, 2048])
    return output

def extract_vit_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)  # pool5 for eva 评估模式下返回值只有一个
    # print('shape of outputs:',outputs.shape)#([64, 2048])
    if isinstance(outputs, list) and len(outputs) > 1:
        outputs = outputs[0][: , 0]
    outputs = outputs.data.cpu()
    # print('shape of outputs:',outputs.shape)#([64, 2048])
    return outputs

def extract_cnn_feature_clip_2m(model, inputs):
    inputs = to_torch(inputs).cuda()#转为tensor张量并放到gpu上
    outputs = model(inputs)  # pool5 for eva #获得结果
    avp = nn.AdaptiveAvgPool2d(1)
    bn = nn.BatchNorm1d(outputs[1].size()[1], affine=False).cuda()
    # print('shape of outputs[1]:',outputs[1].shape)#([64, 2048, 14, 14])
    output1 = bn(avp(outputs[1]).view(outputs[1].shape[0],-1)).data.cpu()
    output2 = bn(avp(outputs[1]).view(outputs[1].shape[0],-1)).data.cpu()
    # print('shape of output:',output.shape)#([64, 2048])
    return output1,output2

def extract_cnn_text_feature(model, labels):
    text_feature = model(label = labels,get_text = True)  # pool5 for eva #获得结果
    return text_feature


def extract_features(model, data_loader, print_freq=50,test = False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    #存储特征和标签
    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs = extract_cnn_feature(model, imgs)#特征提取
            # print('device of outputs:',outputs.device)
            for fname, output, pid in zip(fnames, outputs, pids):#记录存储特征和对应的标签
                if test == True:
                    output = output.cuda()
                    pid = pid.cuda()
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0: #输出处理进度信息
                print('Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg))
    return features, labels

def extract_features_tsne(model, data_loader, device='cuda', max_iter=3):
    """
    一个标准的、高效的特征提取函数。
    它接收一个现代化的PyTorch DataLoader，并提取所有样本的特征。

    Args:
        model (nn.Module): 你的模型 (TransReID)。
        data_loader (DataLoader): 一个能够产出 (图像张量, pid, domain_id, ...) 的加载器。
        device (str): 'cuda' 或 'cpu'。

    Returns:
        tuple: (all_features, all_pids, all_domain_ids)
               - all_features: (N, D) numpy array
               - all_pids: (N,) numpy array, 全局唯一的行人ID
               - all_domain_ids: (N,) numpy array, 域ID (来自camid通道)
    """
    model.to(device)
    model.eval()  # 确保模型处于评估模式

    all_features = []
    all_pids = []
    all_domain_ids = []
    
    i = 0
    with torch.no_grad():
        # tqdm可以很好地包装DataLoader，提供进度条
        for (images, fnames, pids, _, _) in tqdm(data_loader, desc="Extracting Features for t-SNE"):
            if i == max_iter:
                break
            images = images.to(device)

            # 核心：通过模型获取特征
            # 注意：确保你的模型在eval模式下，返回的是最终的特征向量
            features = model(images)

            # 将数据移动到CPU并转换为numpy
            all_features.append(features.cpu().numpy())
            all_pids.append(pids.numpy())
            # all_domain_ids.append(domain_ids.numpy())
            i += 1
    # 将所有批次的列表拼接成一个大的numpy数组
    all_features = np.concatenate(all_features, axis=0)
    all_pids = np.concatenate(all_pids, axis=0)
    # all_domain_ids = np.concatenate(all_domain_ids, axis=0)

    return all_features, all_pids

def extract_features_clip(model, data_loader, print_freq=50,test = False , backbone='ViT-B-16'):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    #存储特征和标签
    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            if backbone == 'ViT-B-16':
                outputs = extract_vit_feature(model, imgs)
            else:
                outputs = extract_cnn_feature_clip(model, imgs)#特征提取
            # print('device of outputs:',outputs.device)
            for fname, output, pid in zip(fnames, outputs, pids):#记录存储特征和对应的标签
                if test == True:
                    output = output.cuda()
                    pid = pid.cuda()
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0: #输出处理进度信息
                print('Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg))
    return features, labels

def extract_features_clip_2m(model, data_loader, print_freq=50,test = False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    #存储特征和标签
    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature_clip_2m(model, imgs)#特征提取
            # print('device of outputs:',outputs.device)
            for fname, output, pid in zip(fnames, outputs, pids):#记录存储特征和对应的标签
                if test == True:
                    output = output.cuda()
                    pid = pid.cuda()
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0: #输出处理进度信息
                print('Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg))
    return features, labels

def extract_text_features(args,model, data_loader, print_freq=50):
    batch = args.batch_size
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    text_features = []

    #存储特征和标签
    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            text_feature = extract_cnn_text_feature(model, pids)#特征提取
            text_features.append(text_feature)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0: #输出处理进度信息
                print('Extract Text Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg))
        text_features = torch.cat(text_features, 0).cuda()

    return text_features


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None: #无查询和候选
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m #获取所有特征之间的自距离矩阵

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)#提取查询集特征
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)#提取候选集特征
    #重塑为矩阵形式
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1) 
    y = y.view(n, -1)
    #计算查询集和候选集之间的距离矩阵
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    # return dist_m, x.numpy(), y.numpy()
    return dist_m, x, y


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if not cmc_flag:
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True), }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k - 1]))
    return mAP, cmc_scores['market1501']


class Evaluator(object):
    def __init__(self, model=None):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)#提取特征
        # features, _ = extract_features(self.model.module.image_encoder, data_loader)#提取特征
        distmat, _, _ = pairwise_distance(features, query, gallery)#计算距离矩阵
        results = evaluate_all(distmat, query=query, gallery=gallery,
                               cmc_flag=cmc_flag)
        return results

    def evaluate_pacs(self, test_loader, set_name):
        self.model.eval()
        y_pred, y_labels = [], []
        for (images, pids) in test_loader:
            images, pids = images.cuda(), pids.cuda()
            with torch.no_grad():
                y_pred.append(self.model(images))
                y_labels.append(pids)
        y_pred, y_labels = torch.cat(y_pred, 0).cpu().numpy(), torch.cat(y_labels).cpu().numpy()
        
        accuracy = compute_accuracy(predictions=y_pred, labels=y_labels)
        print(f'----------accuracy test on {set_name.upper()}----------: {100*accuracy:4.2f} %')
        return accuracy
    