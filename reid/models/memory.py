import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
import numpy as np

class MC(autograd.Function):

    @staticmethod #静态方法，无需实例化即可调用
    def forward(ctx, inputs, indexes, features, momentum): #输入特征，索引，特征，动量
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        return grad_inputs, None, None, None


def mc(inputs, indexes, features, momentum=0.5):
    return MC.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class MemoryClassifier(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(MemoryClassifier, self).__init__()
        self.num_features = num_features #初始化的特征
        self.num_samples = num_samples #样本数量
        self.momentum = momentum #更新动量
        self.temp = temp #调整相似度分布的平滑度

        self.register_buffer('features', torch.zeros(num_samples, num_features)) #注册一个持久缓冲区，记录特征
        self.register_buffer('labels', torch.zeros(num_samples).long())#注册一个持久缓冲区，记录标签

    def MomentumUpdate(self, inputs, indexes): #用于更新内存库
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x #以动量方式更新
            self.features[y] = self.features[y] / self.features[y].norm() #对更新后的特征向量进行归一化，使其具有单位范数

    def forward(self, inputs, indexes):
        # inputs = torch.cat(inputs,dim=0)
        # print(inputs)
        sim = mc(inputs, indexes, self.features, self.momentum) ## B * C 计算每个输入特征与内存中所有特征的相似度
        sim = sim / self.temp #缩放相似值
        loss = F.cross_entropy(sim, indexes)
        return loss



