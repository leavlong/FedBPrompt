from __future__ import absolute_import

import random
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.distributions import Normal, Uniform
import torchvision

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


class UBS(nn.Module):

    def __init__(self, p=1.0, rho=3.0, eps=1e-6):
        super().__init__()
        self.p = p #p表示对图像风格化的程度
        self.rho = rho #控制扰动的超参数
        self.eps = eps #防止数值不稳定的极小值

    def __repr__(self): #用于查看参数设置
        return f'UBS(rho={self.rho}, p={self.p})'

    def forward(self, x):
        if not self.training: #测试/评估模式-不进行风格化
            return x

        if random.random() > self.p: #根据概率决定是否进行风格化，随机数在0到1之间
            return x

        B = x.size(0) #获取批次大小
        mu = x.mean(dim=[2, 3], keepdim=True) #计算每个通道的均值
        var = x.var(dim=[2, 3], keepdim=True) #计算每个通道的方差
        sig = (var + self.eps).sqrt() #得到标准差 
        mu, sig = mu.detach(), sig.detach() #避免影响梯度更新
        x_normed = (x - mu) / sig #得到归一化后的特征图

        mu_1 = x.mean(dim=[2, 3], keepdim=True)
        std_1 = x.std(dim=[2, 3], keepdim=True)

        #批次维度求平均，得到每个通道的均值的均值，以及方差的均值 即方差的分布情况
        mu_mu = mu_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1) 
        mu_std = mu_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        #批次维度求平均，得到每个通道的方差的均值，以及方差的方差 即方差的分布情况
        std_mu = std_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        std_std = std_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        #限制二者最小值，防止数值不稳定
        mu_std.data.clamp_(min=self.eps)
        std_std.data.clamp_(min=self.eps)

        Distri_mu = Uniform(mu_mu - self.rho * mu_std, mu_mu + self.rho * mu_std)#使用均匀分布（采样）生成新的均值
        Distri_std = Uniform(std_mu - self.rho * std_std, std_mu + self.rho * std_std)#使用均匀分布（采样）生成新的方差

        mu_b = Distri_mu.sample([B, ])#采样
        sig_b = Distri_std.sample([B, ])#采样
        mu_b = mu_b.unsqueeze(2).unsqueeze(2)#形状扩展为【B，1，1】
        sig_b = sig_b.unsqueeze(2).unsqueeze(2)
        mu_b, sig_b = mu_b.detach(), sig_b.detach()#计算图中分离出来

        return x_normed * sig_b + mu_b #获得全新的风格化图像

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.style_layers = ['layer1']
        if self.style_layers:
            self.style = UBS()
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        if depth==50:
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
        
        #ResNet模型主干
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.feat_bn0 = nn.BatchNorm1d(2048)
        # self.feat_bn0.bias.requires_grad_(False)
        self.gap = nn.AdaptiveAvgPool2d(1)

        #根据用户指定的参数配置ResNet模块的输出，用于图像提取
        if not self.cut_at_pooling: #是否使用全局平均池化层  使用
            self.num_features = num_features
            print("num_features:",self.num_features)
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0 #用于指示ResNet模块是否包含用于提取特征向量的线性层
            self.num_classes = num_classes
            out_planes = resnet.fc.in_features
            # Append new layers
            if self.has_embedding: #是否需要添加特征提取层  #无
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features, affine=False)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')#kaiming初始化
                init.constant_(self.feat.bias, 0)#常数初始化
            else:#有
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features, affine=False)
            if self.dropout > 0: #无
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0: #有
                self.classifier = nn.Linear(
                    self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        if not pretrained:
            self.reset_params()

    def forward(self, x,style = False):
        # test
        # self.norm = norm
        if self.training is False:
            x = self.gap(self.base(x))
            x = x.view(x.shape[0], -1)
            if self.has_embedding:#无
                bn_x = self.feat_bn(self.feat(x))
            else:#有
                bn_x = self.feat_bn(x)
            if self.num_classes > 0:#无
                return self.classifier(bn_x)
            return bn_x

        # train

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if self.style is not None and style and 'layer1' in self.style_layers:
            x = self.style(x)
        x = self.layer2(x)

        if self.style is not None and style and 'layer2' in self.style_layers:
            x = self.style(x)
        x = self.layer3(x)

        if self.style is not None and style and 'layer3' in self.style_layers:
            x = self.style(x)
        x = self.layer4(x)

        x = self.gap(x)#池化
        x = x.view(x.size(0), -1) 

        if self.cut_at_pooling: #无
            return x
        
        if self.has_embedding: #无
            bn_x = self.feat_bn(self.feat(x))
        else: #有
            bn_x = self.feat_bn(x)

        if self.norm: #有
            bn_x_norm = F.normalize(bn_x)
        elif self.has_embedding:  #无
            bn_x_norm = F.relu(bn_x)

        if self.dropout > 0: #无
            bn_x_norm = self.drop(bn_x_norm)

        if self.num_classes > 0: #无
            prob = self.classifier(bn_x)
        else: #有
            return [bn_x,bn_x_norm]#返回包含单个元素的列表

        # return prob, x
        return prob, bn_x_norm

    #重置模型参数
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

def resnet50(**kwargs): #**kwargs允许用户传递任意数量的关键字参数
    return ResNet(50, **kwargs)

def resnet18(**kwargs):
    return ResNet(18, **kwargs)

class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""
    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels //
                             reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates,
                             kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x) # pool5
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x
