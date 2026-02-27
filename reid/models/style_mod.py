import torch
import torch.nn as nn


class AugMod(nn.Module):
    def __init__(self, num_features=3, width=128, height=256):
        super(AugMod, self).__init__()
        #两个可学习参数，分别学习均值和标准差
        self.shift_std = nn.Parameter(torch.empty(num_features, height, width)) #学习方差
        self.shift_mean = nn.Parameter(torch.zeros(num_features, height, width)) #学习均值
        self.num_features = num_features 
        # init
        nn.init.normal_(self.shift_std, 1, 0.1) #初始化为均值为1，标准化为0.1
        nn.init.normal_(self.shift_mean, 0, 0.1) #初始化为均值为0，标准化为0.1
        
    def get_mean_var(self):
        return self.shift_mean, self.shift_std**2

    def forward(self, x):
        # x is bn normed
        # 对图像进行变换
        return self.shift_std.to(x.device) * x + self.shift_mean.to(x.device)