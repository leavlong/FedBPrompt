import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from functools import reduce
from operator import mul

from torch.nn import Dropout
from torch.nn.modules.utils import _pair


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False,
        qk_scale=None, attn_drop=0., proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        # dim=768
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.head_dim = head_dim

    def forward(self, x):
        B, N, C = x.shape  # B--32, N--211, C--768
        # (32,261,2304) --> (32,261,3,12,64) --> (3,32,12,261,64)
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        # same layer, 3 outputs
        q, k, v = qkv[0], qkv[1], qkv[2]
        # (32*12*261*261), self.scale--1/8
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # attn_drop -- 0
        # scores attened on v-features
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # same dim
        x = self.proj_drop(x)
        return x

class part_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # add mask to q k v
        mask = mask.to(q.device.type)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.bool(), torch.tensor(-1e3, dtype=torch.float16)) # mask
        attn = attn.softmax(dim=-1)
        attn = torch.mul(attn, mask) ###
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x , attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class part_Attention_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., H=16, W=8,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = part_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask = None):
        # 保存原始输入 x 以用于残差连接
        residual = x 
        
        # 应用注意力模块
        attn_output, attn_weights = self.attn(self.norm1(x), mask)
        
        # 正确的第一个残差连接
        x = residual + self.drop_path(attn_output)
        
        # 正确的第二个残差连接 (MLP部分)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weights

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(
                    1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    # last feature if backbone outputs list/tuple of features
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(
            stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # use conv to extract patch features
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # [64, 8, 768], the stacked patch embeddings
        x = x.flatten(2).transpose(1, 2)
        return x


class TransReID(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 local_feature=False, args=None , prompt_cfg = None):
        super().__init__()
        self.num_classes = num_classes
        self.args = args
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.prompt_cfg = prompt_cfg
        self.local_feature = local_feature  # True for Market

        # 初始化提示配置
        if prompt_cfg is None:
            self.prompt_cfg = type('Args', (), {
                'LOCATION': 'prepend',
                'INITIATION': 'random',
                'NUM_TOKENS': 50,
                'DROPOUT': 0.0,
                'PROJECT': -1,
                'DEEP': True,
                'NUM_DEEP_LAYERS': None,
                'DEEP_SHARED': False
            })

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size,
                in_chans=in_chans, embed_dim=embed_dim
            )
        else:
            # this branch
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size,
                stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim
            )

        num_patches = self.patch_embed.num_patches
        self.depth = depth

        self._init_prompt_tokens(num_patches, embed_dim)

        # cls token for global info
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos embed
        pos_dim = num_patches + 1 + (self.prompt_cfg.NUM_TOKENS if self.prompt_cfg.LOCATION == 'prepend' else 0)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            part_Attention_Block( # 注意这里的变化
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.fc = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.prompt_group_config = {'upper': 10, 'middle':10, 'lower': 10, 'shared': 20}
        assert sum(self.prompt_group_config.values()) == self.prompt_cfg.NUM_TOKENS, "Prompt group sizes must sum to total prompt tokens."
        
        # 创建并存储掩码，这样就不用每次前向传播都重新计算
        self.prompt_attn_mask = self._create_prompt_attention_mask()

    def _create_prompt_attention_mask(self):
        """
        创建控制prompt注意力的静态掩码。这是实现分区注意力的核心。
        """
        H, W = self.patch_embed.num_y, self.patch_embed.num_x
        num_patches = H * W
        num_prompts = self.prompt_cfg.NUM_TOKENS
        N = 1 + num_prompts + num_patches # 总序列长度

        # 创建一个全False的掩码
        mask = torch.zeros(N, N, dtype=torch.bool)

        # 定义各个部分的索引范围
        cls_idx = 0
        prompt_start_idx = 1
        patch_start_idx = 1 + num_prompts

        # 1. 允许所有token关注自己
        mask.fill_diagonal_(True)

        # 2. CLS token 和 Shared Prompts 可以关注所有token，所有token也可以关注它们
        shared_start_idx = prompt_start_idx + self.prompt_group_config['upper'] + self.prompt_group_config['middle'] + self.prompt_group_config['lower']
        shared_end_idx = shared_start_idx + self.prompt_group_config['shared']
        
        mask[cls_idx, :] = True
        mask[:, cls_idx] = True
        mask[shared_start_idx:shared_end_idx, :] = True
        mask[:, shared_start_idx:shared_end_idx] = True

        # 3. 允许所有patch之间互相关注
        mask[patch_start_idx:, patch_start_idx:] = True

        # 4. 【关键】为每个局部prompt组设置其关注的patch区域
        # 上身区域 (前 1/2 高度)
        upper_patches_end = H // 2 * W
        upper_prompt_start = prompt_start_idx
        upper_prompt_end = upper_prompt_start + self.prompt_group_config['upper']
        mask[upper_prompt_start:upper_prompt_end, patch_start_idx : patch_start_idx + upper_patches_end] = True

        # 中身区域 (1/4 到 3/4 高度，重叠区域)
        middle_patches_start = H // 4 * W
        middle_patches_end = H * 3 // 4 * W
        middle_prompt_start = upper_prompt_end
        middle_prompt_end = middle_prompt_start + self.prompt_group_config['middle']
        mask[middle_prompt_start:middle_prompt_end, patch_start_idx + middle_patches_start : patch_start_idx + middle_patches_end] = True
        
        # 下身区域 (后 1/2 高度)
        lower_patches_start = H // 2 * W
        lower_prompt_start = middle_prompt_end
        lower_prompt_end = lower_prompt_start + self.prompt_group_config['lower']
        mask[lower_prompt_start:lower_prompt_end, patch_start_idx + lower_patches_start :] = True
        
        # 5. (可选) 允许所有prompt之间互相关注
        mask[prompt_start_idx:patch_start_idx, prompt_start_idx:patch_start_idx] = True

        return mask.unsqueeze(0).unsqueeze(0) # 扩展维度以匹配多头注意力的格式 (B, H, N, N)

    def _init_prompt_tokens(self, num_patches, embed_dim):
        """初始化提示token"""
        num_tokens = self.prompt_cfg.NUM_TOKENS
        self.num_tokens = num_tokens

        # 提示dropout
        self.prompt_dropout = Dropout(self.prompt_cfg.DROPOUT)

        # 提示投影
        if self.prompt_cfg.PROJECT > -1:
            prompt_dim = self.prompt_cfg.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, embed_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = embed_dim
            self.prompt_proj = nn.Identity()

        # 随机初始化提示
        if self.prompt_cfg.INITIATION == "random":
            patch_size = _pair(num_patches)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))

            # 浅层提示
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            # 深层提示
            if self.prompt_cfg.DEEP:
                total_d_layer = self.depth - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        """将提示token与图像patch融合"""
        B = x.shape[0]
        x = torch.cat((
            x[:, :1, :],  # CLS token
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]  # 图像patch
        ), dim=1)
        return x

    def forward_deep_prompt(self, x):
        B = x.shape[0]
        # 将掩码移动到正确的设备
        attn_mask = self.prompt_attn_mask.to(x.device)

        for i in range(self.depth):
            if i == 0:
                # 【核心修改 3】: 传递掩码
                x = self.blocks[i](x, mask=attn_mask)
            else:
                if i <= len(self.deep_prompt_embeddings):
                    deep_prompt = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))

                    # 重新拼接token序列
                    x = torch.cat((
                        x[:, :1, :],
                        deep_prompt,
                        # 注意：这里要跳过上一层的prompt，所以是 1 + num_tokens
                        x[:, 1 + self.num_tokens:, :]
                    ), dim=1)
                if i < self.depth - 1:
                    x , _ = self.blocks[i](x, mask=attn_mask)
                else:
                    x , last_layer_attention = self.blocks[i](x, mask=attn_mask)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def set_training_mode(self, mode="vpt"):
        """
        设置模型的训练模式:
        - 'vpt': 冻结backbone, 只训练prompt和分类头。
        - 'full': 训练所有参数 (传统的全量微调)。
        """
        if mode.lower() == "vpt":
            print("设置训练模式: VPT (只训练 prompts 和分类头)")
            # 冻结所有参数
            for param in self.parameters():
                param.requires_grad = False

            # 解冻 prompt 相关参数
            self.prompt_embeddings.requires_grad = True
            if isinstance(self.prompt_proj, nn.Linear):
                for param in self.prompt_proj.parameters():
                    param.requires_grad = True
            if self.prompt_cfg.DEEP:
                self.deep_prompt_embeddings.requires_grad = True

            # 解冻分类头
            for param in self.fc.parameters():
                param.requires_grad = True

        elif mode.lower() == "full":
            print("设置训练模式: Full (全量微调)")
            for param in self.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"不支持的训练模式: {mode}")


    # 在 class TransReID_with_Part_Prompt(nn.Module) 内部：

# 删除你原来的 forward 和 forward_deep_prompt 两个方法，
# 用下面这一个 forward 方法替换它们。

    def forward(self, x, **kwargs):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.incorporate_prompt(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        last_layer_attn = None
        attn_mask = self.prompt_attn_mask.to(x.device)

        if self.prompt_cfg.DEEP:
            # --- 处理 Deep Prompt 的情况 ---
            for i in range(self.depth):
                # 注入深层 prompt (除了第一层)
                if i > 0 and i <= len(self.deep_prompt_embeddings):
                    deep_prompt = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))
                    x = torch.cat((
                        x[:, :1, :],
                        deep_prompt,
                        x[:, 1 + self.num_tokens:, :]
                    ), dim=1)
                
                # 判断是否为最后一层
                if i < self.depth - 1:
                    # 对于中间层，解包元组，但忽略注意力权重
                    x, _ = self.blocks[i](x, mask=attn_mask)
                else:
                    # 对于最后一层，同时捕获 x 和注意力权重
                    x, last_layer_attn = self.blocks[i](x, mask=attn_mask)
        else:
            # --- 处理非 Deep Prompt 的情况 ---
            for i, blk in enumerate(self.blocks):
                if i < self.depth - 1:
                    # 中间层
                    x, _ = blk(x, mask=attn_mask)
                else:
                    # 最后一层
                    x, last_layer_attn = blk(x, mask=attn_mask)

        x = self.norm(x)
        
            # 训练模式：只返回特征
        return x[:, 0]

    def load_param(self, model_path, index_num=None):
        param_dict = torch.load(model_path, map_location='cpu')
        # 处理联邦学习参数结构
        if 'client_models' in param_dict and index_num is not None:
            param_dict = param_dict['client_models'][index_num]
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        # 获取提示token数量（兼容无prompt_cfg的情况）
        num_prompts = getattr(self, 'num_tokens', 0) if hasattr(self, 'prompt_cfg') else 0

        for k, v in param_dict.items():
            if 'module' in k:
                k = k.replace('module.', '')
            if 'head' in k or 'dist' in k or 'prompt' in k:
                continue  # 跳过分类头、dist参数和提示参数

            # 特殊处理patch embedding权重
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)

            # 特殊处理位置编码
            elif k == 'pos_embed':
                # 确保patch数量计算正确
                if not hasattr(self.patch_embed, 'num_y'):
                    self.patch_embed.num_y = self.patch_embed.num_x = int(math.sqrt(self.patch_embed.num_patches))

                v = self.resize_pos_embed(
                    v,
                    self.pos_embed,
                    num_prompts=getattr(self, 'num_tokens', 0),
                    strict_size=True  # 开启严格尺寸检查
                )
                self.pos_embed.data.copy_(v)

                # 验证形状匹配
                expected_shape = self.pos_embed.shape
                if v.shape != expected_shape:
                    print(f"Warning: pos_embed shape mismatch. Expected {expected_shape}, got {v.shape}")
                    continue
                print("  - 'pos_embed' 移植完成。")

            # 参数加载
            try:
                if k in self.state_dict():
                    self.state_dict()[k].copy_(v)
                else:
                    print(f'Skip unexpected key: {k}')
            except Exception as e:
                print(f'Failed to copy {k}: {str(e)}')
                print(f'Expected shape: {self.state_dict()[k].shape}, got {v.shape}')


    def resize_pos_embed(self, posemb, posemb_new, num_prompts=0, strict_size=False):
        """
        参数说明：
        posemb: 原始位置编码 [1, 1+N, D]
        posemb_new: 目标张量（仅用于获取设备信息）
        num_prompts: 提示token数量
        strict_size: 是否强制严格匹配尺寸
        """
        # 计算原始网格尺寸
        cls_pos = posemb[:, :1]  # [1,1,D]
        img_pos = posemb[:, 1:]  # [1,N,D]
        gs_old = int(math.sqrt(img_pos.shape[1]))

        # 获取目标网格尺寸（确保整数）
        if hasattr(self, 'patch_embed'):
            gs_new = (self.patch_embed.num_y, self.patch_embed.num_x)
        else:
            gs_new = (int(math.sqrt(posemb_new.shape[1] - 1 - num_prompts)),) * 2

        # 插值处理
        img_pos = img_pos.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        img_pos = F.interpolate(img_pos, size=gs_new, mode='bilinear') # 双线性插值 将图像进行缩放
        img_pos = img_pos.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)

        # 构建新位置编码
        if num_prompts > 0:
            prompt_pos = torch.zeros(1, num_prompts, posemb.shape[-1],
                                     device=posemb.device)  # 设备保持一致
            new_posemb = torch.cat([cls_pos, prompt_pos, img_pos], dim=1)
        else:
            new_posemb = torch.cat([cls_pos, img_pos], dim=1)

        # 尺寸验证
        if strict_size and new_posemb.shape[1] != posemb_new.shape[1]:
            raise ValueError(f"Size mismatch: {new_posemb.shape} vs {posemb_new.shape}")

        return new_posemb



def vit_base_patch16_224_TransReID_Prompt_BAPM(img_size=(256, 128), stride_size=16, drop_rate=0.0,
                                   attn_drop_rate=0.0, drop_path_rate=0.1, local_feature=False,
                                   args=None, **kwargs):
    model = TransReID(
        num_classes=0, img_size=img_size, patch_size=16, stride_size=stride_size,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        local_feature=local_feature, args=args, **kwargs)
    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
