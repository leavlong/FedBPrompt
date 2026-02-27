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
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# ... (Mlp, Attention, Block, etc. classes remain unchanged, so they are omitted for brevity) ...
# I will include them in the final full code block below.

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
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.head_dim = head_dim

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class PatchEmbed_overlap(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        # ... (weights init)
    def forward(self, x):
        # ... (forward logic)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

### NEW ###
class SNR_Module_ViT(nn.Module):
    """
    SNR Module adapted for Vision Transformer features (B, N, D).
    """
    def __init__(self, embed_dim, r=16):
        super().__init__()
        self.IN = nn.InstanceNorm1d(embed_dim, affine=True)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // r),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // r, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        F_hat = self.IN(x.permute(0, 2, 1)).permute(0, 2, 1)
        R = x - F_hat
        r_pooled = R.mean(dim=1)
        a = self.attention(r_pooled).unsqueeze(1)
        R_plus = a * R
        R_minus = (1 - a) * R
        F_plus = F_hat + R_plus
        F_useful = F_hat + R_plus
        F_useless = F_hat + R_minus
        F_hat_pool = F_hat[:, 0]
        F_useful_pool = F_useful[:, 0]
        F_useless_pool = F_useless[:, 0]
        return F_plus, F_hat_pool, F_useful_pool, F_useless_pool
### END NEW ###


class TransReID(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 local_feature=False, args=None, prompt_cfg=None):
        super().__init__()
        # --- Start of your original __init__ ---
        self.num_classes = num_classes
        self.args = args
        self.num_features = self.embed_dim = embed_dim
        self.prompt_cfg = prompt_cfg
        self.local_feature = local_feature

        if prompt_cfg is None:
            self.prompt_cfg = type('Args', (), {
                'LOCATION': 'prepend', 'INITIATION': 'random',
                'NUM_TOKENS': 50, 'DROPOUT': 0.0, 'PROJECT': -1,
                'DEEP': True, 'NUM_DEEP_LAYERS': None, 'DEEP_SHARED': False
            })

        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size, patch_size=patch_size,
            stride_size=stride_size, in_chans=in_chans,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches
        self.depth = depth

        self._init_prompt_tokens(num_patches, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos_dim = num_patches + 1 + (self.prompt_cfg.NUM_TOKENS if self.prompt_cfg.LOCATION == 'prepend' else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_dim, embed_dim))
        self.embed_dim = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # --- End of your original __init__ ---

        ### NEW ###
        # Define SNR modules and insertion depths
        self.snr_depths = [3, 7, 11] # Insert after 4th, 8th, and 12th block (0-indexed)
        print(f"Integrating SNR modules after Transformer blocks: {self.snr_depths}")
        self.snr1 = SNR_Module_ViT(embed_dim)
        self.snr2 = SNR_Module_ViT(embed_dim)
        self.snr3 = SNR_Module_ViT(embed_dim)
        ### END NEW ###

        self.apply(self._init_weights)

    # All your original helper methods (_init_prompt_tokens, incorporate_prompt, _init_weights, etc.)
    # remain here, unchanged. I'm omitting them for brevity but they are in the final code.
    def _init_prompt_tokens(self, num_patches, embed_dim):
        num_tokens = self.prompt_cfg.NUM_TOKENS
        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_cfg.DROPOUT)
        if self.prompt_cfg.PROJECT > -1:
            prompt_dim = self.prompt_cfg.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, embed_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = embed_dim
            self.prompt_proj = nn.Identity()
        if self.prompt_cfg.INITIATION == "random":
            patch_size = _pair(num_patches)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            if self.prompt_cfg.DEEP:
                total_d_layer = self.depth - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        B = x.shape[0]
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    # ... other methods like set_training_mode, load_param, etc.

    ### MODIFIED FORWARD METHOD ###
    def forward(self, x, **kwargs):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Incorporate prompt tokens
        x = self.incorporate_prompt(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Initialize list for SNR intermediate outputs
        intermediate_outputs = []

        # Transformer processing with DEEP PROMPT logic and SNR integration
        if self.prompt_cfg.DEEP:
            # This is the 'forward_deep_prompt' logic, now with SNR
            for i in range(self.depth):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    if i <= len(self.deep_prompt_embeddings):
                        deep_prompt = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))
                        x = torch.cat((
                            x[:, :1, :],
                            deep_prompt,
                            x[:, 1 + self.num_tokens:, :]
                        ), dim=1)
                    x = self.blocks[i](x)
                
                # SNR Module Insertion
                if i == self.snr_depths[0]:
                    x, x_IN_1, x_1_useful, x_1_useless = self.snr1(x)
                    intermediate_outputs.extend([x_IN_1, x_1_useful, x_1_useless])
                elif i == self.snr_depths[1]:
                    x, x_IN_2, x_2_useful, x_2_useless = self.snr2(x)
                    intermediate_outputs.extend([x_IN_2, x_2_useful, x_2_useless])
                elif i == self.snr_depths[2]:
                    x, x_IN_3, x_3_useful, x_3_useless = self.snr3(x)
                    intermediate_outputs.extend([x_IN_3, x_3_useful, x_3_useless])

        else: # Standard "Shallow Prompt" logic with SNR integration
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                # SNR Module Insertion
                if i == self.snr_depths[0]:
                    x, x_IN_1, x_1_useful, x_1_useless = self.snr1(x)
                    intermediate_outputs.extend([x_IN_1, x_1_useful, x_1_useless])
                elif i == self.snr_depths[1]:
                    x, x_IN_2, x_2_useful, x_2_useless = self.snr2(x)
                    intermediate_outputs.extend([x_IN_2, x_2_useful, x_2_useless])
                elif i == self.snr_depths[2]:
                    x, x_IN_3, x_3_useful, x_3_useless = self.snr3(x)
                    intermediate_outputs.extend([x_IN_3, x_3_useful, x_3_useless])

        x = self.norm(x)
        final_features = x[:, 0]

        # Combine all 10 outputs into a single tuple
        all_outputs = [final_features] + intermediate_outputs

        if len(all_outputs) != 10:
            raise ValueError(f"Expected 10 outputs, but got {len(all_outputs)}.")

        return tuple(all_outputs)
    
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
# Keep all other functions like load_param, resize_pos_embed, _no_grad_trunc_normal_ etc. as they are.
# ...

def vit_base_patch16_224_TransReID_SNR_Prompt(img_size=(256, 128), stride_size=16, drop_rate=0.0,
                                       attn_drop_rate=0.0, drop_path_rate=0.1, local_feature=False,
                                       args=None, **kwargs):
    """
    Constructor for the TransReID model with both Prompt Tuning and SNR modules.
    """
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