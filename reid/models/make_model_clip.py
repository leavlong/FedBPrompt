import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 
        tokenized_prompts = tokenized_prompts.to(x.device)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg , promptconfig):
        super(build_transformer, self).__init__()
        # self.model_name = cfg.MODEL.NAME 
        self.model_name = cfg.backbone
        # self.cos_layer = cfg.MODEL.COS_LAYER
        self.cos_layer = False
        # self.neck = cfg.MODEL.NECK
        self.neck =  'bnneck'
        # self.neck_feat = cfg.TEST.NECK_FEAT
        self.neck_feat = 'before'
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        # self.sie_coe = cfg.MODEL.SIE_COE  
        self.sie_coe = 3.0  

        #分类器剥离出来
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # self.classifier.apply(weights_init_classifier)
        # self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)#投影分类器，作用于投影后的特征
        # self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.h_resolution = int((cfg.size_train[0]-16)//16 + 1)
        # self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.w_resolution = int((cfg.size_train[1]-16)//16 + 1)
        # self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        self.vision_stride_size = 16
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size , promptconfig)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        # if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
        if False and False:
            if True and True:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            # elif cfg.MODEL.SIE_CAMERA:
            elif True:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            # elif cfg.MODEL.SIE_VIEW:
            elif True:
                self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(view_num))

        # dataset_name = cfg.DATASETS.NAMES
        dataset_name = "ReID"
        #返回拼接后的提示嵌入，这些嵌入可以用于引导预训练模型生成与特定类别相关的输出
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)#获得文本编码器

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text == True:
            prompts = self.prompt_learner(label) #根据标签生成提示
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)#生成文本特征
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)  #x3,x4,x_proj
            if self.model_name == 'RN50':
                return image_features_proj[0]#可以理解为： cls令牌 ？
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            #做平均池化
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        # if self.training:
        #     cls_score = self.classifier(feat)
        #     cls_score_proj = self.classifier_proj(feat_proj)
        #     return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj#返回结果是一个元组
        if self.training:#剥离分类层之后的训练模式返回   feat对标DACS原本的模型输出（未归一化）
            return [feat, feat_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj#返回结果是一个元组


        # else:
        #     if self.neck_feat == 'after': #before
        #         # print("Test with feature after BN")
        #         return feat
        #     else:#before
        #         # return torch.cat([img_feature, img_feature_proj], dim=1)
        #         return feat#换用DACS的测试返回
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:#before √
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class build_transformer_sub(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg , promptconfig):
        super(build_transformer, self).__init__()
        # self.model_name = cfg.MODEL.NAME 
        self.model_name = cfg.backbone
        # self.cos_layer = cfg.MODEL.COS_LAYER
        self.cos_layer = False
        # self.neck = cfg.MODEL.NECK
        self.neck =  'bnneck'
        # self.neck_feat = cfg.TEST.NECK_FEAT
        self.neck_feat = 'before'
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        # self.sie_coe = cfg.MODEL.SIE_COE  
        self.sie_coe = 3.0  

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)#投影分类器，作用于投影后的特征
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.h_resolution = int((cfg.size_train[0]-16)//16 + 1)
        # self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.w_resolution = int((cfg.size_train[1]-16)//16 + 1)
        # self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        self.vision_stride_size = 16
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size , promptconfig= promptconfig)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        # if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
        if True and True:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_CAMERA:
        elif True:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_VIEW:
        elif True:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

    def forward(self, x = None, get_image = False, cam_label= None, view_label=None):

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj#返回结果是一个元组

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
            
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_sub_model(cfg, num_class, camera_num = 0, view_num = 0):
    model = build_transformer_sub(num_class, camera_num, view_num, cfg , promptconfig= promptconfig)
    model = model.cuda()
    model = nn.DataParallel(model) if cfg.is_parallel else model
    return model

def make_models(cfg, num_class=None, camera_num = 0, view_num = 0 , promptconfig = None):
    model = build_transformer(num_class, camera_num, view_num, cfg , promptconfig)
    model = model.cuda()
    model = nn.DataParallel(model) if cfg.is_parallel else model
    return model

from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size , promptconfig):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size , promptconfig)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person." #初始化提示模板

        ctx_dim = 512 #嵌入维度
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4 #可学习部分的token数量
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() #将初始化的提示文本token化
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) #将提示转换为嵌入表示
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4 #每个类别特定的上下文向量数量
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) #为每个类别初始化一个可学习的上下文向量矩阵
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) #将矩阵注册为模型参数

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):   
        cls_ctx = self.cls_ctx[label] #为每个输入标签选择上下文向量
        b = label.shape[0] #获得批次大小
        prefix = self.token_prefix.expand(b, -1, -1) #前缀张量
        suffix = self.token_suffix.expand(b, -1, -1) #后缀张量
            
        prompts = torch.cat( #形成完整的嵌入提示
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 

