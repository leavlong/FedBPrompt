# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(num_classes):    # modified by gu
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    triplet = TripletLoss(0.3)  # triplet loss
    print("using triplet loss with margin:{}".format(0.3))
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, numclasses:", num_classes)
    def loss_func(score, feat, target, target_cam, i2tscore = None):
        if isinstance(score, list):
            ID_LOSS = [xent(scor, target) for scor in score[0:]]
            ID_LOSS = sum(ID_LOSS)
        else:
            ID_LOSS = xent(score, target)

        if isinstance(feat, list):
            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
            TRI_LOSS = sum(TRI_LOSS) 
        else:   
            TRI_LOSS = triplet(feat, target)[0]
        
        loss = 1.0 * ID_LOSS + 1.0 * TRI_LOSS

        if i2tscore != None:
            I2TLOSS = xent(i2tscore, target)
            loss = 1.0 * I2TLOSS + loss
            
        return loss
            
    return loss_func, center_criterion

def make_loss_mm_clip(num_classes):    # modified by gu
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    triplet = TripletLoss(0.3)  # triplet loss
    print("using triplet loss with margin:{}".format(0.3))
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, numclasses:", num_classes)
    def loss_func(score, feat, target, target_cam, i2tscore = None):
        if isinstance(score, list):
            ID_LOSS = [xent(scor, target) for scor in score[0:]]
            ID_LOSS = sum(ID_LOSS)
        else:
            ID_LOSS = xent(score, target)

        if isinstance(feat, list):
            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
            TRI_LOSS = sum(TRI_LOSS) 
        else:   
            TRI_LOSS = triplet(feat, target)[0]
        
        loss = 1.0 * ID_LOSS + 1.0 * TRI_LOSS

        if i2tscore != None:
            I2TLOSS = xent(i2tscore, target)
            loss = 1.0 * I2TLOSS + loss
            
        return loss
            
    return loss_func, center_criterion

