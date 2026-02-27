import torch
import torch.nn as nn
from .backbone import build_backbone
from .attention import BidirectionalAttention


class BAPM(nn.Module):
    """Bidirectional Attention-based Pose Matching (BAPM) model."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(
            cfg.MODEL.BACKBONE,
            pretrained=getattr(cfg.MODEL, 'PRETRAINED', True),
            out_dim=cfg.MODEL.EMBED_DIM,
        )
        self.attention = BidirectionalAttention(
            dim=cfg.MODEL.EMBED_DIM,
            num_heads=cfg.MODEL.NUM_HEADS,
            dropout=cfg.MODEL.DROPOUT,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM // 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.MODEL.EMBED_DIM // 2, cfg.MODEL.NUM_CLASSES),
        )

    def forward(self, x_query, x_key):
        """
        Args:
            x_query: query input tensor, shape (B, C, H, W)
            x_key: key input tensor, shape (B, C, H, W)
        Returns:
            output: pose matching scores, shape (B, num_classes)
        """
        feat_q = self.backbone(x_query)
        feat_k = self.backbone(x_key)

        feat_q, feat_k = self.attention(feat_q, feat_k)

        # Global average pooling
        feat_q = feat_q.mean(dim=1)
        output = self.head(feat_q)
        return output
