import torch.nn as nn
import timm


def build_backbone(name, pretrained=True, out_dim=256):
    """Build a feature extraction backbone.

    Args:
        name: backbone architecture name (e.g., 'resnet50', 'vit_base_patch16_224')
        pretrained: whether to load pretrained weights
        out_dim: output feature dimension
    Returns:
        backbone module
    """
    return TimmBackbone(name, pretrained=pretrained, out_dim=out_dim)

class TimmBackbone(nn.Module):
    """Backbone wrapper using the timm library."""

    def __init__(self, name, pretrained=True, out_dim=256):
        super().__init__()
        self.encoder = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        in_dim = self.encoder.num_features
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: input image tensor, shape (B, C, H, W)
        Returns:
            features: shape (B, N, out_dim)
        """
        features = self.encoder.forward_features(x)
        # Flatten spatial dimensions if needed
        if features.dim() == 4:
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        features = self.proj(features)
        return features
