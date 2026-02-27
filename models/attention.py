import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BidirectionalAttention(nn.Module):
    """Bidirectional cross-attention module for feature interaction."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn_q = nn.LayerNorm(dim)
        self.norm_ffn_k = nn.LayerNorm(dim)

    def _attend(self, feat_a, feat_b):
        """Compute cross-attention from feat_a (query) to feat_b (key/value)."""
        B, N, C = feat_a.shape

        q = self.q_proj(feat_a)
        k = self.k_proj(feat_b)
        v = self.v_proj(feat_b)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        return out

    def forward(self, feat_q, feat_k):
        """
        Args:
            feat_q: query features, shape (B, N, C)
            feat_k: key features, shape (B, N, C)
        Returns:
            feat_q, feat_k: updated features after bidirectional attention
        """
        # Bidirectional cross-attention
        feat_q = self.norm1(feat_q + self._attend(feat_q, feat_k))
        feat_k = self.norm2(feat_k + self._attend(feat_k, feat_q))

        # Feed-forward
        feat_q = self.norm_ffn_q(feat_q + self.ffn(feat_q))
        feat_k = self.norm_ffn_k(feat_k + self.ffn(feat_k))

        return feat_q, feat_k
