import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGlobalFusion(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()
        self.attention = nn.Conv2d(channel_dim * 2, 2, kernel_size=1)

    def forward(self, base_latent, refined_latent):
        x = torch.cat([base_latent, refined_latent], dim=1)  # [B, 2C, H, W]
        logits = self.attention(x)                           # [B, 2, H, W]
        weights = F.softmax(logits, dim=1)                   # [B, 2, H, W]
        w_b = weights[:, 0:1, :, :]
        w_r = weights[:, 1:2, :, :]
        return w_b * base_latent + w_r * refined_latent


class DynamicSpatialFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, base_latent, refined_latent):
        avg_pool = torch.mean(refined_latent, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(base_latent, dim=1, keepdim=True)   # [B, 1, H, W]
        concat = torch.cat([avg_pool, max_pool], dim=1)             # [B, 2, H, W]
        attention = torch.sigmoid(self.spatial_attn(concat))        # [B, 1, H, W]
        return attention * refined_latent + (1 - attention) * base_latent
