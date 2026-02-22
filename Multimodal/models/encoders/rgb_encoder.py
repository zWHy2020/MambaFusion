# models/encoders/rgb_encoder.py
"""
RGB图像编码器模块
基于Vision Transformer架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

class PatchEmbedding(nn.Module):
    """将图像分割成patches并进行嵌入"""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 
                     kernel_size=patch_size, 
                     stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        
    def forward(self, x):
        return self.projection(x)

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力到值
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class RGBEncoder(nn.Module):
    """
    RGB图像编码器 - Vision Transformer
    
    Args:
        image_size: 输入图像大小
        patch_size: patch大小
        in_channels: 输入通道数
        embed_dim: 嵌入维度
        depth: Transformer块的数量
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例
        dropout: dropout率
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        
        # 位置编码
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化权重
        self.initialize_weights()
        
    def initialize_weights(self):
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化其他层
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            cls_token: 分类token [B, embed_dim]
            patch_tokens: 所有patch tokens [B, N, embed_dim]
        """
        B = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(x)
        
        # 添加CLS token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer块
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 返回CLS token和patch tokens
        return x[:, 0], x[:, 1:]

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = RGBEncoder(
        image_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    # 测试输入
    x = torch.randn(2, 3, 224, 224)
    
    # 前向传播
    cls_token, patch_tokens = model(x)
    
    print(f"CLS token shape: {cls_token.shape}")
    print(f"Patch tokens shape: {patch_tokens.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")