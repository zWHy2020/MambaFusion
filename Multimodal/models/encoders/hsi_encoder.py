# models/encoders/hsi_encoder.py
"""
SOTA轻量级高光谱图像编码器 - 修复版
修复了维度不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class SpectralGroupAttention(nn.Module):
    """光谱分组注意力"""
    def __init__(self, in_channels, num_groups=8, reduction=4):
        super().__init__()
        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups
        
        self.group_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.group_channels, self.group_channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.group_channels // reduction, self.group_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_groups)
        ])
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.num_groups, self.group_channels, H, W)
        
        out = []
        for i in range(self.num_groups):
            group = x[:, i]
            att = self.group_attention[i](group)
            out.append(group * att)
        
        out = torch.cat(out, dim=1)
        return out


class EfficientSpectralReduction(nn.Module):
    """高效光谱降维模块"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.spectral_att = SpectralGroupAttention(in_channels, num_groups)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.spectral_att(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class SpatialReductionBlock(nn.Module):
    """空间下采样块"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GroupedSpatialAttention(nn.Module):
    """
    分组空间注意力 - 修复版
    自动计算合适的注意力头数
    """
    def __init__(self, channels, num_heads=None):
        super().__init__()
        
        # 🔧 修复：自动选择能整除的头数
        if num_heads is None:
            # 优先选择的头数列表（从大到小）
            preferred_heads = [16, 12, 8, 6, 4, 2, 1]
            for h in preferred_heads:
                if channels % h == 0:
                    num_heads = h
                    break
        else:
            # 如果指定了头数但不能整除，自动调整
            if channels % num_heads != 0:
                print(f"Warning: channels({channels}) % num_heads({num_heads}) != 0")
                print(f"Auto-adjusting num_heads...")
                preferred_heads = [16, 12, 8, 6, 4, 2, 1]
                for h in preferred_heads:
                    if channels % h == 0 and h <= num_heads:
                        num_heads = h
                        break
                print(f"Adjusted num_heads to {num_heads}")
        
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.scale = self.head_channels ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 重塑为多头
        q = rearrange(q, 'b (h c) x y -> b h (x y) c', h=self.num_heads)
        k = rearrange(k, 'b (h c) x y -> b h (x y) c', h=self.num_heads)
        v = rearrange(v, 'b (h c) x y -> b h (x y) c', h=self.num_heads)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        out = attn @ v
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', x=H, y=W)
        out = self.proj(out)
        
        return out


class SpectralSpatialBlock(nn.Module):
    """光谱-空间联合处理块"""
    def __init__(self, channels, num_heads=None, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(channels)
        self.spatial_attn = GroupedSpatialAttention(channels, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, channels),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 空间注意力
        identity = x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.spatial_attn(x)
        x = identity + self.dropout1(x)
        
        # MLP
        identity = x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm2(x)
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = identity + x
        
        return x


class HSIEncoder(nn.Module):
    """
    SOTA轻量级高光谱编码器 - 修复版
    """
    
    def __init__(
        self,
        hsi_channels=224,
        spatial_size=64,
        embed_dim=768,
        num_groups=8,
        num_heads=8,  # 这个参数会被自动调整
        depth=2,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hsi_channels = hsi_channels
        self.spatial_size = spatial_size
        self.embed_dim = embed_dim
        
        # Stage 1: 光谱降维 224 -> 64
        self.spectral_reduction = EfficientSpectralReduction(
            in_channels=hsi_channels,
            out_channels=64,
            num_groups=num_groups
        )
        
        # Stage 2: 空间下采样 64x64 -> 32x32
        self.spatial_down1 = SpatialReductionBlock(64, 128, stride=2)
        
        # Stage 3: 光谱-空间联合处理
        # 🔧 修复：128通道，自动选择num_heads=8（128能被8整除）
        self.ss_block1 = SpectralSpatialBlock(128, num_heads=8, dropout=dropout)
        
        # Stage 4: 进一步空间下采样 32x32 -> 16x16
        self.spatial_down2 = SpatialReductionBlock(128, 256, stride=2)
        
        # Stage 5: 深层特征提取
        # 🔧 修复：256通道，自动选择num_heads=8（256能被8整除）
        self.ss_blocks = nn.ModuleList([
            SpectralSpatialBlock(256, num_heads=8, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Stage 6: 特征聚合
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 最终投影到embed_dim
        self.final_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Patch projection
        self.patch_size = 4
        num_patches = (16 // self.patch_size) ** 2
        
        self.patch_proj = nn.Conv2d(256, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self._initialize_weights()
        
        # 打印模型信息
        print(f"✓ HSI Encoder initialized:")
        print(f"  - Stage 3: 128 channels, 8 heads")
        print(f"  - Stage 5: 256 channels, 8 heads")
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 高光谱图像
            
        Returns:
            cls_token: [B, embed_dim]
            patch_tokens: [B, num_patches, embed_dim]
        """
        B = x.shape[0]
        
        # Stage 1: 光谱降维
        x = self.spectral_reduction(x)
        
        # Stage 2: 空间下采样1
        x = self.spatial_down1(x)
        
        # Stage 3: 光谱-空间处理
        x = self.ss_block1(x)
        
        # Stage 4: 空间下采样2
        x = self.spatial_down2(x)
        
        # Stage 5: 深层特征提取
        for block in self.ss_blocks:
            x = block(x)
        
        # 生成全局特征
        global_feat = self.global_pool(x).flatten(1)
        cls_token = self.final_proj(global_feat)
        
        # 生成patch tokens
        patch_feat = self.patch_proj(x)
        patch_tokens = rearrange(patch_feat, 'b c h w -> b (h w) c')
        patch_tokens = self.patch_norm(patch_tokens)
        patch_tokens = patch_tokens + self.pos_embed
        
        return cls_token, patch_tokens
    
    def get_feature_maps(self, x):
        """返回中间特征图"""
        features = {}
        
        x = self.spectral_reduction(x)
        features['stage1'] = x
        
        x = self.spatial_down1(x)
        features['stage2'] = x
        
        x = self.ss_block1(x)
        features['stage3'] = x
        
        x = self.spatial_down2(x)
        features['stage4'] = x
        
        for i, block in enumerate(self.ss_blocks):
            x = block(x)
            features[f'stage5_{i}'] = x
        
        return features


# 内存优化版本
class HSIEncoderCheckpoint(HSIEncoder):
    """带梯度检查点的HSI编码器"""
    def forward(self, x):
        B = x.shape[0]
        
        from torch.utils.checkpoint import checkpoint
        
        x = checkpoint(self.spectral_reduction, x, use_reentrant=False)
        x = checkpoint(self.spatial_down1, x, use_reentrant=False)
        x = checkpoint(self.ss_block1, x, use_reentrant=False)
        x = checkpoint(self.spatial_down2, x, use_reentrant=False)
        
        for block in self.ss_blocks:
            x = checkpoint(block, x, use_reentrant=False)
        
        global_feat = self.global_pool(x).flatten(1)
        cls_token = self.final_proj(global_feat)
        
        patch_feat = self.patch_proj(x)
        patch_tokens = rearrange(patch_feat, 'b c h w -> b (h w) c')
        patch_tokens = self.patch_norm(patch_tokens)
        patch_tokens = patch_tokens + self.pos_embed
        
        return cls_token, patch_tokens


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试修复后的HSI编码器")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = HSIEncoder(
        hsi_channels=224,
        spatial_size=64,
        embed_dim=768,
        num_groups=8,
        num_heads=8,  # 会被自动调整
        depth=2
    ).to(device)
    
    # 测试不同batch size
    print("\n显存测试:")
    print("-"*60)
    
    for batch_size in [1, 4, 8, 16, 20]:
        try:
            x = torch.randn(batch_size, 224, 64, 64).to(device)
            
            with torch.cuda.amp.autocast():
                cls_token, patch_tokens = model(x)
            
            if device == 'cuda':
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"Batch {batch_size:2d}: cls={cls_token.shape}, "
                      f"patches={patch_tokens.shape}, 显存={mem:.2f}GB")
            else:
                print(f"Batch {batch_size:2d}: cls={cls_token.shape}, "
                      f"patches={patch_tokens.shape}")
            
            del x, cls_token, patch_tokens
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Batch {batch_size:2d}: OOM")
                break
            else:
                raise e
    
    print("\n" + "="*60)
    print("✓ 测试完成！")
    print("="*60)