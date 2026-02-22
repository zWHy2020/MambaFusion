# models/fusion/cross_modal_alignment.py
"""
跨模态对齐模块 - 修复版
实现RGB、高光谱图像和文本之间的特征对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 投影层
        self.q_proj = nn.Linear(dim_q, embed_dim)
        self.k_proj = nn.Linear(dim_kv, embed_dim)
        self.v_proj = nn.Linear(dim_kv, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, N_q, dim_q]
            key: [B, N_kv, dim_kv]
            value: [B, N_kv, dim_kv]
            mask: 注意力掩码 [B, N_kv] 或 [B, 1, 1, N_kv]
        Returns:
            output: [B, N_q, embed_dim]
            attention: [B, num_heads, N_q, N_kv]
        """
        B, N_q, _ = query.shape
        N_kv = key.shape[1]
        
        # 投影Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim)
        
        # 调整维度 [B, num_heads, N, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N_q, N_kv]
        
        # 处理mask
        if mask is not None:
            # 确保mask的维度正确
            if mask.dim() == 2:  # [B, N_kv]
                # 扩展为 [B, 1, 1, N_kv]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:  # [B, 1, N_kv]
                # 扩展为 [B, 1, 1, N_kv]
                mask = mask.unsqueeze(2)
            
            # 应用mask (将padding位置设为-inf)
            # mask中1表示有效位置，0表示padding
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, self.embed_dim)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out, attn


class ContrastiveLearning(nn.Module):
    """对比学习模块，用于模态对齐"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            features_a: [B, D] 模态A的特征
            features_b: [B, D] 模态B的特征
        Returns:
            loss: 对比损失
        """
        # 归一化
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.matmul(features_a, features_b.t()) / self.temperature
        
        # 标签：对角线为正样本
        batch_size = features_a.shape[0]
        labels = torch.arange(batch_size, device=features_a.device)
        
        # 对比损失
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        
        return (loss_a + loss_b) / 2


class CrossModalAlignmentModule(nn.Module):
    """
    完整的跨模态对齐模块
    实现RGB-Text, HSI-Text, RGB-HSI三组对齐
    """
    def __init__(
        self,
        rgb_dim: int = 768,
        hsi_dim: int = 768,
        text_dim: int = 768,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07,
        use_hsi: bool = True
    ):
        super().__init__()
        self.use_hsi = use_hsi
        # 跨模态注意力层
        # RGB -> Text
        self.rgb_to_text_attn = CrossModalAttention(
            rgb_dim, text_dim, embed_dim, num_heads, dropout
        )
        
        # HSI -> Text
        self.hsi_to_text_attn = CrossModalAttention(
            hsi_dim, text_dim, embed_dim, num_heads, dropout
        )
        
        # RGB -> HSI
        self.rgb_to_hsi_attn = CrossModalAttention(
            rgb_dim, hsi_dim, embed_dim, num_heads, dropout
        )
        
        # Text -> RGB
        self.text_to_rgb_attn = CrossModalAttention(
            text_dim, rgb_dim, embed_dim, num_heads, dropout
        )
        
        # Text -> HSI
        self.text_to_hsi_attn = CrossModalAttention(
            text_dim, hsi_dim, embed_dim, num_heads, dropout
        )
        
        # 对比学习
        self.contrastive = ContrastiveLearning(temperature)
        
        # 特征融合层
        self.fusion_rgb = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_hsi = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_text = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        rgb_features: torch.Tensor,
        hsi_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            rgb_features: [B, N_rgb, D_rgb]
            hsi_features: [B, N_hsi, D_hsi]
            text_features: [B, N_text, D_text]
            text_mask: [B, N_text] 文本掩码，1表示有效token，0表示padding
            
        Returns:
            对齐后的特征和损失
        """
        # RGB池化特征（用于对比学习）
        rgb_pooled = rgb_features.mean(dim=1)  # [B, D]
        hsi_pooled = hsi_features.mean(dim=1)
        text_pooled = text_features.mean(dim=1)
        
        # 跨模态注意力
        # RGB <-> Text
        rgb_from_text, _ = self.text_to_rgb_attn(
            rgb_features, text_features, text_features, text_mask
        )
        text_from_rgb, _ = self.rgb_to_text_attn(
            text_features, rgb_features, rgb_features, None  # RGB不需要mask
        )
        
        # HSI <-> Text
        hsi_from_text, _ = self.text_to_hsi_attn(
            hsi_features, text_features, text_features, text_mask
        )
        text_from_hsi, _ = self.hsi_to_text_attn(
            text_features, hsi_features, hsi_features, None  # HSI不需要mask
        )
        
        # RGB <-> HSI
        rgb_from_hsi, _ = self.rgb_to_hsi_attn(
            rgb_features, hsi_features, hsi_features, None  # 都不需要mask
        )
        
        # 特征融合
        rgb_aligned = self.fusion_rgb(
            torch.cat([rgb_features, rgb_from_text], dim=-1)
        )
        hsi_aligned = self.fusion_hsi(
            torch.cat([hsi_features, hsi_from_text], dim=-1)
        )
        text_aligned = self.fusion_text(
            torch.cat([text_features, text_from_rgb], dim=-1)
        )
        
        # 对比学习损失
        loss_rgb_text = self.contrastive(rgb_pooled, text_pooled)
        loss_hsi_text = self.contrastive(hsi_pooled, text_pooled)
        loss_rgb_hsi = self.contrastive(rgb_pooled, hsi_pooled)
        
        if self.use_hsi:
            loss_hsi_text = self.contrastive(hsi_pooled, text_pooled)
            loss_rgb_hsi = self.contrastive(rgb_pooled, hsi_pooled)
            alignment_loss = (loss_rgb_text + loss_hsi_text + loss_rgb_hsi) / 3
        else:
            alignment_loss = loss_rgb_text

        return {
            'rgb_aligned': rgb_aligned,
            'hsi_aligned': hsi_aligned,
            'text_aligned': text_aligned,
            'alignment_loss': alignment_loss
        }


# 测试代码
if __name__ == "__main__":
    # 创建对齐模块
    alignment = CrossModalAlignmentModule(
        rgb_dim=768,
        hsi_dim=768,
        text_dim=768,
        embed_dim=768
    )
    
    # 模拟输入
    B, N_rgb, N_hsi, N_text = 4, 197, 65, 50
    rgb_features = torch.randn(B, N_rgb, 768)
    hsi_features = torch.randn(B, N_hsi, 768)
    text_features = torch.randn(B, N_text, 768)
    text_mask = torch.ones(B, N_text)  # [B, N_text]
    text_mask[:, 40:] = 0  # 模拟padding
    
    # 前向传播
    outputs = alignment(rgb_features, hsi_features, text_features, text_mask)
    
    print(f"RGB对齐特征: {outputs['rgb_aligned'].shape}")
    print(f"HSI对齐特征: {outputs['hsi_aligned'].shape}")
    print(f"文本对齐特征: {outputs['text_aligned'].shape}")
    print(f"对齐损失: {outputs['alignment_loss'].item():.4f}")
    print("✓ 测试通过！")