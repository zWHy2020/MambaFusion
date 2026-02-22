# models/fusion/multi_modal_fusion.py
"""
多模态融合模块
将对齐后的RGB、HSI、Text特征进行深度融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import math

class MultiModalTransformerBlock(nn.Module):
    """多模态Transformer块"""
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 自注意力
        x = x + self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=mask, need_weights=False
        )[0]
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        return x


class GatedFusion(nn.Module):
    """门控融合机制"""
    def __init__(self, num_modalities: int = 3, embed_dim: int = 768):
        super().__init__()
        
        # 为每个模态学习一个门控权重
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 模态特征列表，每个 [B, D]
        Returns:
            fused: [B, D]
        """
        # 拼接所有模态
        concat_features = torch.cat(modality_features, dim=-1)  # [B, num_modalities * D]
        
        # 计算门控权重
        gates = self.gate(concat_features)  # [B, num_modalities]
        
        # 加权融合
        stacked = torch.stack(modality_features, dim=1)  # [B, num_modalities, D]
        gates = gates.unsqueeze(-1)  # [B, num_modalities, 1]
        
        fused = (stacked * gates).sum(dim=1)  # [B, D]
        
        return fused


class HierarchicalFusion(nn.Module):
    """层次化融合策略"""
    def __init__(self, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        
        # 第一层：两两融合
        self.fusion_rgb_hsi = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_rgb_text = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_hsi_text = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 第二层：三模态融合
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(
        self,
        rgb_features: torch.Tensor,
        hsi_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb_features, hsi_features, text_features: [B, D]
        Returns:
            fused: [B, D]
        """
        # 两两融合
        rgb_hsi = self.fusion_rgb_hsi(torch.cat([rgb_features, hsi_features], dim=-1))
        rgb_text = self.fusion_rgb_text(torch.cat([rgb_features, text_features], dim=-1))
        hsi_text = self.fusion_hsi_text(torch.cat([hsi_features, text_features], dim=-1))
        
        # 最终融合
        final = self.final_fusion(torch.cat([rgb_hsi, rgb_text, hsi_text], dim=-1))
        
        return final


class MultiModalFusionModule(nn.Module):
    """
    完整的多模态融合模块
    结合Transformer、门控融合和层次化融合
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 4,
        dropout: float = 0.1,
        fusion_strategy: str = 'hierarchical'  # 'concat', 'gated', 'hierarchical'
    ):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        
        # 模态特定的token（类似CLIP的learnable tokens）
        self.rgb_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.hsi_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.text_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            MultiModalTransformerBlock(embed_dim, num_heads, 4.0, dropout)
            for _ in range(num_layers)
        ])
        
        # 不同的融合策略
        if fusion_strategy == 'gated':
            self.fusion = GatedFusion(num_modalities=3, embed_dim=embed_dim)
        elif fusion_strategy == 'hierarchical':
            self.fusion = HierarchicalFusion(embed_dim, dropout)
        else:  # concat
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        
        # 最终的特征增强
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        rgb_features: torch.Tensor,
        hsi_features: torch.Tensor,
        text_features: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            rgb_features: [B, N_rgb, D]
            hsi_features: [B, N_hsi, D]
            text_features: [B, N_text, D]
            return_intermediate: 是否返回中间特征
            
        Returns:
            融合后的特征
        """
        B = rgb_features.shape[0]
        
        # 添加模态特定的token
        rgb_token = self.rgb_token.expand(B, -1, -1)
        hsi_token = self.hsi_token.expand(B, -1, -1)
        text_token = self.text_token.expand(B, -1, -1)
        
        rgb_features = torch.cat([rgb_token, rgb_features], dim=1)
        hsi_features = torch.cat([hsi_token, hsi_features], dim=1)
        text_features = torch.cat([text_token, text_features], dim=1)
        
        # 拼接所有模态的特征
        # [B, N_rgb+N_hsi+N_text+3, D]
        all_features = torch.cat([rgb_features, hsi_features, text_features], dim=1)
        
        # 通过Transformer层
        intermediate_features = []
        for block in self.transformer_blocks:
            all_features = block(all_features)
            if return_intermediate:
                intermediate_features.append(all_features)
        
        # 提取每个模态的token（第一个位置）
        n_rgb = rgb_features.shape[1]
        n_hsi = hsi_features.shape[1]
        
        rgb_token_out = all_features[:, 0, :]  # [B, D]
        hsi_token_out = all_features[:, n_rgb, :]
        text_token_out = all_features[:, n_rgb + n_hsi, :]
        
        # 融合策略
        if self.fusion_strategy == 'concat':
            fused = self.fusion(
                torch.cat([rgb_token_out, hsi_token_out, text_token_out], dim=-1)
            )
        elif self.fusion_strategy == 'gated':
            fused = self.fusion([rgb_token_out, hsi_token_out, text_token_out])
        else:  # hierarchical
            fused = self.fusion(rgb_token_out, hsi_token_out, text_token_out)
        
        # 最终投影
        fused = self.final_proj(fused)
        
        outputs = {
            'fused_features': fused,
            'rgb_token': rgb_token_out,
            'hsi_token': hsi_token_out,
            'text_token': text_token_out,
            'all_features': all_features
        }
        
        if return_intermediate:
            outputs['intermediate_features'] = intermediate_features
        
        return outputs


# 测试代码
if __name__ == "__main__":
    # 创建融合模块
    fusion = MultiModalFusionModule(
        embed_dim=768,
        num_heads=12,
        num_layers=4,
        fusion_strategy='hierarchical'
    )
    
    # 模拟输入
    B = 4
    rgb_features = torch.randn(B, 196, 768)
    hsi_features = torch.randn(B, 64, 768)
    text_features = torch.randn(B, 50, 768)
    
    # 前向传播
    outputs = fusion(rgb_features, hsi_features, text_features, return_intermediate=True)
    
    print(f"融合特征: {outputs['fused_features'].shape}")
    print(f"RGB token: {outputs['rgb_token'].shape}")
    print(f"HSI token: {outputs['hsi_token'].shape}")
    print(f"Text token: {outputs['text_token'].shape}")
    print(f"中间层数: {len(outputs['intermediate_features'])}")
    print(f"模型参数量: {sum(p.numel() for p in fusion.parameters())/1e6:.2f}M")