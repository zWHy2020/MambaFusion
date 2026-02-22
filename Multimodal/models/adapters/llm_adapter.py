# models/adapters/llm_adapter.py
"""
大语言模型适配器
将多模态特征转换为LLM可用的输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class LLMAdapter(nn.Module):
    """
    大语言模型适配器
    通过可学习的查询token和跨注意力机制，将多模态特征映射到LLM的输入空间
    """
    def __init__(
        self,
        multimodal_dim: int = 768,
        llm_hidden_size: int = 4096,
        num_query_tokens: int = 32,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        """
        Args:
            multimodal_dim: 多模态特征维度
            llm_hidden_size: LLM隐藏层维度
            num_query_tokens: 可学习查询token数量
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        
        self.multimodal_dim = multimodal_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_query_tokens = num_query_tokens
        
        # 可学习的查询token
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, multimodal_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # 跨注意力层：查询多模态特征
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=multimodal_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(multimodal_dim)
        self.norm2 = nn.LayerNorm(multimodal_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(multimodal_dim, multimodal_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(multimodal_dim * 4, multimodal_dim),
            nn.Dropout(dropout)
        )
        
        # 投影到LLM维度
        self.projector = nn.Sequential(
            nn.Linear(multimodal_dim, multimodal_dim * 2),
            nn.GELU(),
            nn.LayerNorm(multimodal_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(multimodal_dim * 2, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size)
        )
        
    def forward(
        self,
        multimodal_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            multimodal_features: [B, N, D] 融合后的多模态特征
            attention_mask: [B, N] 注意力掩码（可选）
            
        Returns:
            llm_tokens: [B, num_query_tokens, llm_hidden_size] LLM输入token
        """
        B = multimodal_features.shape[0]
        
        # 扩展查询token
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, num_query, D]
        
        # 跨注意力：从多模态特征中提取信息
        attended_features, attn_weights = self.cross_attn(
            query=query_tokens,
            key=multimodal_features,
            value=multimodal_features,
            key_padding_mask=attention_mask if attention_mask is not None else None
        )
        
        # 残差连接和LayerNorm
        query_tokens = self.norm1(query_tokens + attended_features)
        
        # FFN
        ffn_output = self.ffn(query_tokens)
        query_tokens = self.norm2(query_tokens + ffn_output)
        
        # 投影到LLM维度
        llm_tokens = self.projector(query_tokens)  # [B, num_query, llm_hidden]
        
        return llm_tokens


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler
    基于Perceiver架构的高效重采样器，用于压缩多模态特征
    """
    def __init__(
        self,
        dim: int = 768,
        depth: int = 6,
        num_latents: int = 64,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        """
        Args:
            dim: 特征维度
            depth: Perceiver层数
            num_latents: 潜在变量数量
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        
        self.num_latents = num_latents
        
        # 可学习的潜在变量
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)
        
        # Perceiver层
        self.layers = nn.ModuleList([
            PerceiverBlock(dim, num_heads, dropout)
            for _ in range(depth)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] 输入特征
        Returns:
            [B, num_latents, D] 重采样后的特征
        """
        B = x.shape[0]
        
        # 扩展潜在变量
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        
        # 通过Perceiver层
        for layer in self.layers:
            latents = layer(latents, x)
        
        return latents


class PerceiverBlock(nn.Module):
    """Perceiver块"""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # 跨注意力
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, M, D] 潜在变量
            x: [B, N, D] 输入特征
        """
        # 跨注意力
        cross_out, _ = self.cross_attn(
            query=latents,
            key=x,
            value=x
        )
        latents = self.norm1(latents + cross_out)
        
        # 自注意力
        self_out, _ = self.self_attn(latents, latents, latents)
        latents = self.norm2(latents + self_out)
        
        # FFN
        ffn_out = self.ffn(latents)
        latents = self.norm3(latents + ffn_out)
        
        return latents


class QFormerAdapter(nn.Module):
    """
    Q-Former适配器
    基于BLIP-2的Q-Former架构
    """
    def __init__(
        self,
        dim: int = 768,
        num_queries: int = 32,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_queries = num_queries
        
        # 查询嵌入
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, dim))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            image_features: [B, N, D] 图像特征
            text_features: [B, M, D] 文本特征（可选）
        Returns:
            [B, num_queries, D] 查询输出
        """
        B = image_features.shape[0]
        
        # 扩展查询
        queries = self.query_embed.expand(B, -1, -1)
        
        # 拼接所有特征
        if text_features is not None:
            all_features = torch.cat([queries, image_features, text_features], dim=1)
        else:
            all_features = torch.cat([queries, image_features], dim=1)
        
        # 通过Transformer
        output = self.transformer(all_features)
        
        # 只返回查询部分
        query_output = output[:, :self.num_queries, :]
        
        return query_output


# 测试代码
if __name__ == "__main__":
    # 测试LLM适配器
    adapter = LLMAdapter(
        multimodal_dim=768,
        llm_hidden_size=4096,
        num_query_tokens=32
    )
    
    # 模拟输入
    B, N, D = 4, 196, 768
    multimodal_features = torch.randn(B, N, D)
    
    # 前向传播
    llm_tokens = adapter(multimodal_features)
    
    print(f"输入形状: {multimodal_features.shape}")
    print(f"输出形状: {llm_tokens.shape}")
    print(f"参数量: {sum(p.numel() for p in adapter.parameters())/1e6:.2f}M")
    
    # 测试Perceiver Resampler
    print("\n测试Perceiver Resampler:")
    resampler = PerceiverResampler(dim=768, num_latents=64)
    resampled = resampler(multimodal_features)
    print(f"重采样输出: {resampled.shape}")
    
    # 测试Q-Former
    print("\nQ-Former:")
    qformer = QFormerAdapter(dim=768, num_queries=32)
    queries = qformer(multimodal_features)
    print(f"Q-Former输出: {queries.shape}")