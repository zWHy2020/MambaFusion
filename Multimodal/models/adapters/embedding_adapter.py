# models/adapters/embedding_adapter.py
"""
LLM Embedding 适配器
将多模态特征转换为 LLM 可理解的 token embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class MultiModalEmbeddingAdapter(nn.Module):
    """
    使用 Q-Former 风格的架构进行特征压缩和对齐
    """
    def __init__(
        self,
        multimodal_dim: int = 768,
        llm_hidden_size: int = 4096,  # Qwen2.5-7B 的隐藏层维度
        num_query_tokens: int = 32,
        num_layers: int = 4,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_query_tokens = num_query_tokens
        self.llm_hidden_size = llm_hidden_size
        
        # 可学习的查询 tokens (类似 BLIP-2 的 Q-Former)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, multimodal_dim)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Cross-Attention 层:从多模态特征中提取信息
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                query_dim=multimodal_dim,
                context_dim=multimodal_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 投影到 LLM 空间
        self.projector = nn.Sequential(
            nn.Linear(multimodal_dim, multimodal_dim * 2),
            nn.LayerNorm(multimodal_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(multimodal_dim * 2, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size)
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_query_tokens, multimodal_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(
        self,
        multimodal_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            multimodal_features: [B, N, D] 融合后的多模态特征
            attention_mask: [B, N] 注意力掩码
            
        Returns:
            llm_embeddings: [B, num_query_tokens, llm_hidden_size]
        """
        B = multimodal_features.shape[0]
        
        # 扩展查询 tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_tokens = query_tokens + self.pos_embed
        
        # 通过多层 Cross-Attention 提取特征
        for layer in self.cross_attention_layers:
            query_tokens = layer(
                query=query_tokens,
                context=multimodal_features,
                mask=attention_mask
            )
        
        # 投影到 LLM 空间
        llm_embeddings = self.projector(query_tokens)
        
        return llm_embeddings


class CrossAttentionLayer(nn.Module):
    """Cross-Attention 层"""
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(query_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Cross-Attention
        attn_out, _ = self.cross_attn(
            query=self.norm1(query),
            key=context,
            value=context,
            key_padding_mask=mask if mask is not None else None
        )
        query = query + attn_out
        
        # FFN
        query = query + self.ffn(self.norm2(query))
        
        return query


class TokenTypeEmbedding(nn.Module):
    """
    Token 类型嵌入
    区分不同模态的 token (类似 BERT 的 segment embedding)
    """
    def __init__(self, hidden_size: int, num_types: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(num_types, hidden_size)
        
    def forward(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_type_ids)


class ModalityFusion(nn.Module):
    """
    模态融合层
    在 LLM embedding 空间中融合不同模态
    """
    def __init__(
        self,
        hidden_size: int,
        num_modalities: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, modality_embeddings: list) -> torch.Tensor:
        """
        Args:
            modality_embeddings: List of [B, N, D]
        Returns:
            fused: [B, N, D]
        """
        B, N, D = modality_embeddings[0].shape
        
        # 拼接所有模态用于计算门控权重
        concat = torch.cat([
            emb.mean(dim=1) for emb in modality_embeddings
        ], dim=-1)
        
        # 计算门控权重
        gates = self.gate(concat)  # [B, num_modalities]
        
        # 加权融合
        stacked = torch.stack(modality_embeddings, dim=1)  # [B, M, N, D]
        gates = gates.view(B, -1, 1, 1)  # [B, M, 1, 1]
        
        fused = (stacked * gates).sum(dim=1)  # [B, N, D]
        
        return self.dropout(self.norm(fused))


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试 Embedding 适配器")
    print("="*60)
    
    # 创建适配器
    adapter = MultiModalEmbeddingAdapter(
        multimodal_dim=768,
        llm_hidden_size=4096,
        num_query_tokens=32,
        num_layers=4
    )
    
    # 模拟输入
    B, N, D = 4, 256, 768
    multimodal_features = torch.randn(B, N, D)
    
    # 前向传播
    llm_embeddings = adapter(multimodal_features)
    
    print(f"\n输入形状: {multimodal_features.shape}")
    print(f"输出形状: {llm_embeddings.shape}")
    print(f"参数量: {sum(p.numel() for p in adapter.parameters())/1e6:.2f}M")
    
    # 测试模态融合
    print("\n测试模态融合:")
    fusion = ModalityFusion(hidden_size=4096, num_modalities=3)
    
    rgb_emb = torch.randn(B, 32, 4096)
    hsi_emb = torch.randn(B, 32, 4096)
    text_emb = torch.randn(B, 32, 4096)
    
    fused = fusion([rgb_emb, hsi_emb, text_emb])
    print(f"融合输出: {fused.shape}")
    
    print("\n✓ 测试通过!")