# models/adapters/lora.py
"""
LoRA (Low-Rank Adaptation) 适配器
用于大语言模型的参数高效微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math

class LoRALayer(nn.Module):
    """
    LoRA层的基础实现
    使用低秩矩阵A和B来近似权重更新
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, ..., in_features]
        Returns:
            [B, ..., out_features]
        """
        # LoRA增量：x @ A @ B
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    带LoRA的线性层
    原始权重被冻结，只训练LoRA参数
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        merge_weights: bool = False
    ):
        super().__init__()
        
        # 原始线性层（将被冻结）
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA层
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
        
        # 是否合并权重
        self.merge_weights = merge_weights
        self.merged = False
        
        # 冻结原始权重
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # 如果已经合并，直接使用线性层
            return self.linear(x)
        else:
            # 原始输出 + LoRA增量
            return self.linear(x) + self.lora(x)
    
    def merge_lora_weights(self):
        """将LoRA权重合并到原始权重中"""
        if not self.merged:
            # W' = W + α/r * B @ A
            with torch.no_grad():
                lora_weight = (self.lora.lora_B @ self.lora.lora_A.t()) * self.lora.scaling
                self.linear.weight.data += lora_weight.t()
            self.merged = True
    
    def unmerge_lora_weights(self):
        """分离LoRA权重"""
        if self.merged:
            with torch.no_grad():
                lora_weight = (self.lora.lora_B @ self.lora.lora_A.t()) * self.lora.scaling
                self.linear.weight.data -= lora_weight.t()
            self.merged = False


class LoRAAttention(nn.Module):
    """
    在注意力层中应用LoRA
    只对Q、K、V投影添加LoRA
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        apply_to_qkv: List[str] = ['q', 'k', 'v']
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 原始投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 冻结原始权重
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.weight.requires_grad = False
            if proj.bias is not None:
                proj.bias.requires_grad = False
        
        # 添加LoRA层
        self.lora_layers = nn.ModuleDict()
        if 'q' in apply_to_qkv:
            self.lora_layers['q'] = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout)
        if 'k' in apply_to_qkv:
            self.lora_layers['k'] = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout)
        if 'v' in apply_to_qkv:
            self.lora_layers['v'] = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = query.shape
        
        # Q, K, V投影（带LoRA）
        q = self.q_proj(query)
        if 'q' in self.lora_layers:
            q = q + self.lora_layers['q'](query)
        
        k = self.k_proj(key)
        if 'k' in self.lora_layers:
            k = k + self.lora_layers['k'](key)
        
        v = self.v_proj(value)
        if 'v' in self.lora_layers:
            v = v + self.lora_layers['v'](value)
        
        # 重塑为多头
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: List[str] = ['q_proj', 'v_proj', 'k_proj', 'out_proj']
) -> nn.Module:
    """
    将LoRA应用到现有模型的指定模块
    
    Args:
        model: 原始模型
        rank: LoRA秩
        alpha: LoRA缩放因子
        dropout: Dropout率
        target_modules: 要应用LoRA的模块名称列表
        
    Returns:
        带LoRA的模型
    """
    for name, module in model.named_modules():
        # 检查是否是目标模块
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 获取父模块和属性名
                parent = model
                attrs = name.split('.')
                for attr in attrs[:-1]:
                    parent = getattr(parent, attr)
                
                # 替换为LoRA线性层
                lora_linear = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    bias=module.bias is not None
                )
                
                # 复制原始权重
                lora_linear.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_linear.linear.bias.data = module.bias.data.clone()
                
                # 替换模块
                setattr(parent, attrs[-1], lora_linear)
    
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """获取模型中所有的LoRA参数"""
    lora_params = []
    for module in model.modules():
        if isinstance(module, (LoRALayer, LoRALinear, LoRAAttention)):
            lora_params.extend([p for p in module.parameters() if p.requires_grad])
    return lora_params


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """统计LoRA参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'lora': lora_params,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
    }


# 测试代码
if __name__ == "__main__":
    # 测试LoRA线性层
    lora_linear = LoRALinear(768, 768, rank=8, alpha=16)
    x = torch.randn(4, 196, 768)
    out = lora_linear(x)
    print(f"LoRA线性层输出: {out.shape}")
    
    # 测试LoRA注意力
    lora_attn = LoRAAttention(768, 12, rank=8, alpha=16)
    q = k = v = torch.randn(4, 196, 768)
    attn_out = lora_attn(q, k, v)
    print(f"LoRA注意力输出: {attn_out.shape}")
    
    # 统计参数
    print("\nLoRA线性层参数统计:")
    print(f"总参数: {sum(p.numel() for p in lora_linear.parameters())}")
    print(f"可训练参数: {sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)}")
    
    print("\nLoRA注意力参数统计:")
    print(f"总参数: {sum(p.numel() for p in lora_attn.parameters())}")
    print(f"可训练参数: {sum(p.numel() for p in lora_attn.parameters() if p.requires_grad)}")