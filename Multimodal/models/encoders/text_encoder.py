# models/encoders/text_encoder.py
"""
文本编码器模块 - 基于BERT架构
用于编码病虫害的文本描述信息
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from typing import Optional, Dict

class TextEncoder(nn.Module):
    """
    文本编码器
    支持BERT、RoBERTa等预训练模型
    """
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        embed_dim: int = 768,
        freeze_encoder: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 加载预训练模型和tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        # 是否冻结预训练编码器
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 投影层：将BERT输出映射到统一的嵌入维度
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 额外的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_pooled: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入的token ids [B, L]
            attention_mask: 注意力掩码 [B, L]
            return_pooled: 是否返回池化后的特征
            
        Returns:
            包含编码结果的字典
        """
        # BERT编码
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取序列输出和池化输出
        sequence_output = outputs.last_hidden_state  # [B, L, hidden_size]
        pooled_output = outputs.pooler_output  # [B, hidden_size]
        
        # 投影到统一维度
        sequence_features = self.projection(sequence_output)  # [B, L, embed_dim]
        pooled_features = self.projection(pooled_output)  # [B, embed_dim]
        
        # 特征增强
        enhanced_features = self.feature_extractor(pooled_features)
        
        return {
            'sequence_features': sequence_features,  # 序列级特征
            'pooled_features': pooled_features,      # 池化特征
            'enhanced_features': enhanced_features,  # 增强特征
            'attention_mask': attention_mask         # 注意力掩码
        }
    
    def encode_text(self, texts: list, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        便捷的文本编码接口
        
        Args:
            texts: 文本列表
            device: 设备
            
        Returns:
            编码结果
        """
        # Tokenization
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 移到指定设备
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # 编码
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.forward(input_ids, attention_mask)
        
        return outputs

class BertTextEncoder(TextEncoder):
    """BERT文本编码器的简化版本"""
    def __init__(self, *args, **kwargs):
        if 'model_name' not in kwargs:
            kwargs['model_name'] = 'bert-base-chinese'
        super().__init__(*args, **kwargs)

class RobertaTextEncoder(TextEncoder):
    """RoBERTa文本编码器"""
    def __init__(self, *args, **kwargs):
        if 'model_name' not in kwargs:
            kwargs['model_name'] = 'hfl/chinese-roberta-wwm-ext'
        super().__init__(*args, **kwargs)


# 测试代码
if __name__ == "__main__":
    # 创建文本编码器
    encoder = TextEncoder(
        model_name='bert-base-chinese',
        embed_dim=768,
        freeze_encoder=False
    )
    
    # 测试文本
    test_texts = [
        "叶片出现黄色斑点，边缘枯萎",
        "果实表面有虫蛀痕迹"
    ]
    
    # 编码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = encoder.to(device)
    
    outputs = encoder.encode_text(test_texts, device=device)
    
    print(f"序列特征形状: {outputs['sequence_features'].shape}")
    print(f"池化特征形状: {outputs['pooled_features'].shape}")
    print(f"增强特征形状: {outputs['enhanced_features'].shape}")
    print(f"模型参数量: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M")