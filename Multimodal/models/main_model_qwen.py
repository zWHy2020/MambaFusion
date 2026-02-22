# models/main_model_qwen.py
"""
集成 Qwen2.5-7B 的完整多模态模型
采用参数高效微调策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import os
import sys

from models.encoders.rgb_encoder import RGBEncoder
from models.encoders.hsi_encoder import HSIEncoder
from models.encoders.text_encoder import TextEncoder
from models.fusion.cross_modal_alignment import CrossModalAlignmentModule
from models.fusion.multi_modal_fusion import MultiModalFusionModule
from models.adapters.embedding_adapter import MultiModalEmbeddingAdapter, ModalityFusion
from models.adapters.lora import apply_lora_to_model


class MultiModalPestDetectionWithQwen(nn.Module):
    """
    集成 Qwen2.5-7B 的多模态病虫害识别系统
    
    架构:
    1. 多模态编码器 (RGB + HSI + Text)
    2. 跨模态对齐
    3. 多模态融合
    4. Embedding 适配器
    5. Qwen2.5-7B (冻结大部分参数)
    6. 分类头
    """
    def __init__(
        self,
        # 任务配置
        num_classes: int = 50,
        
        # 编码器配置
        rgb_image_size: int = 224,
        hsi_image_size: int = 64,
        hsi_channels: int = 224,
        text_model_name: str = "bert-base-chinese",
        
        # 通用配置
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        
        # 融合配置
        fusion_layers: int = 4,
        fusion_strategy: str = 'hierarchical',
        
        # Qwen 配置
        qwen_path: str = "./models/qwen2.5-7b",
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        num_query_tokens: int = 32,
        use_hsi: bool = True,
        
        # 微调策略
        freeze_encoders: bool = False,
        trainable_layers: List[str] = ['embedding', 'lm_head'],  # Qwen 中可训练的层
    ):
        super().__init__()
        self.use_hsi = use_hsi
        self.num_classes = num_classes
        self.qwen_path = qwen_path
        
        print("\n" + "="*60)
        print("初始化 Qwen 增强多模态模型")
        print("="*60)
        
        # ============ 编码器模块 ============
        print("1. 加载编码器...")
        
        self.rgb_encoder = RGBEncoder(
            image_size=rgb_image_size,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.hsi_encoder = HSIEncoder(
            hsi_channels=hsi_channels,
            spatial_size=hsi_image_size,
            embed_dim=embed_dim,
            depth=6,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
            freeze_encoder=freeze_encoders,
            dropout=dropout
        )
        
        if freeze_encoders:
            self._freeze_encoders()
        
        print("   ✓ 编码器加载完成")
        
        # ============ 跨模态对齐 ============
        print("2. 初始化跨模态对齐...")
        
        self.alignment = CrossModalAlignmentModule(
            rgb_dim=embed_dim,
            hsi_dim=embed_dim,
            use_hsi=self.use_hsi,
            text_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        print("   ✓ 对齐模块初始化完成")
        
        # ============ 多模态融合 ============
        print("3. 初始化多模态融合...")
        
        self.fusion = MultiModalFusionModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=fusion_layers,
            dropout=dropout,
            fusion_strategy=fusion_strategy
        )
        
        print("   ✓ 融合模块初始化完成")
        
        # ============ Qwen2.5-7B ============
        print("4. 加载 Qwen2.5-7B...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 加载 Qwen
            self.qwen = AutoModelForCausalLM.from_pretrained(
                qwen_path,
                torch_dtype=torch.float16,
                device_map=None,  # 手动管理设备
                trust_remote_code=True
            )
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                qwen_path,
                trust_remote_code=True
            )
            
            qwen_hidden_size = self.qwen.config.hidden_size
            print(f"   ✓ Qwen 加载成功 (hidden_size={qwen_hidden_size})")
            
            # 冻结 Qwen 大部分参数
            self._freeze_qwen(trainable_layers)
            
            # 应用 LoRA
            if use_lora:
                print("   5. 应用 LoRA...")
                self.qwen = apply_lora_to_model(
                    self.qwen,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=dropout,
                    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
                )
                print(f"   ✓ LoRA 应用完成 (rank={lora_rank})")
            
        except Exception as e:
            print(f"   ✗ Qwen 加载失败: {e}")
            raise
        
        # ============ Embedding 适配器 ============
        print("6. 初始化 Embedding 适配器...")
        
        self.embedding_adapter = MultiModalEmbeddingAdapter(
            multimodal_dim=embed_dim,
            llm_hidden_size=qwen_hidden_size,
            num_query_tokens=num_query_tokens,
            num_layers=4,
            num_heads=num_heads,
            dropout=dropout
        )
        
        print("   ✓ Embedding 适配器初始化完成")
        
        # ============ 分类头 ============
        print("7. 初始化分类头...")
        
        # 基于 Qwen 的隐藏状态分类
        self.classifier = nn.Sequential(
            nn.Linear(qwen_hidden_size, qwen_hidden_size // 2),
            nn.LayerNorm(qwen_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(qwen_hidden_size // 2, num_classes)
        )
        
        print("   ✓ 分类头初始化完成")
        
        # ============ 统计参数 ============
        self._print_trainable_parameters()
        
        print("="*60)
        print("✓ 模型初始化完成\n")
    
    def _freeze_encoders(self):
        """冻结编码器"""
        for param in self.rgb_encoder.parameters():
            param.requires_grad = False
        for param in self.hsi_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False
    
    def _freeze_qwen(self, trainable_layers: List[str]):
        """冻结 Qwen 大部分参数"""
        # 首先冻结所有参数
        for param in self.qwen.parameters():
            param.requires_grad = False
        
        # 解冻指定层
        trainable_params = 0
        for name, param in self.qwen.named_parameters():
            for layer_name in trainable_layers:
                if layer_name in name:
                    param.requires_grad = True
                    trainable_params += param.numel()
                    break
        
        print(f"   ✓ Qwen 冻结完成 (可训练参数: {trainable_params/1e6:.2f}M)")
    
    def _print_trainable_parameters(self):
        """打印可训练参数统计"""
        total = 0
        trainable = 0
        
        for param in self.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        
        print(f"\n参数统计:")
        print(f"  总参数: {total/1e6:.2f}M")
        print(f"  可训练: {trainable/1e6:.2f}M ({100*trainable/total:.2f}%)")
    
    def forward(
        self,
        rgb_images: torch.Tensor,
        hsi_images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            rgb_images: [B, 3, H, W]
            hsi_images: [B, C, H, W]
            text_input_ids: [B, L]
            text_attention_mask: [B, L]
            labels: [B]
            return_features: 是否返回中间特征
        """
        B = rgb_images.shape[0]
        
        # ============ 1. 特征编码 ============
        rgb_cls, rgb_patches = self.rgb_encoder(rgb_images)
        rgb_features = torch.cat([rgb_cls.unsqueeze(1), rgb_patches], dim=1)
        if self.use_hsi:
            hsi_cls, hsi_patches = self.hsi_encoder(hsi_images)
            hsi_features = torch.cat([hsi_cls.unsqueeze(1), hsi_patches], dim=1)
        else:
            hsi_features = torch.zeros_like(rgb_features)
        
        text_outputs = self.text_encoder(text_input_ids, text_attention_mask)
        text_features = text_outputs['sequence_features']
        
        # ============ 2. 跨模态对齐 ============
        alignment_outputs = self.alignment(
            rgb_features, hsi_features, text_features, text_attention_mask
        )
        
        rgb_aligned = alignment_outputs['rgb_aligned']
        hsi_aligned = alignment_outputs['hsi_aligned']
        text_aligned = alignment_outputs['text_aligned']
        alignment_loss = alignment_outputs['alignment_loss']
        
        # ============ 3. 多模态融合 ============
        fusion_outputs = self.fusion(
            rgb_aligned, hsi_aligned, text_aligned, return_intermediate=False
        )
        
        fused_features = fusion_outputs['fused_features']  # [B, D]
        
        # 扩展维度以适配 embedding adapter
        fused_features_expanded = fused_features.unsqueeze(1).expand(-1, 64, -1)
        
        # ============ 4. 转换为 LLM Embeddings ============
        llm_embeddings = self.embedding_adapter(fused_features_expanded)  # [B, Q, D_llm]
        
        # ============ 5. Qwen 处理 ============
        # 使用 fp16 推理
        with torch.cuda.amp.autocast(dtype=torch.float16):
            qwen_outputs = self.qwen(
                inputs_embeds=llm_embeddings,
                output_hidden_states=True,
                return_dict=True
            )
        
        # 获取最后一层隐藏状态
        last_hidden = qwen_outputs.hidden_states[-1]  # [B, Q, D_llm]
        
        # 池化
        pooled = last_hidden.mean(dim=1)  # [B, D_llm]
        
        # ============ 6. 分类 ============
        logits = self.classifier(pooled.float())  # [B, num_classes]
        
        # ============ 7. 计算损失 ============
        total_loss = None
        cls_loss = None
        
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
            total_loss = cls_loss + 0.1 * alignment_loss
        
        # ============ 8. 返回结果 ============
        outputs = {
            'logits': logits,
            'pooled_features': pooled,
            'alignment_loss': alignment_loss,
            'cls_loss': cls_loss if cls_loss is not None else torch.tensor(0.0),
            'total_loss': total_loss if total_loss is not None else torch.tensor(0.0)
        }
        
        if return_features:
            outputs.update({
                'rgb_features': rgb_features,
                'hsi_features': hsi_features,
                'text_features': text_features,
                'fused_features': fused_features,
                'llm_embeddings': llm_embeddings
            })
        
        return outputs


# 测试代码
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n测试 Qwen 增强多模态模型")
    
    # 检查 Qwen 路径
    qwen_path = "./models/qwen2.5-7b"
    if not os.path.exists(qwen_path):
        print(f"错误: Qwen 路径不存在: {qwen_path}")
        print("请确保已下载 Qwen2.5-7B 模型")
        exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = MultiModalPestDetectionWithQwen(
        num_classes=50,
        qwen_path=qwen_path,
        use_lora=True,
        lora_rank=8
    )
    
    model = model.to(device)
    model.eval()
    
    # 测试输入
    B = 2
    rgb = torch.randn(B, 3, 224, 224).to(device)
    hsi = torch.randn(B, 224, 64, 64).to(device)
    text_ids = torch.randint(0, 1000, (B, 50)).to(device)
    text_mask = torch.ones(B, 50).to(device)
    labels = torch.randint(0, 50, (B,)).to(device)
    
    print("\n开始测试前向传播...")
    
    with torch.no_grad():
        outputs = model(rgb, hsi, text_ids, text_mask, labels=labels)
    
    print("\n测试结果:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Total Loss: {outputs['total_loss'].item():.4f}")
    print(f"  显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    print("\n✓ 测试通过!")