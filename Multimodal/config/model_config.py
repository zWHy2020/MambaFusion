# config/model_config.py
"""
模型配置系统
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import json

@dataclass
class EncoderConfig:
    """编码器配置"""
    # RGB编码器
    rgb_image_size: int = 224
    rgb_patch_size: int = 16
    rgb_depth: int = 12
    
    # HSI编码器
    hsi_image_size: int = 64
    hsi_channels: int = 224
    hsi_depth: int = 6
    
    # 文本编码器
    text_model_name: str = "bert-base-chinese"
    max_text_length: int = 128
    freeze_text_encoder: bool = False

@dataclass
class FusionConfig:
    """融合配置"""
    fusion_strategy: str = "hierarchical"  # concat, gated, hierarchical
    fusion_layers: int = 4
    alignment_weight: float = 0.1
    contrastive_temperature: float = 0.07

@dataclass
class LLMConfig:
    """大语言模型配置"""
    use_llm: bool = False
    llm_model_name: Optional[str] = None
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16
    num_query_tokens: int = 32

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # 优化器
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    
    # 正则化
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    
    # 训练技巧
    use_amp: bool = False
    gradient_accumulation_steps: int = 1
    
    # 数据增强
    use_augmentation: bool = True
    
    # 早停
    early_stopping_patience: int = 20
    
    # 保存
    save_interval: int = 10
    log_interval: int = 50

@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = "./data"
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

@dataclass
class ModelConfig:
    """完整的模型配置"""
    # 基本信息
    model_name: str = "MultiModalPestDetection"
    num_classes: int = 50
    
    # 通用配置
    embed_dim: int = 768
    num_heads: int = 12
    
    # 子配置
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # 其他
    seed: int = 42
    device: str = "cuda"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置"""
        encoder_config = EncoderConfig(**config_dict.get('encoder', {}))
        fusion_config = FusionConfig(**config_dict.get('fusion', {}))
        llm_config = LLMConfig(**config_dict.get('llm', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        return cls(
            model_name=config_dict.get('model_name', 'MultiModalPestDetection'),
            num_classes=config_dict.get('num_classes', 50),
            embed_dim=config_dict.get('embed_dim', 768),
            num_heads=config_dict.get('num_heads', 12),
            encoder=encoder_config,
            fusion=fusion_config,
            llm=llm_config,
            training=training_config,
            data=data_config,
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda')
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ModelConfig':
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'encoder': self.encoder.__dict__,
            'fusion': self.fusion.__dict__,
            'llm': self.llm.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'seed': self.seed,
            'device': self.device
        }
    
    def save_yaml(self, yaml_path: str):
        """保存为YAML文件"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, json_path: str):
        """保存为JSON文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

# 预定义配置
def get_baseline_config() -> ModelConfig:
    """基础配置"""
    return ModelConfig()

def get_small_config() -> ModelConfig:
    """小型配置（适合显存受限）"""
    config = ModelConfig()
    config.embed_dim = 512
    config.num_heads = 8
    config.encoder.rgb_depth = 6
    config.encoder.hsi_depth = 4
    config.fusion.fusion_layers = 2
    config.training.batch_size = 32
    return config

def get_large_config() -> ModelConfig:
    """大型配置（高性能）"""
    config = ModelConfig()
    config.embed_dim = 1024
    config.num_heads = 16
    config.encoder.rgb_depth = 16
    config.encoder.hsi_depth = 8
    config.fusion.fusion_layers = 6
    config.training.batch_size = 8
    return config

def get_llm_config(llm_name: str = "Qwen/Qwen2.5-7B-Instruct") -> ModelConfig:
    """使用LLM的配置"""
    config = ModelConfig()
    config.llm.use_llm = True
    config.llm.llm_model_name = llm_name
    config.llm.use_lora = True
    config.training.batch_size = 4
    config.training.use_amp = True
    return config