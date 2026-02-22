# data/dataset.py
"""
多模态病虫害数据集 - 修改版
增加了文本遮蔽功能，强制模型学习图像特征
"""

import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class PestDataset(Dataset):
    """
    多模态病虫害数据集
    """
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        rgb_transform = None,
        hsi_transform = None,
        text_model_name: str = 'bert-base-chinese',
        max_text_length: int = 128,
        use_hsi: bool = True,
        use_augmentation: bool = True,
        use_text_dropout: bool = True, # 新增参数，用于控制是否启用文本遮蔽
        text_dropout_ratio: float = 0.2 # 新增参数，设置文本遮蔽的概率
    ):
        """
        Args:
            (...)
            use_text_dropout: 是否在训练时启用文本遮蔽
            text_dropout_ratio: 文本遮蔽的概率
        """
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.max_text_length = max_text_length
        self.use_hsi = use_hsi
        self.use_text_dropout = use_text_dropout
        self.text_dropout_ratio = text_dropout_ratio
        
        # 加载数据索引
        self.samples = self._load_samples()
        
        # 加载类别映射
        with open(os.path.join(data_root, 'class_mapping.json'), 'r', encoding='utf-8') as f:
            self.class_to_idx = json.load(f)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.num_classes = len(self.class_to_idx)
        
        # 文本tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # 设置RGB图像变换
        if rgb_transform is None:
            self.rgb_transform = self._get_default_rgb_transform(use_augmentation)
        else:
            self.rgb_transform = rgb_transform
        
        # 设置HSI图像变换
        if hsi_transform is None:
            self.hsi_transform = self._get_default_hsi_transform(use_augmentation)
        else:
            self.hsi_transform = hsi_transform
    
    def _load_samples(self) -> List[Dict]:
        """加载数据样本列表"""
        split_file = os.path.join(self.data_root, f'{self.split}.json')
        with open(split_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        return samples
    
    def _get_default_rgb_transform(self, use_augmentation: bool):
        """获取默认的RGB图像变换"""
        if use_augmentation and self.split == 'train':
            transform = A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.GaussNoise(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        return transform
    
    def _get_default_hsi_transform(self, use_augmentation: bool):
        """获取默认的高光谱图像变换"""
        return None 
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        """
        sample = self.samples[idx]
        
        # 加载RGB图像
        rgb_path = os.path.join(self.data_root, sample['rgb_path'])
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_image = np.array(rgb_image)
        
        if self.rgb_transform:
            rgb_image = self.rgb_transform(image=rgb_image)['image']
        else:
            rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        
        # 加载高光谱图像
        if self.use_hsi:
            hsi_path = os.path.join(self.data_root, sample['hsi_path'])
            hsi_image = np.load(hsi_path)
            hsi_image = (hsi_image - hsi_image.min()) / (hsi_image.max() - hsi_image.min() + 1e-8)
            hsi_image = torch.from_numpy(hsi_image).float()
            if hsi_image.dim() == 3 and hsi_image.shape[-1] > hsi_image.shape[0]:
                hsi_image = hsi_image.permute(2,0,1)
        else:
            hsi_image = torch.zeros((1, 224, 64, 64), dtype=torch.float32)[0] # 保持维度一致
        
        
        # 加载文本描述
        text = sample['description']

        # --- 核心修改：实现文本遮蔽 ---
        if self.split == 'train' and self.use_text_dropout and random.random() < self.text_dropout_ratio:
            text = "一张需要分析的植物叶片图片。"
        
        # Tokenize文本
        text_encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        # 获取标签
        label = self.class_to_idx[sample['class']]
        
        return {
            'rgb_image': rgb_image,
            'hsi_image': hsi_image,
            'text_input_ids': text_encoded['input_ids'].squeeze(0),
            'text_attention_mask': text_encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'class_name': sample['class'],
            'sample_id': sample.get('id', idx)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数
    """
    rgb_images = torch.stack([item['rgb_image'] for item in batch])
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    hsi_images = torch.stack([item['hsi_image'] for item in batch])
    
    return {
        'rgb_images': rgb_images,
        'hsi_images': hsi_images,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'labels': labels,
        'class_names': [item['class_name'] for item in batch],
        'sample_ids': [item['sample_id'] for item in batch]
    }


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    text_model_name: str = 'bert-base-chinese',
    use_augmentation: bool = True,
    use_hsi: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    """
    train_dataset = PestDataset(
        data_root=data_root,
        split='train',
        text_model_name=text_model_name,
        use_augmentation=use_augmentation,
        use_hsi=use_hsi,
        use_text_dropout=True, # 确保开启
        text_dropout_ratio=0.2 # 20%的概率遮蔽文本
    )
    
    val_dataset = PestDataset(
        data_root=data_root,
        split='val',
        text_model_name=text_model_name,
        use_augmentation=False,
        use_hsi=use_hsi,
        use_text_dropout=False # 验证集不使用
    )
    
    test_dataset = PestDataset(
        data_root=data_root,
        split='test',
        text_model_name=text_model_name,
        use_hsi=use_hsi,
        use_augmentation=False,
        use_text_dropout=False # 测试集不使用
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader