# scripts/evaluate_qwen.py
"""
Qwen 增强模型评估脚本
"""

import torch
import torch.nn.functional as F
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.main_model_qwen import MultiModalPestDetectionWithQwen
from data import create_dataloaders
from utils import (calculate_metrics, plot_confusion_matrix, 
                   plot_per_class_metrics, plot_feature_distribution)


class QwenEvaluator:
    """Qwen 模型评估器"""
    
    def __init__(self, args):
        self.args = args
        self.setup_device()
        self.load_checkpoint()
        self.setup_dataloader()
        self.setup_model()
        
    def setup_device(self):
        """设置设备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {self.device}")
        
    def load_checkpoint(self):
        """加载检查点"""
        print(f"加载检查点: {self.args.checkpoint}")
        
        self.checkpoint = torch.load(
            self.args.checkpoint,
            map_location='cpu'
        )
        
        print("✓ 检查点加载成功")
        if 'epoch' in self.checkpoint:
            print(f"  Epoch: {self.checkpoint['epoch']}")
        if 'val_acc' in self.checkpoint:
            print(f"  验证准确率: {self.checkpoint['val_acc']:.4f}")
    
    def setup_dataloader(self):
        """创建数据加载器"""
        print("\n创建数据加载器...")
        
        _, _, self.test_loader = create_dataloaders(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            text_model_name='bert-base-chinese',
            use_augmentation=False,
            use_hsi=self.args.use_hsi
        )
        
        self.num_classes = self.test_loader.dataset.num_classes
        self.class_names = list(self.test_loader.dataset.class_to_idx.keys())
        
        print(f"✓ 测试集: {len(self.test_loader.dataset)} 样本")
        print(f"✓ 类别数: {self.num_classes}")
    
    def setup_model(self):
        """创建并加载模型"""
        print("\n创建模型...")
        
        self.model = MultiModalPestDetectionWithQwen(
            num_classes=self.num_classes,
            qwen_path=self.args.qwen_path,
            use_lora=True,
            use_hsi=self.args.use_hsi
        )
        
        # 加载权重
        state_dict = self.checkpoint['model_state_dict']
        
        # 处理 DataParallel 的键名
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✓ 模型加载完成")
    
    @torch.no_grad()
    def evaluate(self):
        """执行评估"""
        print("\n" + "="*60)
        print("开始评估")
        print("="*60)
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        
        pbar = tqdm(self.test_loader, desc='评估', ncols=100)
        
        for batch in pbar:
            rgb = batch['rgb_images'].to(self.device)
            hsi = batch['hsi_images'].to(self.device)
            text_ids = batch['text_input_ids'].to(self.device)
            text_mask = batch['text_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                rgb, hsi, text_ids, text_mask,
                labels=None,
                return_features=True
            )
            
            logits = outputs['logits']
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if 'pooled_features' in outputs:
                all_features.extend(outputs['pooled_features'].cpu().numpy())
            
            # 更新进度
            acc = np.mean(np.array(all_preds) == np.array(all_labels))
            pbar.set_postfix({'acc': f'{acc:.3f}'})
        
        # 转换
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        results = self.compute_metrics(all_preds, all_labels, all_probs)
        
        # 可视化
        self.visualize_results(all_preds, all_labels, results)
        
        # 特征可视化
        if len(all_features) > 0:
            self.visualize_features(np.array(all_features), all_labels)
        
        # 保存结果
        self.save_results(results, all_preds, all_labels)
        
        return results
    
    def compute_metrics(self, preds, labels, probs):
        """计算指标"""
        print("\n计算指标...")
        
        metrics = calculate_metrics(labels, preds, probs, self.num_classes)
        
        print(f"\n总体性能:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        
        return {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
    
    def visualize_results(self, preds, labels, results):
        """可视化"""
        print("\n生成可视化...")
        
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 混淆矩阵
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            labels, preds,
            class_names=self.class_names,
            save_path=cm_path,
            normalize=True
        )
        print(f"  ✓ 混淆矩阵: {cm_path}")
    
    def visualize_features(self, features, labels):
        """特征可视化"""
        print("\n特征可视化...")
        
        output_dir = self.args.output_dir
        
        # t-SNE
        tsne_path = os.path.join(output_dir, 'tsne.png')
        plot_feature_distribution(
            features, labels,
            method='tsne',
            save_path=tsne_path
        )
        print(f"  ✓ t-SNE: {tsne_path}")
    
    def save_results(self, results, preds, labels):
        """保存结果"""
        output_dir = self.args.output_dir
        
        # JSON
        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 预测
        np.save(os.path.join(output_dir, 'predictions.npy'), preds)
        np.save(os.path.join(output_dir, 'labels.npy'), labels)
        
        print(f"\n✓ 结果保存到: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--qwen_path', type=str, default='./models/qwen2.5-7b')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--use_hsi', action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    
    evaluator = QwenEvaluator(args)
    results = evaluator.evaluate()
    
    print("\n" + "="*60)
    print("✓ 评估完成!")
    print("="*60)
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()