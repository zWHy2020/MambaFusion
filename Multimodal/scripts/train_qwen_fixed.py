# scripts/train_qwen_fixed.py
"""
修复版Qwen训练脚本 - 强化图像学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from tqdm import tqdm
import json
import numpy as np
import warnings
import gc
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.main_model_qwen import MultiModalPestDetectionWithQwen
from data import create_dataloaders
from utils import calculate_metrics, AverageMeter, plot_training_curves


class MemoryEfficientQwenTrainer:
    """显存优化的Qwen训练器"""
    
    def __init__(self, args):
        self.args = args
        self.setup_device()
        self.setup_dataloader()
        self.setup_model()
        self.setup_optimizer()
        self.setup_training()
    
    def setup_device(self):
        """设置设备"""
        if not torch.cuda.is_available():
            raise RuntimeError("需要GPU")
        
        self.device = torch.device('cuda:0')
        
        print("\n" + "="*60)
        print("显存优化的Qwen训练器")
        print("="*60)
        print(f"设备: {self.device}")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"总显存: {total_memory:.2f} GB")
    
    def setup_dataloader(self):
        """创建数据加载器"""
        print("\n创建数据加载器...")
        
        self.train_loader, self.val_loader, _ = create_dataloaders(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            num_workers=min(self.args.num_workers, 4), # 可以适当增加
            text_model_name='bert-base-chinese',
            use_hsi=self.args.use_hsi,
            use_augmentation=True  # <<< 核心修改 1：开启图像增强
        )
        
        self.num_classes = self.train_loader.dataset.num_classes
        
        print(f"✓ 训练集: {len(self.train_loader.dataset)} 样本")
        print(f"✓ 验证集: {len(self.val_loader.dataset)} 样本")
        print(f"✓ 类别数: {self.num_classes}")
        print(f"✓ Batch size: {self.args.batch_size}")
        print(f"✓ 梯度累积步数: {self.args.accumulation_steps}")
        print(f"✓ 有效batch size: {self.args.batch_size * self.args.accumulation_steps}")
    
    def setup_model(self):
        """创建模型"""
        print("\n创建Qwen增强模型...")
        
        self.model = MultiModalPestDetectionWithQwen(
            num_classes=self.num_classes,
            rgb_image_size=224,
            hsi_image_size=64,
            hsi_channels=224,
            text_model_name='bert-base-chinese',
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            fusion_layers=4,
            fusion_strategy='hierarchical',
            qwen_path=self.args.qwen_path,
            use_hsi=self.args.use_hsi,
            freeze_encoders=False,
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            num_query_tokens=32
        )

        self.model = self.model.to(self.device)
        
        print("✓ 模型创建完成")
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"模型加载后显存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
    
    def setup_optimizer(self):
        """设置优化器"""
        print("\n设置优化器...")
        
        encoder_params = []
        fusion_params = []
        adapter_params = []
        qwen_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'fusion' in name or 'alignment' in name:
                fusion_params.append(param)
            elif 'embedding_adapter' in name:
                adapter_params.append(param)
            elif 'qwen' in name:
                qwen_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
        
        param_groups = [
            # <<< 核心修改 2：提高编码器的学习率
            {'params': encoder_params, 'lr': self.args.lr * 0.8}, # 从0.1大幅提升
            {'params': fusion_params, 'lr': self.args.lr * 0.8}, # 融合层也一并提升
            {'params': adapter_params, 'lr': self.args.lr},
            {'params': qwen_params, 'lr': self.args.lr * 0.1}, # LLM部分保持低学习率
            {'params': classifier_params, 'lr': self.args.lr}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=self.args.lr * 0.01
        )
        
        print("✓ 优化器设置完成")
    
    def setup_training(self):
        # ... (这部分代码无需修改，此处省略)
        self.scaler = GradScaler() if self.args.use_amp else None
        log_dir = os.path.join(self.args.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = os.path.join(self.args.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.global_step = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    def train_epoch(self, epoch):
        # ... (这部分代码无需修改，此处省略)
        self.model.train()
        losses = AverageMeter()
        cls_losses = AverageMeter()
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', ncols=100)
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            try:
                rgb = batch['rgb_images'].to(self.device, non_blocking=True)
                hsi = batch['hsi_images'].to(self.device, non_blocking=True)
                text_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
                text_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                with autocast(dtype=torch.float16):
                    outputs = self.model(rgb, hsi, text_ids, text_mask, labels=labels)
                    loss = outputs['total_loss']
                    if isinstance(loss, tuple): loss = loss[0]
                    loss = loss.mean()
                    loss = loss / self.args.accumulation_steps
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.args.accumulation_steps == 0:
                    if self.args.clip_grad > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                real_loss = loss.item() * self.args.accumulation_steps
                losses.update(real_loss, rgb.size(0))
                cls_loss = outputs.get('cls_loss', loss)
                if isinstance(cls_loss, tuple): cls_loss = cls_loss[0]
                cls_losses.update(cls_loss.mean().item(), rgb.size(0))
                pbar.set_postfix({'loss': f'{losses.avg:.3f}', 'cls': f'{cls_losses.avg:.3f}'})
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    self.writer.add_scalar('Train/Loss', losses.avg, self.global_step)
                    self.writer.add_scalar('Train/ClsLoss', cls_losses.avg, self.global_step)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f'\n❌ Batch {batch_idx} 显存不足!')
                    torch.cuda.empty_cache()
                    gc.collect()
                    raise e
                else:
                    raise e
        torch.cuda.empty_cache()
        gc.collect()
        return losses.avg

    @torch.no_grad()
    def validate(self, epoch):
        # ... (这部分代码无需修改，此处省略)
        self.model.eval()
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        pbar = tqdm(self.val_loader, desc=f'Val {epoch}', ncols=100)
        for batch in pbar:
            try:
                rgb = batch['rgb_images'].to(self.device, non_blocking=True)
                hsi = batch['hsi_images'].to(self.device, non_blocking=True)
                text_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
                text_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                with autocast(dtype=torch.float16):
                    outputs = self.model(rgb, hsi, text_ids, text_mask, labels=labels)
                loss = outputs['total_loss']
                if isinstance(loss, tuple): loss = loss[0]
                loss = loss.mean()
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                losses.update(loss.item(), rgb.size(0))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.set_postfix({'loss': f'{losses.avg:.3f}'})
            except Exception as e:
                print(f'\n❌ 验证错误: {e}')
                continue
        torch.cuda.empty_cache()
        gc.collect()
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes=self.num_classes)
        return losses.avg, metrics['accuracy'], metrics['f1']

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        # ... (这部分代码无需修改，此处省略)
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'val_acc': val_acc, 'best_val_acc': self.best_val_acc}
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"   ✓ 保存最佳模型 (Acc: {val_acc:.4f})")

    def train(self):
        # ... (这部分代码无需修改，此处省略)
        print("\n" + "="*60 + "\n开始Qwen训练\n" + "="*60)
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}\n" + "-" * 60)
            try:
                train_loss = self.train_epoch(epoch)
                val_loss, val_acc, val_f1 = self.validate(epoch)
                self.scheduler.step()
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
                self.writer.add_scalar('Val/F1', val_f1, epoch)
                print(f"\n结果:\n  Train Loss: {train_loss:.4f}\n  Val Loss: {val_loss:.4f}\n  Val Acc: {val_acc:.4f}\n  Val F1: {val_f1:.4f}")
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                self.save_checkpoint(epoch, val_acc, is_best)
                if epoch - self.best_epoch >= self.args.early_stopping:
                    print(f"\n早停触发 (最佳epoch: {self.best_epoch})")
                    break
            except Exception as e:
                print(f"\n❌ Epoch {epoch} 失败: {e}")
                import traceback
                traceback.print_exc()
                break
        history_path = os.path.join(self.args.output_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        try:
            curves_path = os.path.join(self.args.output_dir, 'curves.png')
            plot_training_curves(self.history, save_path=curves_path)
        except:
            pass
        print("\n" + "="*60 + "\n✓ 训练完成!\n" + "="*60 + f"\n最佳准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})\n" + f"输出目录: {self.args.output_dir}\n" + "="*60)
        self.writer.close()

def parse_args():
    # ... (这部分代码无需修改，此处省略)
    parser = argparse.ArgumentParser(description='显存优化的Qwen训练脚本')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2, help='每个GPU的batch size（默认2）')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数（默认4）')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度（必须启用）')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='梯度累积步数（默认4，有效batch=2*4=8）')
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--qwen_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/qwen_fixed')
    parser.add_argument('--use_hsi', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def main():
    # ... (这部分代码无需修改，此处省略)
    args = parse_args()
    if not args.use_amp:
        print("警告: 自动启用混合精度以节省显存")
        args.use_amp = True
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    print("="*60 + "\n  显存优化的Qwen训练脚本\n" + "="*60 + f"\nBatch size: {args.batch_size}\n" + f"梯度累积: {args.accumulation_steps}\n" + f"有效batch: {args.batch_size * args.accumulation_steps}\n" + "="*60)
    try:
        trainer = MemoryEfficientQwenTrainer(args)
        trainer.train()
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()