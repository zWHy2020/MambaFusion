# training/loss.py
"""
自定义损失函数
包含多种适用于多模态学习的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class FocalLoss(nn.Module):
    """Focal Loss - 适用于类别不平衡问题"""
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """对比学习损失 (InfoNCE Loss)"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> torch.Tensor:
        # 归一化
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.matmul(features_a, features_b.t()) / self.temperature
        
        # 标签：对角线为正样本
        batch_size = features_a.shape[0]
        labels = torch.arange(batch_size, device=features_a.device)
        
        # 双向对比损失
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        
        return (loss_a + loss_b) / 2


class TripletLoss(nn.Module):
    """三元组损失"""
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class MultiModalAlignmentLoss(nn.Module):
    """多模态对齐损失"""
    def __init__(
        self,
        temperature: float = 0.07,
        kl_weight: float = 0.1
    ):
        super().__init__()
        self.contrastive = ContrastiveLoss(temperature)
        self.kl_weight = kl_weight
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        probs_a: Optional[torch.Tensor] = None,
        probs_b: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        contrastive_loss = self.contrastive(features_a, features_b)
        
        total_loss = contrastive_loss
        kl_loss = torch.tensor(0.0, device=features_a.device)
        
        if probs_a is not None and probs_b is not None:
            kl_loss = F.kl_div(
                F.log_softmax(probs_a, dim=-1),
                F.softmax(probs_b, dim=-1),
                reduction='batchmean'
            )
            total_loss = total_loss + self.kl_weight * kl_loss
        
        return {
            'total': total_loss,
            'contrastive': contrastive_loss,
            'kl': kl_loss
        }


class MultiTaskLoss(nn.Module):
    """多任务学习损失 - 自动平衡"""
    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: list) -> torch.Tensor:
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss


class CenterLoss(nn.Module):
    """Center Loss - 增强类内紧凑性"""
    def __init__(self, num_classes: int, feat_dim: int, lambda_c: float = 0.003):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        loss = (features - centers_batch).pow(2).sum() / batch_size
        
        return loss * self.lambda_c


class MultiModalLoss(nn.Module):
    """完整的多模态学习损失函数"""
    def __init__(
        self,
        num_classes: int,
        use_focal: bool = False,
        use_label_smoothing: bool = True,
        label_smoothing: float = 0.1,
        alignment_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        focal_alpha: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.alignment_weight = alignment_weight
        
        # 分类损失
        if use_focal:
            self.cls_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif use_label_smoothing:
            self.cls_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.cls_criterion = nn.CrossEntropyLoss()
        
        # 对齐损失
        self.alignment_criterion = ContrastiveLoss(temperature=contrastive_temperature)
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rgb_features: Optional[torch.Tensor] = None,
        hsi_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 分类损失
        cls_loss = self.cls_criterion(logits, labels)
        
        total_loss = cls_loss
        losses_dict = {'cls_loss': cls_loss}
        
        # 对齐损失
        if rgb_features is not None and text_features is not None:
            align_loss_rt = self.alignment_criterion(rgb_features, text_features)
            total_loss += self.alignment_weight * align_loss_rt
            losses_dict['align_rgb_text'] = align_loss_rt
        
        if hsi_features is not None and text_features is not None:
            align_loss_ht = self.alignment_criterion(hsi_features, text_features)
            total_loss += self.alignment_weight * align_loss_ht
            losses_dict['align_hsi_text'] = align_loss_ht
        
        if rgb_features is not None and hsi_features is not None:
            align_loss_rh = self.alignment_criterion(rgb_features, hsi_features)
            total_loss += self.alignment_weight * align_loss_rh
            losses_dict['align_rgb_hsi'] = align_loss_rh
        
        losses_dict['total_loss'] = total_loss
        
        return losses_dict