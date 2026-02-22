# utils/metrics.py
"""
评估指标计算工具
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Optional
import torch


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签 [N]
        y_pred: 预测标签 [N]
        y_prob: 预测概率 [N, C] (可选)
        num_classes: 类别数
        
    Returns:
        包含各种指标的字典
    """
    metrics = {}
    
    # 基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 多类别指标（macro平均）
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 每个类别的指标
    metrics['precision_per_class'] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics['recall_per_class'] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics['f1_per_class'] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # AUC (如果提供了概率)
    if y_prob is not None:
        try:
            # 将标签转换为one-hot编码
            if num_classes is None:
                num_classes = len(np.unique(y_true))
            
            y_true_onehot = np.zeros((len(y_true), num_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            
            # 计算macro AUC
            metrics['auc_macro'] = roc_auc_score(
                y_true_onehot, y_prob, average='macro', multi_class='ovr'
            )
            
            # 计算weighted AUC
            metrics['auc_weighted'] = roc_auc_score(
                y_true_onehot, y_prob, average='weighted', multi_class='ovr'
            )
        except:
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0
    
    return metrics


def topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple = (1, 5)
) -> List[float]:
    """
    计算top-k准确率
    
    Args:
        output: 模型输出 [B, C]
        target: 真实标签 [B]
        topk: k值元组
        
    Returns:
        top-k准确率列表
    """
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    
    return res


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    计算每个类别的详细指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        
    Returns:
        每个类别的指标字典
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    return report


def calculate_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平衡准确率（对不平衡数据集更有意义）
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    return np.mean(per_class)


class MetricsTracker:
    """指标追踪器，用于训练过程中记录指标"""
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def update(self, metrics: Dict, phase: str = 'train'):
        """更新指标"""
        for key, value in metrics.items():
            history_key = f"{phase}_{key}"
            if history_key not in self.history:
                self.history[history_key] = []
            self.history[history_key].append(value)
    
    def get_best(self, metric: str = 'val_acc', mode: str = 'max'):
        """获取最佳指标值"""
        if metric not in self.history:
            return None
        
        values = self.history[metric]
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return {
            'value': values[best_idx],
            'epoch': best_idx + 1
        }
    
    def save(self, path: str):
        """保存指标历史"""
        import json
        with open(path, 'w') as f:
            # 转换numpy类型为python类型
            history_serializable = {}
            for key, values in self.history.items():
                history_serializable[key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in values
                ]
            json.dump(history_serializable, f, indent=2)
    
    def load(self, path: str):
        """加载指标历史"""
        import json
        with open(path, 'r') as f:
            self.history = json.load(f)


# 测试代码
if __name__ == "__main__":
    # 模拟数据
    y_true = np.random.randint(0, 10, 1000)
    y_pred = np.random.randint(0, 10, 1000)
    y_prob = np.random.rand(1000, 10)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred, y_prob, num_classes=10)
    
    print("评估指标:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC (macro): {metrics['auc_macro']:.4f}")
    
    # 测试AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i * 0.1)
    print(f"\nAverage: {meter.avg:.4f}")
