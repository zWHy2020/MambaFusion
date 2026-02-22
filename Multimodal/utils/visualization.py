# utils/visualization.py
"""
Visualization tools
Includes functions for confusion matrix, training curves, attention maps, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Dict
import torch
from sklearn.metrics import confusion_matrix
import os

# The following lines for Chinese font settings are no longer needed and have been removed.
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    normalize: bool = True,
    title: str = 'Confusion Matrix'
):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Normalized' if normalize else 'Count'}
    )
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """Plots training and validation curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Loss Curve', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    if 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Accuracy Curve', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 score curve
    if 'val_f1' in history:
        axes[2].plot(epochs, history['val_f1'], 'g-', label='Validation F1', linewidth=2)
    if 'train_f1' in history:
        axes[2].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[2].set_title('F1 Score Curve', fontsize=14)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.close()


def plot_attention_map(
    attention: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    title: str = 'Attention Map'
):
    """Visualizes attention weights."""
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()
    
    num_heads = attention.shape[0]
    
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(num_heads):
        im = axes[i].imshow(attention[i], cmap='viridis', aspect='auto')
        axes[i].set_title(f'Head {i+1}', fontsize=10)
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[i])
    
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention map saved to: {save_path}")
    
    plt.close()


def plot_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    title: str = 'Feature Distribution'
):
    """Visualizes feature distribution using dimensionality reduction."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    
    features_2d = reducer.fit_transform(features)
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='tab20',
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter, label='Class')
    plt.title(f'{title} ({method.upper()})', fontsize=16)
    plt.xlabel(f'{method.upper()} 1', fontsize=12)
    plt.ylabel(f'{method.upper()} 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature distribution map saved to: {save_path}")
    
    plt.close()


def plot_per_class_metrics(
    metrics: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """Plots performance metrics for each class."""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'precision' in metrics:
        ax.bar(x - width, metrics['precision'], width, label='Precision', alpha=0.8)
    if 'recall' in metrics:
        ax.bar(x, metrics['recall'], width, label='Recall', alpha=0.8)
    if 'f1' in metrics:
        ax.bar(x + width, metrics['f1'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics chart saved to: {save_path}")
    
    plt.close()


def plot_learning_rate_schedule(
    lr_history: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """Plots the learning rate schedule."""
    plt.figure(figsize=figsize)
    plt.plot(lr_history, linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate curve saved to: {save_path}")
    
    plt.close()


def visualize_predictions(
    images: np.ndarray,
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10),
    num_samples: int = 12
):
    """Visualizes model predictions on sample images."""
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f'True: {true_labels[i]}\nPred: {pred_labels[i]}\nConf: {confidences[i]:.2f}'
        axes[i].set_title(title, fontsize=10, color=color)
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Prediction Visualization', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction results saved to: {save_path}")
    
    plt.close()