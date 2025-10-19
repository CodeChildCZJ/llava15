"""
注意力权重可视化工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any
import os


class AttentionVisualizer:
    """注意力权重可视化器"""
    
    def __init__(self, save_dir: str = "./attention_visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_attention_heatmap(
        self, 
        attention_weights: torch.Tensor,
        layer_idx: int,
        head_idx: Optional[int] = None,
        save_path: Optional[str] = None,
        title: str = "Attention Weights"
    ) -> str:
        """
        可视化注意力权重热力图
        
        Args:
            attention_weights: 注意力权重 [batch, heads, seq_len, seq_len] 或 [heads, seq_len, seq_len]
            layer_idx: 层索引
            head_idx: 头索引，如果为None则平均所有头
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            str: 保存的文件路径
        """
        # 处理注意力权重维度
        if attention_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
            if head_idx is not None:
                attn = attention_weights[0, head_idx]  # 取第一个batch的指定头
            else:
                attn = attention_weights[0].mean(dim=0)  # 平均所有头
        elif attention_weights.dim() == 3:  # [heads, seq_len, seq_len]
            if head_idx is not None:
                attn = attention_weights[head_idx]
            else:
                attn = attention_weights.mean(dim=0)
        else:
            raise ValueError(f"Unsupported attention weights dimension: {attention_weights.dim()}")
        
        # 转换为numpy
        attn_np = attn.detach().cpu().numpy()
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attn_np, 
            cmap='Blues', 
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False
        )
        
        head_info = f" (Head {head_idx})" if head_idx is not None else " (All Heads)"
        plt.title(f"{title} - Layer {layer_idx}{head_info}")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        
        # 保存图片
        if save_path is None:
            head_suffix = f"_head{head_idx}" if head_idx is not None else "_all_heads"
            save_path = os.path.join(
                self.save_dir, 
                f"attention_layer{layer_idx}{head_suffix}.png"
            )
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_attention_to_visual_tokens(
        self,
        attention_weights: torch.Tensor,
        visual_token_start: int,
        visual_token_end: int,
        layer_idx: int,
        head_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        可视化对视觉token的注意力
        
        Args:
            attention_weights: 注意力权重
            visual_token_start: 视觉token开始位置
            visual_token_end: 视觉token结束位置
            layer_idx: 层索引
            head_idx: 头索引
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        # 处理注意力权重
        if attention_weights.dim() == 4:
            if head_idx is not None:
                attn = attention_weights[0, head_idx]
            else:
                attn = attention_weights[0].mean(dim=0)
        elif attention_weights.dim() == 3:
            if head_idx is not None:
                attn = attention_weights[head_idx]
            else:
                attn = attention_weights.mean(dim=0)
        else:
            raise ValueError(f"Unsupported attention weights dimension: {attention_weights.dim()}")
        
        # 提取对视觉token的注意力
        visual_attention = attn[:, visual_token_start:visual_token_end]  # [seq_len, visual_tokens]
        
        # 计算每个位置对视觉token的平均注意力
        avg_attention_to_visual = visual_attention.mean(dim=1)  # [seq_len]
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 注意力权重热力图
        sns.heatmap(
            visual_attention.detach().cpu().numpy(),
            cmap='Blues',
            cbar=True,
            ax=ax1,
            xticklabels=False,
            yticklabels=False
        )
        ax1.set_title(f"Attention to Visual Tokens - Layer {layer_idx}")
        ax1.set_xlabel("Visual Token Position")
        ax1.set_ylabel("Query Position")
        
        # 2. 平均注意力分数
        positions = range(len(avg_attention_to_visual))
        ax2.plot(positions, avg_attention_to_visual.detach().cpu().numpy(), 'b-', linewidth=2)
        ax2.axvspan(visual_token_start, visual_token_end-1, alpha=0.3, color='red', label='Visual Tokens')
        ax2.set_title("Average Attention to Visual Tokens")
        ax2.set_xlabel("Sequence Position")
        ax2.set_ylabel("Attention Score")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            head_suffix = f"_head{head_idx}" if head_idx is not None else "_all_heads"
            save_path = os.path.join(
                self.save_dir,
                f"visual_attention_layer{layer_idx}{head_suffix}.png"
            )
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_multi_layer_attention(
        self,
        attention_weights_list: List[torch.Tensor],
        layer_indices: List[int],
        save_path: Optional[str] = None,
        max_layers: int = 4
    ) -> str:
        """
        可视化多层注意力权重
        
        Args:
            attention_weights_list: 多层注意力权重列表
            layer_indices: 层索引列表
            save_path: 保存路径
            max_layers: 最大显示层数
            
        Returns:
            str: 保存的文件路径
        """
        num_layers = min(len(attention_weights_list), max_layers)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i in range(num_layers):
            if i >= len(axes):
                break
                
            attn = attention_weights_list[i]
            if attn.dim() == 4:
                attn = attn[0].mean(dim=0)  # 平均所有头
            elif attn.dim() == 3:
                attn = attn.mean(dim=0)
            
            sns.heatmap(
                attn.detach().cpu().numpy(),
                cmap='Blues',
                cbar=True,
                ax=axes[i],
                xticklabels=False,
                yticklabels=False
            )
            axes[i].set_title(f"Layer {layer_indices[i]}")
            axes[i].set_xlabel("Key Position")
            axes[i].set_ylabel("Query Position")
        
        # 隐藏多余的子图
        for i in range(num_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle("Multi-Layer Attention Weights", fontsize=16)
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = os.path.join(
                self.save_dir,
                f"multi_layer_attention.png"
            )
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_attention_statistics(
        self,
        attention_weights_list: List[torch.Tensor],
        layer_indices: List[int],
        save_path: Optional[str] = None
    ) -> str:
        """
        可视化注意力统计信息
        
        Args:
            attention_weights_list: 多层注意力权重列表
            layer_indices: 层索引列表
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        # 计算每层的统计信息
        layer_stats = []
        for i, attn in enumerate(attention_weights_list):
            if attn.dim() == 4:
                attn = attn[0].mean(dim=0)  # 平均所有头
            elif attn.dim() == 3:
                attn = attn.mean(dim=0)
            
            attn_np = attn.detach().cpu().numpy()
            
            stats = {
                'layer': layer_indices[i],
                'mean': np.mean(attn_np),
                'std': np.std(attn_np),
                'max': np.max(attn_np),
                'min': np.min(attn_np),
                'entropy': -np.sum(attn_np * np.log(attn_np + 1e-8))
            }
            layer_stats.append(stats)
        
        # 创建统计图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        layers = [s['layer'] for s in layer_stats]
        
        # 1. 平均注意力
        means = [s['mean'] for s in layer_stats]
        axes[0, 0].plot(layers, means, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Mean Attention by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Attention')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 注意力标准差
        stds = [s['std'] for s in layer_stats]
        axes[0, 1].plot(layers, stds, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Attention Std by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Attention Std')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 注意力范围
        maxs = [s['max'] for s in layer_stats]
        mins = [s['min'] for s in layer_stats]
        axes[1, 0].plot(layers, maxs, 'go-', linewidth=2, markersize=8, label='Max')
        axes[1, 0].plot(layers, mins, 'mo-', linewidth=2, markersize=8, label='Min')
        axes[1, 0].set_title('Attention Range by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Attention Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 注意力熵
        entropies = [s['entropy'] for s in layer_stats]
        axes[1, 1].plot(layers, entropies, 'co-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Attention Entropy by Layer')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = os.path.join(
                self.save_dir,
                f"attention_statistics.png"
            )
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
