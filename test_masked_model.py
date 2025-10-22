#!/usr/bin/env python3
"""
测试masked模型的简单脚本
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from llava.model.llava_arch_masked import LlavaMetaModelMasked, MaskStrategy, RandomMaskStrategy, AttentionBasedMaskStrategy, PositionBasedMaskStrategy


def test_mask_strategies():
    """测试不同的mask策略"""
    print("测试mask策略...")
    
    # 创建模拟数据
    batch_size, seq_len, hidden_size = 2, 10, 512
    image_features = torch.randn(seq_len, hidden_size)
    
    # 测试随机mask策略
    print("测试随机mask策略...")
    random_strategy = RandomMaskStrategy(mask_ratio=0.3)
    mask_indices = random_strategy.get_mask_indices(image_features)
    print(f"随机mask索引: {mask_indices.sum().item()}/{len(mask_indices)} tokens被mask")
    
    # 测试基于注意力的mask策略
    print("测试基于注意力的mask策略...")
    attention_weights = torch.randn(8, seq_len, seq_len)  # 8个注意力头
    attention_strategy = AttentionBasedMaskStrategy(mask_ratio=0.3)
    mask_indices = attention_strategy.get_mask_indices(image_features, attention_weights)
    print(f"注意力mask索引: {mask_indices.sum().item()}/{len(mask_indices)} tokens被mask")
    
    # 测试基于位置的mask策略
    print("测试基于位置的mask策略...")
    position_strategy = PositionBasedMaskStrategy(mask_ratio=0.3, mask_center=True)
    mask_indices = position_strategy.get_mask_indices(image_features)
    print(f"位置mask索引: {mask_indices.sum().item()}/{len(mask_indices)} tokens被mask")
    
    print("所有mask策略测试完成!")


def test_visual_token_masking():
    """测试视觉token mask功能"""
    print("测试视觉token mask功能...")
    
    # 创建模拟的视觉特征
    num_tokens, hidden_size = 256, 1024  # 16x16的图像patch
    image_features = torch.randn(num_tokens, hidden_size)
    
    # 测试不同的mask策略
    strategies = {
        'random': RandomMaskStrategy(mask_ratio=0.2),
        'attention': AttentionBasedMaskStrategy(mask_ratio=0.2),
        'position_center': PositionBasedMaskStrategy(mask_ratio=0.2, mask_center=True),
        'position_edge': PositionBasedMaskStrategy(mask_ratio=0.2, mask_center=False)
    }
    
    for name, strategy in strategies.items():
        print(f"\n测试 {name} 策略:")
        mask_indices = strategy.get_mask_indices(image_features)
        masked_features = image_features.clone()
        masked_features[mask_indices] = 0.0
        
        print(f"  - 原始特征形状: {image_features.shape}")
        print(f"  - Mask的token数量: {mask_indices.sum().item()}")
        print(f"  - Mask比例: {mask_indices.sum().item() / len(mask_indices):.3f}")
        print(f"  - 特征变化: 原始均值={image_features.mean():.3f}, Mask后均值={masked_features.mean():.3f}")


def test_attention_visualizer():
    """测试注意力可视化器"""
    print("测试注意力可视化器...")
    
    try:
        from llava.utils.attention_visualizer import AttentionVisualizer
        
        # 创建可视化器
        visualizer = AttentionVisualizer(save_dir="./test_attention_viz")
        
        # 创建模拟注意力权重
        batch_size, num_heads, seq_len = 1, 8, 20
        attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
        
        # 测试注意力热力图
        save_path = visualizer.visualize_attention_heatmap(
            attention_weights, 
            layer_idx=0, 
            head_idx=None,
            title="测试注意力权重"
        )
        print(f"注意力热力图已保存到: {save_path}")
        
        # 测试对视觉token的注意力可视化
        visual_token_start, visual_token_end = 5, 15
        save_path = visualizer.visualize_attention_to_visual_tokens(
            attention_weights,
            visual_token_start,
            visual_token_end,
            layer_idx=0
        )
        print(f"视觉token注意力图已保存到: {save_path}")
        
        print("注意力可视化测试完成!")
        
    except ImportError as e:
        print(f"无法导入注意力可视化器: {e}")
        print("请确保安装了matplotlib和seaborn")


def main():
    """主测试函数"""
    print("开始测试masked模型...")
    
    # 测试mask策略
    test_mask_strategies()
    
    # 测试视觉token mask
    test_visual_token_masking()
    
    # 测试注意力可视化
    test_attention_visualizer()
    
    print("\n所有测试完成!")


if __name__ == "__main__":
    main()





