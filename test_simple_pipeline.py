#!/usr/bin/env python3
"""
简化的pipeline测试，验证GQA数据加载和mask功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from llava.data.gqa_loader import GQALoader
from llava.model.llava_arch_masked import RandomMaskStrategy
import torch
import numpy as np


def test_simple_pipeline():
    """测试简化的pipeline"""
    print("测试简化的GQA pipeline...")
    print("="*60)
    
    # 1. 测试数据加载
    print("1. 测试GQA数据加载...")
    try:
        gqa_loader = GQALoader(gqa_root="/home/Dataset/Dataset/GQA")
        dataset = gqa_loader.load_dataset(split="train_balanced", num_samples=2)
        print(f"✓ 成功加载 {len(dataset)} 个样本")
        
        # 处理第一个样本
        sample = dataset[0]
        processed_sample = gqa_loader.process_sample(sample)
        print(f"✓ 样本处理成功，图像形状: {processed_sample['image'].size if processed_sample['image'] else 'None'}")
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    # 2. 测试mask策略
    print("\n2. 测试mask策略...")
    try:
        # 创建模拟的视觉token
        batch_size, seq_len, hidden_dim = 1, 576, 4096  # LLaVA-1.5的典型维度
        visual_tokens = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 测试随机mask策略 - 在num_tokens维度上操作
        mask_strategy = RandomMaskStrategy(mask_ratio=0.2)
        
        # 获取mask索引 - 对每个样本的token维度进行mask
        batch_size, num_tokens, hidden_size = visual_tokens.shape
        masked_tokens = visual_tokens.clone()
        
        for i in range(batch_size):
            # 对单个样本的token维度进行mask操作
            single_sample_tokens = visual_tokens[i]  # [num_tokens, hidden_size]
            mask_indices = mask_strategy.get_mask_indices(single_sample_tokens)
            
            # 应用mask - 将选中的整个token置零
            masked_tokens[i][mask_indices] = 0.0
        
        print(f"✓ 原始token形状: {visual_tokens.shape}")
        print(f"✓ Mask后token形状: {masked_tokens.shape}")
        print(f"✓ Mask索引数量: {len(mask_indices)}")
        print(f"✓ Mask比例: {len(mask_indices) / seq_len:.3f}")
        
        # 验证mask是否正确应用
        original_values = visual_tokens[0, mask_indices, :]
        masked_values = masked_tokens[0, mask_indices, :]
        is_masked = torch.allclose(masked_values, torch.zeros_like(masked_values))
        print(f"✓ Mask验证: {'通过' if is_masked else '失败'}")
        
    except Exception as e:
        print(f"✗ Mask策略测试失败: {e}")
        return False
    
    # 3. 测试注意力可视化
    print("\n3. 测试注意力可视化...")
    try:
        from llava.utils.attention_visualizer import AttentionVisualizer
        
        # 创建模拟注意力权重
        attention_weights = torch.randn(1, 12, 576, 576)  # [batch, heads, seq_len, seq_len]
        
        visualizer = AttentionVisualizer("./test_attention")
        visualizer.visualize_attention_heatmap(
            attention_weights,
            layer_idx=0,
            head_idx=0,
            title="测试注意力可视化"
        )
        print("✓ 注意力可视化测试成功")
        
    except Exception as e:
        print(f"✗ 注意力可视化测试失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！Pipeline工作正常")
    print("="*60)
    
    return True


def main():
    """主函数"""
    success = test_simple_pipeline()
    
    if success:
        print("\n🎉 恭喜！您的LLaVA-1.5视觉token mask消融实验环境已经准备就绪！")
        print("\n下一步可以运行完整的评估:")
        print("python eval_gqa_masked.py \\")
        print("    --model_path liuhaotian/llava-v1.5-7b \\")
        print("    --gqa_root /home/Dataset/Dataset/GQA \\")
        print("    --gqa_split train_balanced \\")
        print("    --num_samples 10 \\")
        print("    --mask_visual_token \\")
        print("    --mask_ratio 0.2 \\")
        print("    --mask_strategy random")
    else:
        print("\n❌ 测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
