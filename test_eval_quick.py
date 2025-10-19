#!/usr/bin/env python3
"""
快速测试评估脚本的基本功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from llava.data.gqa_loader import GQALoader
import torch

def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    # 1. 测试数据加载
    print("1. 测试GQA数据加载...")
    gqa_loader = GQALoader(gqa_root="/home/Dataset/Dataset/GQA")
    dataset = gqa_loader.load_dataset(split="train_balanced", num_samples=2)
    print(f"✓ 成功加载 {len(dataset)} 个样本")
    
    # 2. 测试样本处理
    print("2. 测试样本处理...")
    sample = dataset[0]
    processed_sample = gqa_loader.process_sample(sample)
    print(f"✓ 样本处理成功，图像形状: {processed_sample['image'].size if processed_sample['image'] else 'None'}")
    
    # 3. 测试mask策略
    print("3. 测试mask策略...")
    from llava.model.llava_arch_masked import RandomMaskStrategy
    
    # 创建模拟的视觉token
    batch_size, seq_len, hidden_dim = 1, 576, 4096
    visual_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    
    mask_strategy = RandomMaskStrategy(mask_ratio=0.2)
    masked_tokens = visual_tokens.clone()
    
    for i in range(batch_size):
        single_sample_tokens = visual_tokens[i]
        mask_indices = mask_strategy.get_mask_indices(single_sample_tokens)
        masked_tokens[i][mask_indices] = 0.0
    
    print(f"✓ Mask策略测试成功，mask比例: {mask_indices.sum().item() / seq_len:.3f}")
    
    print("\n✅ 所有基本功能测试通过！")
    print("现在可以运行完整的消融实验了。")

if __name__ == "__main__":
    test_basic_functionality()
