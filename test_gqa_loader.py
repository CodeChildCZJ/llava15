#!/usr/bin/env python3
"""
测试GQA数据加载器
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from llava.data.gqa_loader import GQALoader, test_gqa_loader


def main():
    """主测试函数"""
    print("测试GQA数据加载器...")
    print("="*60)
    
    # 运行内置测试
    test_gqa_loader()
    
    print("\n使用方法:")
    print("1. 测试数据加载:")
    print("   python test_gqa_loader.py")
    print("\n2. 本地数据集评估:")
    print("   python eval_gqa_masked.py \\")
    print("       --model_path liuhaotian/llava-v1.5-7b \\")
    print("       --gqa_root /home/Dataset/Dataset/GQA \\")
    print("       --gqa_split train_balanced \\")
    print("       --num_samples 10 \\")
    print("       --mask_visual_token \\")
    print("       --mask_ratio 0.2 \\")
    print("       --mask_strategy random")
    print("\n3. 远程数据集评估:")
    print("   python eval_gqa_masked.py \\")
    print("       --model_path liuhaotian/llava-v1.5-7b \\")
    print("       --use_remote \\")
    print("       --dataset_name lmms-lab/GQA \\")
    print("       --gqa_split train \\")
    print("       --num_samples 10 \\")
    print("       --mask_visual_token \\")
    print("       --mask_ratio 0.2 \\")
    print("       --mask_strategy random")


if __name__ == "__main__":
    main()
