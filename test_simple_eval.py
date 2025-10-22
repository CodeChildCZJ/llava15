#!/usr/bin/env python3
"""
简化的评估测试 - 验证基本功能
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 设置随机种子
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def test_simple_eval():
    """简化的评估测试"""
    print("开始简化评估测试...")
    print("="*60)
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # 构建命令 - 先测试不启用mask的版本
    cmd = [
        "python", "eval_gqa_masked.py",
        "--model_path", "/home/czj/llava15_test/llava-v1.5-7b",
        "--clip_path", "/home/czj/llava15_test/clip-vit-large-patch14-336",
        "--gqa_root", "/home/Dataset/Dataset/GQA", 
        "--gqa_split", "train_balanced",
        "--num_samples", "2",  # 只测试2个样本
        "--device", "cuda",
        "--load_in_8bit",
        "--torch_dtype", "float16"
        # 不启用mask，先测试基本功能
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    # 执行命令
    import subprocess
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"返回码: {result.returncode}")
        
        if result.returncode == 0:
            print("\n🎉 基本功能测试成功！")
            return True
        else:
            print("\n❌ 基本功能测试失败")
            return False
            
    except Exception as e:
        print(f"执行失败: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_eval()
    if success:
        print("\n现在可以测试mask功能了...")
    else:
        print("\n需要先修复基本功能问题")





