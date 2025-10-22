#!/usr/bin/env python3
"""
简化的1号显卡测试 - 避免复杂的设备映射问题
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_gpu1_simple():
    """简化的1号显卡测试"""
    print("开始简化的1号显卡测试...")
    print("="*60)
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # 构建命令 - 使用更简单的参数
    cmd = [
        "python", "eval_gqa_masked.py",
        "--model_path", "liuhaotian/llava-v1.5-7b",
        "--gqa_root", "/home/Dataset/Dataset/GQA", 
        "--gqa_split", "train_balanced",
        "--num_samples", "1",  # 减少样本数量
        "--device", "cuda",  # 使用简单的cuda，让系统自动选择
        "--load_in_8bit",  # 使用8bit量化节省内存
        "--torch_dtype", "float16",
        "--mask_visual_token",
        "--mask_ratio", "0.2",
        "--mask_strategy", "random"
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
    except Exception as e:
        print(f"执行失败: {e}")

if __name__ == "__main__":
    test_gpu1_simple()





