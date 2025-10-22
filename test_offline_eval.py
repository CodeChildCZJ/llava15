#!/usr/bin/env python3
"""
离线测试版本 - 不依赖HuggingFace下载
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_offline_eval():
    """离线测试评估"""
    print("开始离线测试...")
    print("="*60)
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # 检查tokenizer.json是否存在
    tokenizer_json_path = "/home/czj/llava15_test/llava-v1.5-7b/tokenizer.json"
    if not Path(tokenizer_json_path).exists():
        print(f"❌ 缺少tokenizer.json文件: {tokenizer_json_path}")
        print("请先下载tokenizer.json文件:")
        print("方法1: cd /home/czj/llava15_test/llava-v1.5-7b && git lfs pull")
        print("方法2: 手动下载 https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/tokenizer.json")
        return
    
    print("✅ tokenizer.json文件存在")
    
    # 构建命令
    cmd = [
        "python", "eval_gqa_masked.py",
        "--model_path", "/home/czj/llava15_test/llava-v1.5-7b",  # 使用本地路径
        "--gqa_root", "/home/Dataset/Dataset/GQA", 
        "--gqa_split", "train_balanced",
        "--num_samples", "1",
        "--device", "cuda",
        "--load_in_8bit",
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
    test_offline_eval()





