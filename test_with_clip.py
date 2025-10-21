#!/usr/bin/env python3
"""
使用本地CLIP路径进行测试
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_with_clip():
    """使用本地CLIP路径测试"""
    print("开始使用本地CLIP路径测试...")
    print("="*60)
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # 检查CLIP路径是否存在
    clip_path = "/home/czj/llava15_test/clip-vit-large-patch14-336"
    if Path(clip_path).exists():
        print(f"✅ CLIP路径存在: {clip_path}")
        # 列出CLIP文件
        clip_files = list(Path(clip_path).glob("*"))
        print(f"CLIP文件数量: {len(clip_files)}")
        for file in clip_files[:5]:  # 只显示前5个文件
            print(f"  - {file.name}")
        if len(clip_files) > 5:
            print(f"  ... 还有 {len(clip_files) - 5} 个文件")
    else:
        print(f"❌ CLIP路径不存在: {clip_path}")
        return
    
    # 构建命令
    cmd = [
        "python", "eval_gqa_masked.py",
        "--model_path", "/home/czj/llava15_test/llava-v1.5-7b",  # 使用本地路径
        "--clip_path", clip_path,  # 使用本地CLIP路径
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
    
    print("\n执行命令:")
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
            print("\n🎉 测试成功！")
        else:
            print("\n❌ 测试失败")
            
    except Exception as e:
        print(f"执行失败: {e}")

if __name__ == "__main__":
    test_with_clip()




