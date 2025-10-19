#!/usr/bin/env python3
"""
检查LLaVA运行所需的所有文件
"""

import os
from pathlib import Path

def check_requirements():
    """检查所有必需的文件"""
    print("检查LLaVA运行所需文件...")
    print("="*60)
    
    # 1. LLaVA模型文件
    llava_path = "/home/czj/llava15_test/llava-v1.5-7b"
    print(f"1. 检查LLaVA模型文件: {llava_path}")
    
    llava_files = [
        "config.json",
        "generation_config.json", 
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "pytorch_model.bin.index.json",
        "mm_projector.bin"
    ]
    
    missing_llava = []
    for file in llava_files:
        if Path(llava_path, file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            missing_llava.append(file)
    
    # 检查模型权重文件
    model_files = list(Path(llava_path).glob("pytorch_model-*.bin"))
    if model_files:
        print(f"  ✅ 模型权重文件 ({len(model_files)} 个)")
    else:
        print("  ❌ 模型权重文件")
        missing_llava.append("pytorch_model-*.bin")
    
    # 2. CLIP视觉编码器
    print(f"\n2. 检查CLIP视觉编码器:")
    clip_path = "/home/czj/llava15_test/clip-vit-large-patch14-336"
    
    if Path(clip_path).exists():
        print(f"  ✅ CLIP目录存在: {clip_path}")
        # 检查CLIP文件
        clip_files = ["config.json", "pytorch_model.bin", "preprocessor_config.json"]
        for file in clip_files:
            if Path(clip_path, file).exists():
                print(f"    ✅ {file}")
            else:
                print(f"    ❌ {file}")
    else:
        print(f"  ❌ CLIP目录不存在: {clip_path}")
        print(f"    需要下载: openai/clip-vit-large-patch14-336")
    
    # 总结
    print(f"\n总结:")
    if missing_llava:
        print(f"❌ 缺少LLaVA文件: {missing_llava}")
    else:
        print(f"✅ LLaVA文件完整")
    
    print(f"\n下载命令:")
    print(f"# 1. 下载CLIP视觉编码器")
    print(f"git clone https://huggingface.co/openai/clip-vit-large-patch14-336 {clip_path}")
    print(f"")
    print(f"# 2. 如果缺少LLaVA文件，运行:")
    print(f"cd {llava_path}")
    print(f"git lfs pull")
    
    print(f"\n或者手动下载:")
    print(f"1. LLaVA模型: https://huggingface.co/liuhaotian/llava-v1.5-7b")
    print(f"2. CLIP编码器: https://huggingface.co/openai/clip-vit-large-patch14-336")

if __name__ == "__main__":
    check_requirements()
