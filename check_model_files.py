#!/usr/bin/env python3
"""
检查LLaVA模型文件是否完整
"""

import os
from pathlib import Path

def check_model_files(model_path):
    """检查模型文件是否完整"""
    print(f"检查模型路径: {model_path}")
    print("="*60)
    
    # 必需的文件列表
    required_files = [
        "config.json",
        "generation_config.json", 
        "tokenizer_config.json",
        "tokenizer.model",
        "tokenizer.json",  # 这个文件缺失
        "special_tokens_map.json",
        "pytorch_model.bin.index.json",
        "mm_projector.bin"
    ]
    
    # 检查文件是否存在
    missing_files = []
    existing_files = []
    
    for file in required_files:
        file_path = Path(model_path) / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file} ({size:,} bytes)")
            existing_files.append(file)
        else:
            print(f"❌ {file} (缺失)")
            missing_files.append(file)
    
    # 检查模型权重文件
    print("\n检查模型权重文件:")
    model_files = list(Path(model_path).glob("pytorch_model-*.bin"))
    if model_files:
        for model_file in sorted(model_files):
            size = model_file.stat().st_size
            print(f"✅ {model_file.name} ({size:,} bytes)")
    else:
        print("❌ 未找到pytorch_model-*.bin文件")
        missing_files.append("pytorch_model-*.bin")
    
    print(f"\n总结:")
    print(f"现有文件: {len(existing_files)}")
    print(f"缺失文件: {len(missing_files)}")
    
    if missing_files:
        print(f"\n需要下载的文件:")
        for file in missing_files:
            print(f"- {file}")
            
        print(f"\n下载命令:")
        print(f"# 使用git lfs下载缺失文件")
        print(f"cd {model_path}")
        print(f"git lfs pull")
        print(f"\n或者手动下载:")
        print(f"https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main")
    else:
        print(f"\n✅ 所有必需文件都已存在!")

if __name__ == "__main__":
    model_path = "/home/czj/llava15_test/llava-v1.5-7b"
    check_model_files(model_path)
