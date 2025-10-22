#!/usr/bin/env python3
"""
调试图像处理问题
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import random
import numpy as np

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from llava.data.gqa_loader import GQALoader
from llava.mm_utils import process_images
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import llava.utils as utils

# 直接从utils.py导入
import importlib.util
utils_path = Path('.') / 'llava' / 'utils.py'
spec = importlib.util.spec_from_file_location('llava_utils', utils_path)
llava_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llava_utils)
disable_torch_init = llava_utils.disable_torch_init

def debug_image_processing():
    """调试图像处理"""
    print("开始调试图像处理...")
    print("="*60)
    
    # 1. 测试数据加载
    print("1. 测试数据加载...")
    gqa_loader = GQALoader(gqa_root="/home/Dataset/Dataset/GQA")
    dataset = gqa_loader.load_dataset(split="train_balanced", num_samples=1)
    print(f"✓ 成功加载 {len(dataset)} 个样本")
    
    # 2. 处理样本
    print("2. 处理样本...")
    sample = dataset[0]
    processed_sample = gqa_loader.process_sample(sample)
    print(f"✓ 样本处理成功")
    print(f"  图像形状: {processed_sample['image'].size if processed_sample['image'] else 'None'}")
    print(f"  问题: {processed_sample['question'][:100]}...")
    
    # 3. 加载模型
    print("3. 加载模型...")
    disable_torch_init()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path="/home/czj/llava15_test/llava-v1.5-7b",
        model_base=None,
        model_name="llava-v1.5-7b",
        load_8bit=True,
        load_4bit=False,
        device_map="auto"
    )
    print(f"✓ 模型加载成功")
    print(f"  模型设备: {model.device}")
    
    # 4. 测试图像处理
    print("4. 测试图像处理...")
    image = processed_sample['image']
    try:
        print(f"  原始图像形状: {image.size}")
        print(f"  图像模式: {image.mode}")
        
        # 测试图像处理
        image_tensor = process_images([image], image_processor, model.config)
        print(f"✓ 图像处理成功")
        print(f"  处理后类型: {type(image_tensor)}")
        
        if isinstance(image_tensor, list):
            print(f"  张量列表长度: {len(image_tensor)}")
            for i, tensor in enumerate(image_tensor):
                print(f"    张量{i}形状: {tensor.shape}")
        else:
            print(f"  张量形状: {image_tensor.shape}")
            
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("调试完成!")

if __name__ == "__main__":
    debug_image_processing()





