#!/usr/bin/env python3
"""
调试张量大小不匹配问题
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
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
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

def debug_tensor_mismatch():
    """调试张量大小不匹配问题"""
    print("开始调试张量大小不匹配问题...")
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
    
    # 4. 测试图像处理 - 详细调试
    print("4. 测试图像处理...")
    image = processed_sample['image']
    
    print(f"  原始图像形状: {image.size}")
    print(f"  图像模式: {image.mode}")
    print(f"  图像处理器: {type(image_processor)}")
    print(f"  模型配置: {model.config}")
    
    try:
        # 测试不同的图像处理方式
        print("  测试方式1: 直接使用image_processor...")
        result1 = image_processor([image], return_tensors='pt')
        print(f"    ✓ 成功，结果形状: {result1['pixel_values'].shape}")
        
        print("  测试方式2: 使用process_images函数...")
        result2 = process_images([image], image_processor, model.config)
        print(f"    ✓ 成功，结果类型: {type(result2)}")
        if isinstance(result2, list):
            print(f"    列表长度: {len(result2)}")
            for i, tensor in enumerate(result2):
                print(f"    张量{i}形状: {tensor.shape}")
        else:
            print(f"    张量形状: {result2.shape}")
            
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试其他方式
        print("  尝试修复...")
        try:
            # 尝试resize图像
            from PIL import Image
            resized_image = image.resize((336, 336))
            print(f"    Resize后形状: {resized_image.size}")
            
            result3 = image_processor([resized_image], return_tensors='pt')
            print(f"    ✓ Resize后成功，结果形状: {result3['pixel_values'].shape}")
            
        except Exception as e2:
            print(f"    ❌ Resize后仍然失败: {e2}")
    
    # 5. 测试token处理
    print("5. 测试token处理...")
    try:
        question = processed_sample['question']
        prompt = f"<image>\n{question}"
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        print(f"✓ Token处理成功，输入ID形状: {input_ids.shape}")
        
    except Exception as e:
        print(f"❌ Token处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("调试完成!")

if __name__ == "__main__":
    debug_tensor_mismatch()




