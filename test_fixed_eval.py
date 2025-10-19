#!/usr/bin/env python3
"""
测试修复后的评估脚本
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
from llava.mm_utils import tokenizer_image_token
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

def test_fixed_evaluation():
    """测试修复后的评估"""
    print("开始测试修复后的评估...")
    print("="*60)
    
    # 1. 测试数据加载
    print("1. 测试数据加载...")
    gqa_loader = GQALoader(gqa_root="/home/Dataset/Dataset/GQA")
    dataset = gqa_loader.load_dataset(split="train_balanced", num_samples=3)
    print(f"✓ 成功加载 {len(dataset)} 个样本")
    
    # 2. 加载模型
    print("2. 加载模型...")
    disable_torch_init()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path="/home/czj/llava15_test/llava-v1.5-7b",
        model_base=None,
        model_name="llava-v1.5-7b",
        load_8bit=True,
        load_4bit=False,
        device_map="cuda:1"
    )
    print(f"✓ 模型加载成功")
    print(f"  模型设备: {model.device}")
    
    # 3. 测试图像处理
    print("3. 测试图像处理...")
    for i, sample in enumerate(dataset):
        print(f"  处理样本 {i}...")
        
        # 处理样本
        processed_sample = gqa_loader.process_sample(sample)
        image = processed_sample['image']
        question = processed_sample['question']
        
        print(f"    原始图像形状: {image.size}")
        print(f"    图像模式: {image.mode}")
        
        try:
            # 确保图像是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 使用image_processor直接处理，确保尺寸统一
            image_tensor = image_processor(image, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            print(f"    ✓ 图像处理成功: 处理后形状={image_tensor.shape}")
            
            # 测试token处理
            prompt = f"<image>\n{question}"
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            print(f"    ✓ Token处理成功: 输入ID形状={input_ids.shape}")
            
            # 测试简单的forward
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.eos_token_id
                )
                print(f"    ✓ 生成成功: 输出形状={outputs.shape}")
                
        except Exception as e:
            print(f"    ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("测试完成!")

if __name__ == "__main__":
    test_fixed_evaluation()
