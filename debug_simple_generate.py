#!/usr/bin/env python3
"""
调试简单的生成过程，定位张量大小不匹配问题
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

def debug_simple_generate():
    """调试简单的生成过程"""
    print("开始调试简单生成过程...")
    print("="*60)
    
    # 1. 测试数据加载
    print("1. 测试数据加载...")
    gqa_loader = GQALoader(gqa_root="/home/Dataset/Dataset/GQA")
    dataset = gqa_loader.load_dataset(split="train_balanced", num_samples=1)
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
    
    # 3. 处理样本
    print("3. 处理样本...")
    sample = dataset[0]
    processed_sample = gqa_loader.process_sample(sample)
    image = processed_sample['image']
    question = processed_sample['question']
    
    print(f"  原始图像形状: {image.size}")
    print(f"  图像模式: {image.mode}")
    
    # 4. 图像处理
    print("4. 图像处理...")
    try:
        # 确保图像是RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 使用image_processor直接处理，确保尺寸统一
        image_tensor = image_processor(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        print(f"  ✓ 图像处理成功: 处理后形状={image_tensor.shape}")
        
    except Exception as e:
        print(f"  ❌ 图像处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Token处理
    print("5. Token处理...")
    try:
        prompt = f"<image>\n{question}"
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        print(f"  ✓ Token处理成功: 输入ID形状={input_ids.shape}")
        
    except Exception as e:
        print(f"  ❌ Token处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 测试生成 - 不使用mask
    print("6. 测试生成（不使用mask）...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id
            )
            print(f"  ✓ 生成成功: 输出形状={outputs.shape}")
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  生成文本: {generated_text}")
            
    except Exception as e:
        print(f"  ❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 测试生成 - 使用mask
    print("7. 测试生成（使用mask）...")
    try:
        # 设置mask参数
        model.config.mask_visual_token = True
        model.config.mask_ratio = 0.2
        model.config.mask_strategy = "random"
        model.config.mask_token_value = 0.0
        
        # 初始化mask策略
        from llava.model.llava_arch_masked import RandomMaskStrategy
        model.mask_strategy_obj = RandomMaskStrategy(0.2)
        
        print(f"  ✓ Mask策略设置成功")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id
            )
            print(f"  ✓ 带mask生成成功: 输出形状={outputs.shape}")
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0],已跳过特殊tokens=True)
            print(f"  带mask生成文本: {generated_text}")
            
    except Exception as e:
        print(f"  ❌ 带mask生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("调试完成!")

if __name__ == "__main__":
    debug_simple_generate()





