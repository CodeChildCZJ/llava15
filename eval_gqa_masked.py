#!/usr/bin/env python3
"""
GQA评估脚本，支持视觉token mask消融实验
支持本地和远程GQA数据集
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random

# 固定随机种子
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 设置随机种子
set_seed(42)

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
# 直接从utils.py文件导入
import importlib.util
utils_path = Path(__file__).parent / "llava" / "utils.py"
spec = importlib.util.spec_from_file_location("llava_utils", utils_path)
llava_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llava_utils)
disable_torch_init = llava_utils.disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.utils.attention_visualizer import AttentionVisualizer
from llava.data.gqa_loader import GQALoader

from PIL import Image
import transformers


def setup_args():
    parser = argparse.ArgumentParser(description="GQA评估脚本 - 支持视觉token mask")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:1", help="设备类型，支持指定GPU如cuda:1")
    parser.add_argument("--load_in_8bit", action="store_true", help="使用8bit量化加载模型")
    parser.add_argument("--load_in_4bit", action="store_true", help="使用4bit量化加载模型")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="模型数据类型")
    parser.add_argument("--clip_path", type=str, default="/home/czj/llava15_test/clip-vit-large-patch14-336", help="CLIP视觉编码器本地路径")
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14-336", help="视觉编码器名称，如果本地不存在则从网上下载")
    
    # 数据参数
    parser.add_argument("--gqa_root", type=str, default="/home/Dataset/Dataset/GQA", help="本地GQA数据集根目录")
    parser.add_argument("--use_remote", action="store_true", help="使用远程HuggingFace数据集")
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/GQA", help="HuggingFace数据集名称")
    parser.add_argument("--gqa_split", type=str, default="train_balanced", help="GQA数据集分割")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num_samples", type=int, default=None, help="评估样本数量")
    
    # Mask参数
    parser.add_argument("--mask_visual_token", action="store_true", help="是否启用视觉token mask")
    parser.add_argument("--mask_ratio", type=float, default=0.1, help="Mask比例 (0.0-1.0)")
    parser.add_argument("--mask_strategy", type=str, default="random", 
                       choices=["random", "attention", "position_center", "position_edge"],
                       help="Mask策略")
    parser.add_argument("--mask_token_value", type=float, default=0.0, help="Mask后的token值")
    
    # 注意力可视化参数
    parser.add_argument("--visualize_attention", action="store_true", help="是否可视化注意力权重")
    parser.add_argument("--attention_layers", type=str, default="0,6,12,18", help="要可视化的层索引")
    parser.add_argument("--attention_heads", type=str, default="0,4,8,12", help="要可视化的头索引")
    parser.add_argument("--save_attention_dir", type=str, default="./attention_visualizations", 
                       help="注意力可视化保存目录")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./gqa_results", help="结果保存目录")
    parser.add_argument("--save_predictions", action="store_true", help="是否保存预测结果")
    
    return parser.parse_args()


def check_clip_path(clip_path: str, vision_tower: str) -> str:
    """检查CLIP路径是否存在，如果不存在则返回远程路径"""
    if Path(clip_path).exists():
        print(f"✅ 使用本地CLIP路径: {clip_path}")
        return clip_path
    else:
        print(f"❌ 本地CLIP路径不存在: {clip_path}")
        print(f"🔄 将使用远程路径: {vision_tower}")
        return vision_tower

def load_model_and_tokenizer(model_path: str, model_base: Optional[str] = None, device: str = "cuda:1", 
                           load_in_8bit: bool = False, load_in_4bit: bool = False, torch_dtype: str = "float16",
                           clip_path: str = None, vision_tower: str = "openai/clip-vit-large-patch14-336"):
    """加载模型和tokenizer"""
    print(f"加载模型从 {model_path}")
    print(f"使用设备: {device}")
    print(f"量化设置: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
    print(f"数据类型: {torch_dtype}")
    
    # 检查CLIP路径
    if clip_path and Path(clip_path).exists():
        vision_tower_path = check_clip_path(clip_path, vision_tower)
        # 设置环境变量，让transformers使用本地路径
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线模式
        print(f"🔧 设置离线模式，使用本地CLIP路径: {clip_path}")
    else:
        vision_tower_path = vision_tower
    
    # 禁用torch初始化以节省内存
    disable_torch_init()
    
    # 设置设备映射
    if device.startswith("cuda:"):
        gpu_id = int(device.split(":")[1])
        device_map = f"cuda:{gpu_id}"
    else:
        device_map = "auto"
    
    # 设置数据类型
    if torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    # 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path),
        load_8bit=load_in_8bit,
        load_4bit=load_in_4bit,
        device_map=device_map,
        torch_dtype=dtype,
        use_masked_model=True  # 使用masked模型
    )
    
    return tokenizer, model, image_processor, context_len


def prepare_question(question: str, conv_mode: str = "llava_v1") -> str:
    """准备问题格式"""
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def evaluate_model_on_gqa(
    model,
    tokenizer,
    image_processor,
    gqa_loader: GQALoader,
    args
) -> Dict:
    """在GQA数据上评估模型"""
    
    model.eval()
    results = []
    correct = 0
    total = 0
    
    # 初始化注意力可视化器
    attention_visualizer = None
    if args.visualize_attention:
        attention_visualizer = AttentionVisualizer(args.save_attention_dir)
        attention_layers = [int(x) for x in args.attention_layers.split(',')]
        attention_heads = [int(x) for x in args.attention_heads.split(',')]
    
    # 加载数据集
    dataset = gqa_loader.load_dataset(split=args.gqa_split, num_samples=args.num_samples)
    
    print(f"开始评估 {len(dataset)} 个样本...")
    
    # 处理每个样本
    for i, sample in enumerate(tqdm(dataset, desc="评估进度")):
        if args.num_samples and i >= args.num_samples:
            break
            
        try:
            # 处理样本，加载图像
            processed_sample = gqa_loader.process_sample(sample)
            
            # 检查是否有图像
            if processed_sample['image'] is None:
                print(f"样本 {i} 没有图像，跳过")
                continue
            
            image = processed_sample['image']
            question = processed_sample['question']
            ground_truth = processed_sample['answer']
            
            # 准备问题
            prompt = prepare_question(question, args.conv_mode)
            
            # 处理图像 - 确保尺寸一致
            try:
                # 首先确保图像是RGB模式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 使用image_processor直接处理，确保尺寸统一
                image_tensor = image_processor(image, return_tensors='pt')['pixel_values']
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                
                print(f"  图像处理成功: 原始尺寸={image.size}, 处理后形状={image_tensor.shape}")
                
            except Exception as e:
                print(f"图像处理失败: {e}")
                print(f"图像形状: {image.size}")
                import traceback
                traceback.print_exc()
                continue
            
            # 准备输入
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            
            # 生成回答
            with torch.no_grad():
                if args.visualize_attention:
                    # 需要获取注意力权重
                    outputs = model.generate(
                        input_ids=input_ids,
                        images=image_tensor,
                        do_sample=False,
                        temperature=0,
                        top_p=None,
                        num_beams=1,
                        max_new_tokens=512,
                        output_attentions=True,
                        return_dict_in_generate=True
                    )
                    
                    # 可视化注意力权重
                    if hasattr(outputs, 'attentions') and outputs.attentions:
                        attention_weights = outputs.attentions
                        for layer_idx in attention_layers:
                            if layer_idx < len(attention_weights):
                                for head_idx in attention_heads:
                                    if head_idx < attention_weights[layer_idx].shape[1]:
                                        # 可视化特定层的注意力
                                        attention_visualizer.visualize_attention_heatmap(
                                            attention_weights[layer_idx],
                                            layer_idx=layer_idx,
                                            head_idx=head_idx,
                                            title=f"Sample {i} - {question[:50]}..."
                                        )
                else:
                    outputs = model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=False,
                        temperature=0,
                        top_p=None,
                        num_beams=1,
                        max_new_tokens=512
                    )
            
            # 解码输出
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != outputs[0][:len(input_ids[0])]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
            outputs = tokenizer.batch_decode(outputs[0][input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            
            # 评估答案
            predicted_answer = outputs.lower().strip()
            ground_truth_lower = ground_truth.lower().strip()
            
            is_correct = predicted_answer == ground_truth_lower
            if is_correct:
                correct += 1
            total += 1
            
            # 保存结果
            result = {
                'id': processed_sample['id'],
                'question': question,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'correct': is_correct,
                'imageId': processed_sample['imageId']
            }
            results.append(result)
            
            if i % 100 == 0:
                print(f"进度: {i}/{len(dataset)}, 准确率: {correct/total:.4f}")
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue
    
    # 计算最终结果
    accuracy = correct / total if total > 0 else 0.0
    
    final_results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results,
        'mask_config': {
            'mask_visual_token': args.mask_visual_token,
            'mask_ratio': args.mask_ratio,
            'mask_strategy': args.mask_strategy,
            'mask_token_value': args.mask_token_value
        }
    }
    
    return final_results


def save_results(results: Dict, output_dir: str, args):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    if args.save_predictions:
        results_file = os.path.join(output_dir, "detailed_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存到: {results_file}")
    
    # 保存摘要结果
    summary = {
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'total': results['total'],
        'mask_config': results['mask_config']
    }
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要结果已保存到: {summary_file}")
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果:")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"正确: {results['correct']}/{results['total']}")
    print(f"Mask配置: {results['mask_config']}")
    print("="*50)


def main():
    args = setup_args()
    
    print("开始GQA评估...")
    print(f"参数配置: {vars(args)}")
    
    # 创建GQA加载器
    if args.use_remote:
        gqa_loader = GQALoader(dataset_name=args.dataset_name)
        print("使用远程HuggingFace数据集")
    else:
        gqa_loader = GQALoader(gqa_root=args.gqa_root)
        print("使用本地GQA数据集")
    
    # 测试数据加载
    print("测试数据加载...")
    try:
        test_dataset = gqa_loader.load_dataset(split=args.gqa_split, num_samples=1)
        dataset_info = gqa_loader.get_dataset_info(test_dataset)
        print(f"数据集信息: {dataset_info}")
        
        # 显示第一个样本的结构
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            print(f"第一个样本的键: {list(sample.keys())}")
        
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        print("请检查GQA数据集路径和格式")
        return
    
    # 加载模型
    tokenizer, model, image_processor, context_len = load_model_and_tokenizer(
        args.model_path, args.model_base, args.device,
        args.load_in_8bit, args.load_in_4bit, args.torch_dtype,
        args.clip_path, args.vision_tower
    )
    
    # 设置mask参数
    if args.mask_visual_token:
        model.config.mask_visual_token = True
        model.config.mask_ratio = args.mask_ratio
        model.config.mask_strategy = args.mask_strategy
        model.config.mask_token_value = args.mask_token_value
        
        # 初始化mask策略对象
        if args.mask_strategy == 'random':
            from llava.model.llava_arch_masked import RandomMaskStrategy
            model.mask_strategy_obj = RandomMaskStrategy(args.mask_ratio)
        elif args.mask_strategy == 'attention':
            from llava.model.llava_arch_masked import AttentionBasedMaskStrategy
            model.mask_strategy_obj = AttentionBasedMaskStrategy(args.mask_ratio)
        elif args.mask_strategy == 'position_center':
            from llava.model.llava_arch_masked import PositionBasedMaskStrategy
            model.mask_strategy_obj = PositionBasedMaskStrategy(args.mask_ratio, mask_center=True)
        elif args.mask_strategy == 'position_edge':
            from llava.model.llava_arch_masked import PositionBasedMaskStrategy
            model.mask_strategy_obj = PositionBasedMaskStrategy(args.mask_ratio, mask_center=False)
        
        print(f"启用视觉token mask: 比例={args.mask_ratio}, 策略={args.mask_strategy}")
    else:
        model.config.mask_visual_token = False
    
    # 评估模型
    results = evaluate_model_on_gqa(
        model, tokenizer, image_processor, gqa_loader, args
    )
    
    # 保存结果
    save_results(results, args.output_dir, args)
    
    print("评估完成!")


if __name__ == "__main__":
    main()