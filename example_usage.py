#!/usr/bin/env python3
"""
LLaVA-1.5 视觉Token Mask使用示例
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("="*60)
    print("示例1: 基本使用方法")
    print("="*60)
    
    print("""
# 1. 测试模型功能
python test_masked_model.py

# 2. 基线评估（无mask）
python eval_gqa_masked.py \\
    --model_path liuhaotian/llava-v1.5-7b \\
    --gqa_questions /path/to/gqa/questions.json \\
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \\
    --images_path /path/to/gqa/images \\
    --num_samples 100 \\
    --output_dir ./baseline_results

# 3. 带mask的评估
python eval_gqa_masked.py \\
    --model_path liuhaotian/llava-v1.5-7b \\
    --gqa_questions /path/to/gqa/questions.json \\
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \\
    --images_path /path/to/gqa/images \\
    --num_samples 100 \\
    --output_dir ./masked_results \\
    --mask_visual_token \\
    --mask_ratio 0.2 \\
    --mask_strategy random \\
    --mask_token_value 0.0
    """)


def example_2_ablation_study():
    """示例2: 消融实验"""
    print("="*60)
    print("示例2: 消融实验")
    print("="*60)
    
    print("""
# 运行完整的消融实验
python run_ablation_experiments.py \\
    --model_path liuhaotian/llava-v1.5-7b \\
    --gqa_questions /path/to/gqa/questions.json \\
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \\
    --images_path /path/to/gqa/images \\
    --output_dir ./ablation_results \\
    --num_samples 1000 \\
    --mask_ratios "0.0,0.1,0.2,0.3,0.5" \\
    --mask_strategies "random,attention,position_center,position_edge" \\
    --visualize_results \\
    --save_plots
    """)


def example_3_attention_visualization():
    """示例3: 注意力可视化"""
    print("="*60)
    print("示例3: 注意力可视化")
    print("="*60)
    
    print("""
# 启用注意力可视化
python eval_gqa_masked.py \\
    --model_path liuhaotian/llava-v1.5-7b \\
    --gqa_questions /path/to/gqa/questions.json \\
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \\
    --images_path /path/to/gqa/images \\
    --num_samples 10 \\
    --visualize_attention \\
    --attention_layers "0,6,12,18" \\
    --attention_heads "0,4,8,12" \\
    --save_attention_dir ./attention_viz
    """)


def example_4_different_strategies():
    """示例4: 不同mask策略对比"""
    print("="*60)
    print("示例4: 不同mask策略对比")
    print("="*60)
    
    strategies = [
        ("random", "随机mask策略"),
        ("attention", "基于注意力的mask策略"),
        ("position_center", "基于位置的mask策略（中心区域）"),
        ("position_edge", "基于位置的mask策略（边缘区域）")
    ]
    
    for strategy, description in strategies:
        print(f"""
# {description}
python eval_gqa_masked.py \\
    --model_path liuhaotian/llava-v1.5-7b \\
    --gqa_questions /path/to/gqa/questions.json \\
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \\
    --images_path /path/to/gqa/images \\
    --num_samples 100 \\
    --output_dir ./results_{strategy} \\
    --mask_visual_token \\
    --mask_ratio 0.2 \\
    --mask_strategy {strategy} \\
    --mask_token_value 0.0
        """)


def example_5_parameter_sweep():
    """示例5: 参数扫描"""
    print("="*60)
    print("示例5: 参数扫描")
    print("="*60)
    
    print("""
# 测试不同mask比例
for ratio in 0.0 0.1 0.2 0.3 0.5; do
    python eval_gqa_masked.py \\
        --model_path liuhaotian/llava-v1.5-7b \\
        --gqa_questions /path/to/gqa/questions.json \\
        --gqa_scene_graphs /path/to/gqa/scene_graphs.json \\
        --images_path /path/to/gqa/images \\
        --num_samples 100 \\
        --output_dir ./results_ratio_${ratio} \\
        --mask_visual_token \\
        --mask_ratio ${ratio} \\
        --mask_strategy random \\
        --mask_token_value 0.0
done

# 测试不同mask token值
for value in 0.0 -1.0 -0.5; do
    python eval_gqa_masked.py \\
        --model_path liuhaotian/llava-v1.5-7b \\
        --gqa_questions /path/to/gqa/questions.json \\
        --gqa_scene_graphs /path/to/gqa/scene_graphs.json \\
        --images_path /path/to/gqa/images \\
        --num_samples 100 \\
        --output_dir ./results_value_${value} \\
        --mask_visual_token \\
        --mask_ratio 0.2 \\
        --mask_strategy random \\
        --mask_token_value ${value}
done
    """)


def main():
    """主函数"""
    print("LLaVA-1.5 视觉Token Mask使用示例")
    print("="*60)
    
    example_1_basic_usage()
    example_2_ablation_study()
    example_3_attention_visualization()
    example_4_different_strategies()
    example_5_parameter_sweep()
    
    print("\n" + "="*60)
    print("注意事项:")
    print("1. 确保GQA数据集路径正确")
    print("2. 根据GPU内存调整batch size和样本数量")
    print("3. 注意力可视化会增加内存使用")
    print("4. 完整消融实验可能需要较长时间")
    print("5. 建议先用小样本测试流程")
    print("="*60)


if __name__ == "__main__":
    main()
