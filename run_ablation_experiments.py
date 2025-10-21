#!/usr/bin/env python3
"""
消融实验脚本 - 测试不同mask策略和参数对GQA性能的影响
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_args():
    parser = argparse.ArgumentParser(description="消融实验脚本")
    
    # 基础参数
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--gqa_data_path", type=str, required=True, help="GQA数据集路径")
    parser.add_argument("--gqa_questions", type=str, required=True, help="GQA问题文件路径")
    parser.add_argument("--gqa_scene_graphs", type=str, required=True, help="GQA场景图文件路径")
    parser.add_argument("--images_path", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--output_dir", type=str, default="./ablation_results", help="结果保存目录")
    
    # 实验参数
    parser.add_argument("--num_samples", type=int, default=1000, help="每个实验的样本数量")
    parser.add_argument("--mask_ratios", type=str, default="0.0,0.1,0.2,0.3,0.5", help="要测试的mask比例")
    parser.add_argument("--mask_strategies", type=str, default="random,attention,position_center,position_edge", 
                       help="要测试的mask策略")
    parser.add_argument("--mask_token_values", type=str, default="0.0,-1.0", help="要测试的mask token值")
    
    # 可视化参数
    parser.add_argument("--visualize_results", action="store_true", help="是否生成结果可视化")
    parser.add_argument("--save_plots", action="store_true", help="是否保存图表")
    
    return parser.parse_args()


def run_single_experiment(
    model_path: str,
    gqa_questions: str,
    gqa_scene_graphs: str,
    images_path: str,
    output_dir: str,
    num_samples: int,
    mask_visual_token: bool = False,
    mask_ratio: float = 0.0,
    mask_strategy: str = "random",
    mask_token_value: float = 0.0
) -> Dict:
    """运行单个实验"""
    
    # 构建实验名称
    if mask_visual_token:
        exp_name = f"mask_{mask_strategy}_ratio{mask_ratio}_value{mask_token_value}"
    else:
        exp_name = "baseline"
    
    exp_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # 构建命令
    cmd = [
        "python", "eval_gqa_masked.py",
        "--model_path", model_path,
        "--gqa_questions", gqa_questions,
        "--gqa_scene_graphs", gqa_scene_graphs,
        "--images_path", images_path,
        "--output_dir", exp_output_dir,
        "--num_samples", str(num_samples)
    ]
    
    if mask_visual_token:
        cmd.extend([
            "--mask_visual_token",
            "--mask_ratio", str(mask_ratio),
            "--mask_strategy", mask_strategy,
            "--mask_token_value", str(mask_token_value)
        ])
    
    print(f"运行实验: {exp_name}")
    print(f"命令: {' '.join(cmd)}")
    
    # 运行实验
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"实验 {exp_name} 完成")
        
        # 读取结果
        summary_file = os.path.join(exp_output_dir, "summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                results = json.load(f)
            return results
        else:
            print(f"警告: 未找到结果文件 {summary_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"实验 {exp_name} 失败: {e}")
        print(f"错误输出: {e.stderr}")
        return None


def run_ablation_experiments(args):
    """运行消融实验"""
    
    # 解析参数
    mask_ratios = [float(x) for x in args.mask_ratios.split(',')]
    mask_strategies = args.mask_strategies.split(',')
    mask_token_values = [float(x) for x in args.mask_token_values.split(',')]
    
    all_results = []
    
    # 1. 基线实验（无mask）
    print("运行基线实验...")
    baseline_result = run_single_experiment(
        args.model_path,
        args.gqa_questions,
        args.gqa_scene_graphs,
        args.images_path,
        args.output_dir,
        args.num_samples,
        mask_visual_token=False
    )
    if baseline_result:
        all_results.append(baseline_result)
    
    # 2. 不同mask比例实验
    print("运行不同mask比例实验...")
    for ratio in mask_ratios:
        if ratio == 0.0:  # 跳过基线
            continue
            
        for strategy in mask_strategies:
            for token_value in mask_token_values:
                result = run_single_experiment(
                    args.model_path,
                    args.gqa_questions,
                    args.gqa_scene_graphs,
                    args.images_path,
                    args.output_dir,
                    args.num_samples,
                    mask_visual_token=True,
                    mask_ratio=ratio,
                    mask_strategy=strategy,
                    mask_token_value=token_value
                )
                if result:
                    all_results.append(result)
    
    return all_results


def analyze_results(results: List[Dict], output_dir: str):
    """分析实验结果"""
    
    if not results:
        print("没有结果可分析")
        return
    
    # 转换为DataFrame
    df_data = []
    for result in results:
        row = {
            'accuracy': result['accuracy'],
            'correct': result['correct'],
            'total': result['total'],
            'mask_visual_token': result['mask_config']['mask_visual_token'],
            'mask_ratio': result['mask_config']['mask_ratio'],
            'mask_strategy': result['mask_config']['mask_strategy'],
            'mask_token_value': result['mask_config']['mask_token_value']
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # 保存原始数据
    df.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)
    
    # 打印摘要
    print("\n" + "="*60)
    print("消融实验结果摘要:")
    print("="*60)
    print(df.to_string(index=False))
    
    # 分析不同策略的效果
    print("\n按策略分组的平均准确率:")
    strategy_accuracy = df.groupby('mask_strategy')['accuracy'].agg(['mean', 'std', 'count'])
    print(strategy_accuracy)
    
    # 分析不同比例的效果
    print("\n按mask比例分组的平均准确率:")
    ratio_accuracy = df.groupby('mask_ratio')['accuracy'].agg(['mean', 'std', 'count'])
    print(ratio_accuracy)
    
    return df


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """创建可视化图表"""
    
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # 1. 不同mask策略的准确率对比
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='mask_strategy', y='accuracy', hue='mask_ratio')
    plt.title('不同Mask策略的准确率对比')
    plt.xlabel('Mask策略')
    plt.ylabel('准确率')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "strategy_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Mask比例对准确率的影响
    plt.figure(figsize=(12, 8))
    for strategy in df['mask_strategy'].unique():
        strategy_data = df[df['mask_strategy'] == strategy]
        plt.plot(strategy_data['mask_ratio'], strategy_data['accuracy'], 
                marker='o', label=strategy, linewidth=2, markersize=8)
    
    plt.xlabel('Mask比例')
    plt.ylabel('准确率')
    plt.title('Mask比例对准确率的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "ratio_impact.png"), dpi=300)
    plt.close()
    
    # 3. 热力图：策略vs比例
    pivot_table = df.pivot_table(values='accuracy', index='mask_strategy', columns='mask_ratio', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Mask策略和比例的热力图')
    plt.xlabel('Mask比例')
    plt.ylabel('Mask策略')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "heatmap.png"), dpi=300)
    plt.close()
    
    # 4. 基线对比
    baseline_acc = df[df['mask_visual_token'] == False]['accuracy'].iloc[0] if len(df[df['mask_visual_token'] == False]) > 0 else None
    if baseline_acc is not None:
        plt.figure(figsize=(12, 8))
        df_masked = df[df['mask_visual_token'] == True]
        for strategy in df_masked['mask_strategy'].unique():
            strategy_data = df_masked[df_masked['mask_strategy'] == strategy]
            plt.plot(strategy_data['mask_ratio'], strategy_data['accuracy'], 
                    marker='o', label=strategy, linewidth=2, markersize=8)
        
        plt.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2, label=f'基线 (无mask): {baseline_acc:.3f}')
        plt.xlabel('Mask比例')
        plt.ylabel('准确率')
        plt.title('与基线的对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "baseline_comparison.png"), dpi=300)
        plt.close()
    
    print(f"可视化图表已保存到: {os.path.join(output_dir, 'plots')}")


def generate_report(df: pd.DataFrame, output_dir: str):
    """生成实验报告"""
    
    report_path = os.path.join(output_dir, "ablation_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 视觉Token Mask消融实验报告\n\n")
        
        # 实验概述
        f.write("## 实验概述\n\n")
        f.write(f"- 总实验数: {len(df)}\n")
        f.write(f"- 基线准确率: {df[df['mask_visual_token'] == False]['accuracy'].iloc[0]:.4f}\n")
        f.write(f"- 最高准确率: {df['accuracy'].max():.4f}\n")
        f.write(f"- 最低准确率: {df['accuracy'].min():.4f}\n\n")
        
        # 最佳配置
        best_result = df.loc[df['accuracy'].idxmax()]
        f.write("## 最佳配置\n\n")
        f.write(f"- 准确率: {best_result['accuracy']:.4f}\n")
        f.write(f"- Mask策略: {best_result['mask_strategy']}\n")
        f.write(f"- Mask比例: {best_result['mask_ratio']}\n")
        f.write(f"- Mask token值: {best_result['mask_token_value']}\n\n")
        
        # 策略分析
        f.write("## 策略分析\n\n")
        strategy_stats = df.groupby('mask_strategy')['accuracy'].agg(['mean', 'std', 'count'])
        f.write("| 策略 | 平均准确率 | 标准差 | 实验次数 |\n")
        f.write("|------|------------|--------|----------|\n")
        for strategy, stats in strategy_stats.iterrows():
            f.write(f"| {strategy} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['count']} |\n")
        f.write("\n")
        
        # 比例分析
        f.write("## 比例分析\n\n")
        ratio_stats = df.groupby('mask_ratio')['accuracy'].agg(['mean', 'std', 'count'])
        f.write("| 比例 | 平均准确率 | 标准差 | 实验次数 |\n")
        f.write("|------|------------|--------|----------|\n")
        for ratio, stats in ratio_stats.iterrows():
            f.write(f"| {ratio} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['count']} |\n")
        f.write("\n")
        
        # 结论
        f.write("## 结论\n\n")
        baseline_acc = df[df['mask_visual_token'] == False]['accuracy'].iloc[0]
        best_masked_acc = df[df['mask_visual_token'] == True]['accuracy'].max()
        
        if best_masked_acc > baseline_acc:
            f.write(f"视觉token mask可以提升模型性能，最佳提升: {best_masked_acc - baseline_acc:.4f}\n")
        else:
            f.write(f"视觉token mask降低了模型性能，最大下降: {baseline_acc - best_masked_acc:.4f}\n")
    
    print(f"实验报告已保存到: {report_path}")


def main():
    args = setup_args()
    
    print("开始消融实验...")
    print(f"参数配置: {vars(args)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    results = run_ablation_experiments(args)
    
    if not results:
        print("没有成功完成任何实验")
        return
    
    # 分析结果
    df = analyze_results(results, args.output_dir)
    
    # 创建可视化
    if args.visualize_results:
        create_visualizations(df, args.output_dir)
    
    # 生成报告
    generate_report(df, args.output_dir)
    
    print("消融实验完成!")


if __name__ == "__main__":
    main()




