# LLaVA-1.5 视觉Token Mask消融实验

本项目基于LLaVA-1.5实现了视觉token随机mask功能，用于研究视觉token对多模态大语言模型性能的影响。

## 功能特性

### 1. 多种Mask策略
- **随机mask**: 随机选择视觉token进行mask
- **基于注意力的mask**: 根据注意力权重选择要mask的token
- **基于位置的mask**: 根据token位置选择要mask的区域（中心或边缘）

### 2. 参数化控制
- `--mask_visual_token`: 是否启用视觉token mask
- `--mask_ratio`: mask比例 (0.0-1.0)
- `--mask_strategy`: mask策略选择
- `--mask_token_value`: mask后的token值

### 3. 注意力权重可视化
- 支持多层注意力权重可视化
- 支持特定头的注意力分析
- 支持对视觉token的注意力分析

## 文件结构

```
llava15_test/LLaVA/
├── llava/model/
│   ├── llava_arch_masked.py          # 支持mask的LLaVA架构
│   └── language_model/
│       └── llava_llama_masked.py     # 支持mask的LLaVA模型
├── llava/utils/
│   └── attention_visualizer.py        # 注意力可视化工具
├── eval_gqa_masked.py                 # GQA评估脚本
├── run_ablation_experiments.py       # 消融实验脚本
├── test_masked_model.py              # 测试脚本
└── README_MASKED.md                  # 本说明文档
```

## 安装依赖

```bash
# 激活环境
conda activate llava15

# 安装额外依赖
pip install matplotlib seaborn pandas
```

## 使用方法

### 1. 测试模型功能

```bash
# 运行测试脚本
python test_masked_model.py
```

### 2. 单次GQA评估

```bash
# 基线评估（无mask）
python eval_gqa_masked.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --gqa_questions /path/to/gqa/questions.json \
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \
    --images_path /path/to/gqa/images \
    --num_samples 1000 \
    --output_dir ./baseline_results

# 带mask的评估
python eval_gqa_masked.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --gqa_questions /path/to/gqa/questions.json \
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \
    --images_path /path/to/gqa/images \
    --num_samples 1000 \
    --output_dir ./masked_results \
    --mask_visual_token \
    --mask_ratio 0.2 \
    --mask_strategy random \
    --mask_token_value 0.0
```

### 3. 运行消融实验

```bash
# 运行完整的消融实验
python run_ablation_experiments.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --gqa_questions /path/to/gqa/questions.json \
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \
    --images_path /path/to/gqa/images \
    --output_dir ./ablation_results \
    --num_samples 1000 \
    --mask_ratios "0.0,0.1,0.2,0.3,0.5" \
    --mask_strategies "random,attention,position_center,position_edge" \
    --visualize_results \
    --save_plots
```

### 4. 注意力权重可视化

```bash
# 启用注意力可视化
python eval_gqa_masked.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --gqa_questions /path/to/gqa/questions.json \
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \
    --images_path /path/to/gqa/images \
    --visualize_attention \
    --attention_layers "0,6,12,18" \
    --attention_heads "0,4,8,12" \
    --save_attention_dir ./attention_viz
```

## 参数说明

### 评估脚本参数 (eval_gqa_masked.py)

#### 基础参数
- `--model_path`: 模型路径
- `--gqa_questions`: GQA问题文件路径
- `--gqa_scene_graphs`: GQA场景图文件路径
- `--images_path`: 图像文件夹路径
- `--num_samples`: 评估样本数量
- `--output_dir`: 结果保存目录

#### Mask参数
- `--mask_visual_token`: 启用视觉token mask
- `--mask_ratio`: mask比例 (0.0-1.0)
- `--mask_strategy`: mask策略 (random/attention/position_center/position_edge)
- `--mask_token_value`: mask后的token值

#### 可视化参数
- `--visualize_attention`: 启用注意力可视化
- `--attention_layers`: 要可视化的层索引
- `--attention_heads`: 要可视化的头索引
- `--save_attention_dir`: 注意力可视化保存目录

### 消融实验参数 (run_ablation_experiments.py)

- `--mask_ratios`: 要测试的mask比例列表
- `--mask_strategies`: 要测试的mask策略列表
- `--mask_token_values`: 要测试的mask token值列表
- `--visualize_results`: 生成结果可视化
- `--save_plots`: 保存图表

## 输出结果

### 1. 评估结果
- `summary.json`: 摘要结果（准确率、正确数、总数）
- `detailed_results.json`: 详细结果（每个样本的预测和标签）

### 2. 消融实验结果
- `ablation_results.csv`: 所有实验结果的CSV文件
- `ablation_report.md`: 实验报告
- `plots/`: 可视化图表目录
  - `strategy_comparison.png`: 策略对比图
  - `ratio_impact.png`: 比例影响图
  - `heatmap.png`: 策略vs比例热力图
  - `baseline_comparison.png`: 与基线对比图

### 3. 注意力可视化
- `attention_layer{X}_head{Y}.png`: 特定层和头的注意力热力图
- `visual_attention_layer{X}.png`: 对视觉token的注意力分析
- `multi_layer_attention.png`: 多层注意力对比
- `attention_statistics.png`: 注意力统计信息

## 实验设计

### 1. 基线实验
- 无mask的原始模型性能

### 2. Mask比例实验
- 测试不同mask比例 (0.1, 0.2, 0.3, 0.5) 对性能的影响

### 3. Mask策略实验
- 随机mask: 随机选择要mask的token
- 注意力mask: 基于注意力权重选择要mask的token
- 位置mask: 基于位置选择要mask的区域

### 4. Mask值实验
- 测试不同mask token值 (0.0, -1.0) 的影响

## 验证步骤

### 1. 功能验证
```bash
# 运行测试脚本验证基本功能
python test_masked_model.py
```

### 2. 小规模验证
```bash
# 使用少量样本验证评估流程
python eval_gqa_masked.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --gqa_questions /path/to/gqa/questions.json \
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \
    --images_path /path/to/gqa/images \
    --num_samples 10 \
    --mask_visual_token \
    --mask_ratio 0.1 \
    --mask_strategy random
```

### 3. 完整实验验证
```bash
# 运行小规模消融实验
python run_ablation_experiments.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --gqa_questions /path/to/gqa/questions.json \
    --gqa_scene_graphs /path/to/gqa/scene_graphs.json \
    --images_path /path/to/gqa/images \
    --num_samples 100 \
    --mask_ratios "0.0,0.1,0.2" \
    --mask_strategies "random,attention"
```

## 注意事项

1. **内存使用**: 注意力可视化会增加内存使用，建议在GPU内存充足时使用
2. **实验时间**: 完整消融实验可能需要较长时间，建议先用小样本测试
3. **数据路径**: 确保GQA数据集路径正确
4. **模型兼容性**: 当前实现基于LLaVA-1.5，其他模型可能需要适配

## 故障排除

### 1. 导入错误
```bash
# 确保在正确的环境中
conda activate llava15
# 检查Python路径
python -c "import sys; print(sys.path)"
```

### 2. 内存不足
```bash
# 减少batch size或样本数量
--num_samples 100
```

### 3. 可视化错误
```bash
# 安装可视化依赖
pip install matplotlib seaborn
```

## 扩展功能

### 1. 添加新的Mask策略
在 `llava_arch_masked.py` 中继承 `MaskStrategy` 类并实现 `get_mask_indices` 方法。

### 2. 添加新的可视化功能
在 `attention_visualizer.py` 中添加新的可视化方法。

### 3. 支持其他数据集
修改评估脚本以支持其他视觉问答数据集。
