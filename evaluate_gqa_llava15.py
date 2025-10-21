#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_gqa_lmms.py
评估 LLaVA-1.5（或其他 MLLM）在 lmms-lab/GQA 上的表现。
支持宽松匹配 (semantic fuzzy match)。
"""

import argparse
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter
from PIL import Image
import torch
from eval.pred_gt_match import compute_match


# ==== 辅助函数 ====

def normalize_text(s: str) -> str:
    """去除标点、大小写、前后空格"""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def is_match(pred, gold):
    """宽松匹配规则：完全相等、或包含关系"""
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    return (pred_n == gold_n) or (gold_n in pred_n) or (pred_n in gold_n)

# ==== 主流程 ====

def evaluate_gqa(model, tokenizer, image_processor, model_name="llava-v1.5-7b",
                 split="testdev", output_json="gqa_eval_results.json",
                 max_samples=None, device="cuda"):

    print(f"🔹 正在加载数据集 lmms-lab/GQA ({split}) ...")
    dataset = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split=split)

    results = []
    correct, total = 0, 0

    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        if max_samples and i >= max_samples:
            break

        image: Image.Image = sample["image"].convert("RGB")
        question = sample["question"]
        gold_answer = sample["answer"]

        # 预处理图像
        image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float16)

        # 构造输入（LLaVA 风格）
        prompt = f"<image>\n{question}"
        input_ids = tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                images=[image_tensor],          # 注意要用 list
                do_sample=False,
                temperature=0,
                max_new_tokens=64,
            )

        pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # 计算匹配
        total += 1
        if is_match(pred_text, gold_answer):
            correct += 1

        results.append({
            "id": sample["id"],
            "question": question,
            "ground_truth": gold_answer,
            "prediction": pred_text,
            "correct": is_match(pred_text, gold_answer)
        })

        if (i + 1) % 50 == 0:
            acc = correct / total * 100
            print(f"  已评估 {i+1} 个样本，当前准确率：{acc:.2f}%")

    # 计算最终结果
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n✅ 评估完成：总样本数 {total}，准确率 = {accuracy:.2f}%")

    # 输出结果
    output = {
        "model": model_name,
        "split": split,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "results": results
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存至 {output_json}")

    # 可选：输出答案分布（方便分析）
    gold_counts = Counter(normalize_text(s["answer"]) for s in dataset)
    pred_counts = Counter(normalize_text(r["prediction"]) for r in results)
    print(f"\nTop-10 gold答案分布: {gold_counts.most_common(10)}")
    print(f"Top-10 预测答案分布: {pred_counts.most_common(10)}")

    return output

# ==== 命令行入口 ====

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型路径，如 liuhaotian/llava-v1.5-7b")
    parser.add_argument("--split", type=str, default="testdev",
                        help="GQA 数据划分，如 testdev / val")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="仅评估前 N 条样本（调试用）")
    parser.add_argument("--output", type=str, default="gqa_eval_results.json")
    args = parser.parse_args()

    # ===== 加载模型 =====
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "cuda"
    )

    evaluate_gqa(model, tokenizer, image_processor,
                 model_name=args.model_path,
                 split=args.split,
                 output_json=args.output,
                 max_samples=args.max_samples)
