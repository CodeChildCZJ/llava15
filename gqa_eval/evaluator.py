# gqa_eval/evaluator.py
import os, json, torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from pred_gt_match import compute_match_batch
from prompt import build_multimodal_batch_inputs


@torch.no_grad()
def eval_on_gqa(model, tokenizer, image_processor, gqa_loader, args):
    """
    高性能 GQA 评估：
    - DataLoader 内部多线程解码
    - prompt + tokenizer 批量化
    - image_processor 并行化
    - compute_match_batch 向量化匹配
    """
    print(f"Evaluating split={args.split}, batch_size={gqa_loader.batch_size}, num_workers={gqa_loader.num_workers}")
    dataloader = gqa_loader.as_dataloader(args.split, args.max_samples)
    model_to_use = model.module if hasattr(model, "module") else model

    counters = {k: Counter() for k in ["llava_match", "lmms_match", "loose_match"]}
    correct_loose, total = 0, 0
    results = []

    for batch in tqdm(dataloader, desc="Evaluating GQA"):
        # ============================================================
        # 构建批量输入（文本+图像）
        # ============================================================
        input_ids_batch, image_tensors = build_multimodal_batch_inputs(
            batch, tokenizer, image_processor, args.device, torch.float16, args.conv_mode
        )

        # Ground truth 与问题
        gts = [s["answer"].strip().lower() for s in batch if s.get("answer")]
        questions = [s["question"] for s in batch]

        # ============================================================
        # 模型生成
        # ============================================================
        outputs = model_to_use.generate(
            inputs=input_ids_batch,
            images=[img for img in image_tensors],
            max_new_tokens=args.max_new_tokens,
            temperature=0,
            do_sample=False,
        )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [p.strip().lower() for p in preds]

        # ============================================================
        # 批量匹配 + 向量化统计
        # ============================================================
        batch_matches = compute_match_batch(preds, gts)

        llava_match = np.fromiter((m["llava_match"] for m in batch_matches), dtype=bool)
        lmms_match = np.fromiter((m["lmms_match"] for m in batch_matches), dtype=bool)
        loose_match = np.fromiter((m["loose_match"] for m in batch_matches), dtype=bool)

        is_correct = lmms_match | loose_match

        # 向量化统计更新
        counters["llava_match"].update({True: llava_match.sum(), False: len(llava_match) - llava_match.sum()})
        counters["lmms_match"].update({True: lmms_match.sum(), False: len(lmms_match) - lmms_match.sum()})
        counters["loose_match"].update({True: loose_match.sum(), False: len(loose_match) - loose_match.sum()})

        correct_loose += int(is_correct.sum())
        total += len(is_correct)

        if args.save_pred_gt:
            for i in range(len(preds)):
                results.append({
                    "question": questions[i],
                    "pred": preds[i],
                    "gt": gts[i],
                    "matches": batch_matches[i],
                    "is_correct": bool(is_correct[i])
                })

    # ============================================================
    # 汇总与保存
    # ============================================================
    acc = correct_loose / total if total > 0 else 0.0
    summary = {
        "total": total,
        "correct": correct_loose,
        "accuracy": acc,
        "per_match_counts": {k: dict(counters[k]) for k in counters},
        "mask_settings": {  # 新增部分
        "enabled": args.mask_visual_token,
        "ratio": args.mask_ratio,
        "strategy": args.mask_strategy,
        "token_value": args.mask_token_value,
    }
    }

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_json:
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else o)

    if args.save_pred_gt:
        with open(os.path.join(args.output_dir, "pred_gt.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else o)


    print(f"\n完成: {total} 样本, 最宽松准确率={acc*100:.2f}%")
    for k, v in summary["per_match_counts"].items():
        print(f"{k}: {v}")

    return summary
