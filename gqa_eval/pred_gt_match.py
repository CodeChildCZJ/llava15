# gqa_eval/pred_gt_match.py
import re
import string
import numpy as np


def normalize_text(s: str):
    """SQuAD-style normalization (same as lmms-eval)."""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_llava(pred, gt):
    """Strict match: LLaVA official"""
    return pred.strip().lower() == gt.strip().lower()


def exact_match_lmms(pred, gt):
    """SQuAD-style match"""
    return normalize_text(pred) == normalize_text(gt)


def keyword_match(pred, gt):
    """Loose substring match (防止空串误判)"""
    pred, gt = normalize_text(pred), normalize_text(gt)
    if not pred or not gt:
        return False
    return (gt in pred) or (pred in gt)


def compute_match(predicted_answer, ground_truth):
    """单样本匹配"""
    llava = exact_match_llava(predicted_answer, ground_truth)
    lmms = exact_match_lmms(predicted_answer, ground_truth)
    loose = keyword_match(predicted_answer, ground_truth)
    return {
        "llava_match": llava,
        "lmms_match": lmms,
        "loose_match": loose,
    }


def compute_match_batch(preds, gts):
    """批量匹配 (NumPy 向量化 + 防空字符串)"""
    preds = np.array([str(p).strip() if p else "" for p in preds], dtype=object)
    gts   = np.array([str(g).strip() if g else "" for g in gts], dtype=object)

    matches = []
    for p, g in zip(preds, gts):
        matches.append(compute_match(p, g))
    return matches
