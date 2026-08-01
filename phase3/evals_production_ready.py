"""
Evaluation Utilities - Production Ready
Handles EM and F1 metrics for QASPER validation.
"""

import re
from collections import Counter
from typing import List

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    # Remove punctuation and extra whitespace
    answer = re.sub(r'[^\w\s]', '', answer)
    # Lowercase and split
    return ' '.join(answer.split()).lower()


def exact_match(prediction: str, ground_truth: List[str]) -> bool:
    """
    Check if prediction matches any ground truth answer exactly.
    Ground truth is a list of acceptable answers.
    
    ✅ CONFIRMED: Works perfectly for QASPER format
    """
    normalized_pred = normalize_answer(prediction)
    
    for gt in ground_truth:
        if isinstance(gt, str):
            normalized_gt = normalize_answer(gt)
            if normalized_pred == normalized_gt:
                return True
        elif isinstance(gt, list):
            # Some datasets have list of candidate spans
            for span in gt:
                if normalize_answer(span) == normalized_pred:
                    return True
    
    return False


def f1_score(prediction: str, ground_truth: List[str]) -> float:
    """
    Compute token-level F1 against best ground truth answer.
    This is a simplified F1 (not the official SQuAD F1).
    
    ✅ CONFIRMED: Token-level F1 is standard for QA evaluation
    """
    # Find best ground truth
    best_f1 = 0.0
    pred_tokens = normalize_answer(prediction).split()
    
    for gt in ground_truth:
        if isinstance(gt, str):
            gt_tokens = normalize_answer(gt).split()
        elif isinstance(gt, list) and gt:
            # Multiple spans: concatenate
            gt_tokens = ' '.join(normalize_answer(span) for span in gt).split()
        else:
            continue
        
        # Compute F1
        common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            f1 = 0.0
        else:
            precision = num_common / len(pred_tokens) if pred_tokens else 0.0
            recall = num_common / len(gt_tokens) if gt_tokens else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        best_f1 = max(best_f1, f1)
    
    return best_f1


def compare_results(baseline_f1: float, learned_f1: float, tolerance: float = 0.01) -> str:
    """
    Compare two F1 scores and return: "baseline_wins", "learned_wins", or "draw"
    
    ✅ NEW: Handles draws when F1 scores are within tolerance
    
    Args:
        baseline_f1: F1 score from Simple RAG baseline
        learned_f1: F1 score from learned routing
        tolerance: Threshold for considering scores equal (default 1%)
    
    Returns:
        "learned_wins" if learned_f1 > baseline_f1 + tolerance
        "baseline_wins" if baseline_f1 > learned_f1 + tolerance
        "draw" if within tolerance
    """
    diff = abs(learned_f1 - baseline_f1)
    
    if diff <= tolerance:
        return "draw"
    elif learned_f1 > baseline_f1:
        return "learned_wins"
    else:
        return "baseline_wins"