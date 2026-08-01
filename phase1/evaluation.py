"""
Evaluation utilities for Phase 1.
- Exact Match (EM): Is prediction one of the candidate answers?
- F1 Score: Token-level F1 between prediction and best candidate answer.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import re
from collections import Counter


@dataclass
class StrategyResult:
    """Result for a single strategy on a single question."""
    strategy_name: str
    question_id: str
    prediction: str
    em: float
    f1: float
    latency: float
    token_count: int
    metadata: Dict


class Evaluator:
    """Computes EM and F1 metrics."""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        # Remove punctuation and extra whitespace
        answer = re.sub(r'[^\w\s]', '', answer)
        # Lowercase and split
        return ' '.join(answer.split()).lower()
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: List[str]) -> bool:
        """
        Check if prediction matches any ground truth answer exactly.
        Ground truth is a list of acceptable answers.
        """
        normalized_pred = Evaluator.normalize_answer(prediction)
        
        for gt in ground_truth:
            if isinstance(gt, str):
                normalized_gt = Evaluator.normalize_answer(gt)
                if normalized_pred == normalized_gt:
                    return True
            elif isinstance(gt, list):
                # Some datasets have list of candidate spans
                for span in gt:
                    if Evaluator.normalize_answer(span) == normalized_pred:
                        return True
        
        return False
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: List[str]) -> float:
        """
        Compute token-level F1 against best ground truth answer.
        This is a simplified F1 (not the official SQuAD F1).
        """
        # Find best ground truth
        best_f1 = 0.0
        pred_tokens = Evaluator.normalize_answer(prediction).split()
        
        for gt in ground_truth:
            if isinstance(gt, str):
                gt_tokens = Evaluator.normalize_answer(gt).split()
            elif isinstance(gt, list) and gt:
                # Multiple spans: concatenate
                gt_tokens = ' '.join(Evaluator.normalize_answer(span) for span in gt).split()
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
    
    def compute_metrics(self, prediction: str, ground_truth: List[str]) -> Tuple[float, float]:
        """
        Compute EM and F1 for a single prediction.
        
        Args:
            prediction: Model's answer
            ground_truth: List of acceptable answers
        
        Returns:
            (em, f1) scores in [0, 1]
        """
        em = 1.0 if self.exact_match(prediction, ground_truth) else 0.0
        f1 = self.f1_score(prediction, ground_truth)
        
        return em, f1


class QuestionMetrics:
    """Metrics aggregated across all strategies for a single question."""
    
    def __init__(self, question_id: str):
        self.question_id = question_id
        self.results: Dict[str, StrategyResult] = {}
    
    def add_result(self, result: StrategyResult):
        """Add strategy result."""
        self.results[result.strategy_name] = result
    
    def best_strategy_em(self) -> Tuple[str, float]:
        """Return strategy with highest EM (F1 as tiebreaker)."""
        if not self.results:
            return None, 0.0
        
        best_strat = max(self.results.keys(), key=lambda s: (self.results[s].em, self.results[s].f1))
        return best_strat, self.results[best_strat].em
    
    def best_strategy_f1(self) -> Tuple[str, float]:
        """Return strategy with highest F1."""
        if not self.results:
            return None, 0.0
        
        best_strat = max(self.results.keys(), key=lambda s: self.results[s].f1)
        return best_strat, self.results[best_strat].f1
    
    def fastest_strategy(self) -> Tuple[str, float]:
        """Return strategy with lowest latency."""
        if not self.results:
            return None, float('inf')
        
        best_strat = min(self.results.keys(), key=lambda s: self.results[s].latency)
        return best_strat, self.results[best_strat].latency
    
    def cheapest_strategy(self) -> Tuple[str, int]:
        """Return strategy with lowest token count."""
        if not self.results:
            return None, float('inf')
        
        best_strat = min(self.results.keys(), key=lambda s: self.results[s].token_count)
        return best_strat, self.results[best_strat].token_count


class AggregateMetrics:
    """Aggregate metrics across all 100 questions."""
    
    def __init__(self):
        self.question_metrics: Dict[str, QuestionMetrics] = {}
    
    def add_question_metrics(self, qm: QuestionMetrics):
        """Add metrics for a single question."""
        self.question_metrics[qm.question_id] = qm
    
    def strategy_stats(self, strategy_name: str) -> Dict:
        """Compute mean/std/median for a strategy across all questions."""
        ems = []
        f1s = []
        latencies = []
        token_counts = []
        
        for qm in self.question_metrics.values():
            if strategy_name in qm.results:
                result = qm.results[strategy_name]
                ems.append(result.em)
                f1s.append(result.f1)
                latencies.append(result.latency)
                token_counts.append(result.token_count)
        
        import numpy as np
        
        return {
            "strategy": strategy_name,
            "num_questions": len(ems),
            "em_mean": float(np.mean(ems)) if ems else 0.0,
            "em_std": float(np.std(ems)) if ems else 0.0,
            "em_min": float(np.min(ems)) if ems else 0.0,
            "em_max": float(np.max(ems)) if ems else 0.0,
            "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
            "f1_std": float(np.std(f1s)) if f1s else 0.0,
            "latency_median": float(np.median(latencies)) if latencies else 0.0,
            "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "tokens_median": float(np.median(token_counts)) if token_counts else 0.0,
        }
    
    def best_strategy_distribution(self) -> Dict[str, float]:
        """% of questions where each strategy achieved best EM."""
        best_counts = {}
        
        for qm in self.question_metrics.values():
            best_strat, best_em = qm.best_strategy_em()
            best_counts[best_strat] = best_counts.get(best_strat, 0) + 1
        
        total = len(self.question_metrics)
        return {s: 100 * c / total for s, c in best_counts.items()}


def evaluate_question_batch(all_results: List[Dict]) -> AggregateMetrics:
    """
    Evaluate full batch of results.
    Converts raw result dicts to AggregateMetrics object.
    """
    evaluator = Evaluator()
    agg = AggregateMetrics()
    
    for result in all_results:
        q_id = result["question_id"]
        ground_truth = result["ground_truth"]
        
        qm = QuestionMetrics(q_id)
        
        for strategy_name, strat_result in result["strategies"].items():
            if "error" in strat_result:
                # Skip errors
                continue
            
            prediction = strat_result.get("answer", "")
            em = strat_result.get("em", 0.0)
            f1 = strat_result.get("f1", 0.0)
            latency = strat_result.get("latency", -1)
            token_count = strat_result.get("token_count", 0)
            
            sr = StrategyResult(
                strategy_name=strategy_name,
                question_id=q_id,
                prediction=prediction,
                em=em,
                f1=f1,
                latency=latency,
                token_count=token_count,
                metadata=strat_result.get("metadata", {})
            )
            qm.add_result(sr)
        
        agg.add_question_metrics(qm)
    
    return agg
