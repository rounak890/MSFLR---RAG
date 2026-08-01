"""
Analysis utilities for Phase 1 pilot results.
- Distribution Analysis: % of questions where each strategy is best
- Pareto Frontier: Which strategies are on accuracy/latency trade-off curve
- Decision Logic: Determines if routing is valuable (proceed to Phase 2)
"""

from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class DistributionReport:
    """Report from distribution analysis."""
    best_strategy_pct: Dict[str, float]  # Strategy -> % of questions where it's best
    best_strategy_accuracy_pct: Dict[str, float]  # % of questions where strategy achieved >0.5 EM
    is_balanced: bool  # Are strategies balanced (20-35% each)?
    dominant_strategy: str  # Which strategy dominates (if any)
    dominant_pct: float  # Its %


@dataclass
class ParetoReport:
    """Report from Pareto frontier analysis."""
    on_frontier: Dict[str, bool]  # Strategy -> is it on Pareto frontier?
    frontier_strategies: List[str]  # Names of strategies on frontier
    dominated_by: Dict[str, List[str]]  # Strategy -> list of strategies that dominate it


@dataclass
class DecisionReport:
    """Final decision: proceed to Phase 2?"""
    proceed: bool
    reasoning: str
    confidence: str  # "high", "medium", "low"


class DistributionAnalyzer:
    """Analyzes distribution of best strategies."""
    
    def __init__(self, matrix: Dict):
        """
        matrix: Dict[q_id -> Dict[strategy -> {em, f1, latency, tokens}]]
        """
        self.matrix = matrix
    
    def analyze(self) -> DistributionReport:
        """Compute distribution of best strategies."""
        strategies = set()
        best_counts = {}
        accuracy_counts = {}  # num questions where strategy achieved EM > 0.5
        
        # Find best strategy for each question
        for q_id, strat_metrics in self.matrix.items():
            strategies.update(strat_metrics.keys())
            
            # Best by EM, with F1 as tiebreaker (when all EM=0, use F1)
            best_strat = max(
                strat_metrics.keys(),
                key=lambda s: (strat_metrics[s]["em"], strat_metrics[s]["f1"])
            )
            best_counts[best_strat] = best_counts.get(best_strat, 0) + 1
            
            # Accuracy counts
            for strat in strat_metrics.keys():
                if strat_metrics[strat]["em"] > 0.5:
                    accuracy_counts[strat] = accuracy_counts.get(strat, 0) + 1
        
        # Normalize to percentages
        total = len(self.matrix)
        best_pct = {s: 100 * best_counts.get(s, 0) / total for s in strategies}
        accuracy_pct = {s: 100 * accuracy_counts.get(s, 0) / total for s in strategies}
        
        # Check if balanced
        percentages = list(best_pct.values())
        is_balanced = all(20 <= p <= 35 for p in percentages) or \
                     (len(percentages) > 0 and max(percentages) < 75)
        
        # Find dominant strategy
        dominant_strat = max(best_pct.keys(), key=best_pct.get)
        dominant_pct = best_pct[dominant_strat]
        
        return DistributionReport(
            best_strategy_pct=best_pct,
            best_strategy_accuracy_pct=accuracy_pct,
            is_balanced=is_balanced,
            dominant_strategy=dominant_strat,
            dominant_pct=dominant_pct
        )


class ParetoAnalyzer:
    """Analyzes Pareto frontier (accuracy vs latency)."""
    
    def __init__(self, matrix: Dict):
        self.matrix = matrix
    
    def analyze(self) -> ParetoReport:
        """Identify strategies on Pareto frontier."""
        # Aggregate metrics per strategy
        strategy_stats = self._aggregate_stats()
        
        # Find Pareto frontier
        # A strategy is on frontier if no other strategy is better on both accuracy AND latency
        on_frontier = {}
        dominated_by = {}
        
        strategies = list(strategy_stats.keys())
        
        for strat_a in strategies:
            stat_a = strategy_stats[strat_a]
            acc_a = stat_a["em_mean"]
            lat_a = stat_a["latency_median"]
            
            dominators = []
            is_dominated = False
            
            for strat_b in strategies:
                if strat_a == strat_b:
                    continue
                
                stat_b = strategy_stats[strat_b]
                acc_b = stat_b["em_mean"]
                lat_b = stat_b["latency_median"]
                
                # strat_b dominates strat_a if:
                # - acc_b >= acc_a AND lat_b <= lat_a
                # - AND at least one is strictly better
                if acc_b >= acc_a and lat_b <= lat_a:
                    if acc_b > acc_a or lat_b < lat_a:
                        dominators.append(strat_b)
                        is_dominated = True
            
            on_frontier[strat_a] = not is_dominated
            dominated_by[strat_a] = dominators
        
        frontier_strategies = [s for s in strategies if on_frontier[s]]
        
        return ParetoReport(
            on_frontier=on_frontier,
            frontier_strategies=frontier_strategies,
            dominated_by=dominated_by
        )
    
    def _aggregate_stats(self) -> Dict:
        """Aggregate metrics per strategy."""
        stats = {}
        
        for q_id, strat_metrics in self.matrix.items():
            for strat, metrics in strat_metrics.items():
                if strat not in stats:
                    stats[strat] = {
                        "ems": [],
                        "f1s": [],
                        "latencies": [],
                        "tokens": []
                    }
                
                stats[strat]["ems"].append(metrics["em"])
                stats[strat]["f1s"].append(metrics["f1"])
                stats[strat]["latencies"].append(metrics["latency"])
                stats[strat]["tokens"].append(metrics["tokens"])
        
        # Compute aggregates
        result = {}
        for strat, buckets in stats.items():
            latencies = [l for l in buckets["latencies"] if l > 0]  # Filter errors
            
            result[strat] = {
                "em_mean": np.mean(buckets["ems"]),
                "em_std": np.std(buckets["ems"]),
                "f1_mean": np.mean(buckets["f1s"]),
                "f1_std": np.std(buckets["f1s"]),
                "latency_median": np.median(latencies) if latencies else -1,
                "latency_p95": np.percentile(latencies, 95) if latencies else -1,
                "tokens_median": np.median(buckets["tokens"]),
            }
        
        return result


class DecisionAnalyzer:
    """Makes final decision: proceed to Phase 2?"""
    
    def __init__(self, dist_report: DistributionReport, pareto_report: ParetoReport):
        self.dist = dist_report
        self.pareto = pareto_report
    
    def decide(self) -> DecisionReport:
        """
        Decision rule:
        1. If one strategy > 75% and on Pareto frontier: routing may not be valuable → NO
        2. If distribution is balanced (20-35%) and multiple on frontier: YES (high confidence)
        3. If mixed but >1 on frontier: YES (medium confidence)
        4. Otherwise: NO
        """
        
        dominant_pct = self.dist.dominant_pct
        num_frontier = len(self.pareto.frontier_strategies)
        is_balanced = self.dist.is_balanced
        
        # Check 1: One dominant strategy
        if dominant_pct > 75:
            if self.pareto.on_frontier.get(self.dist.dominant_strategy, False):
                return DecisionReport(
                    proceed=False,
                    reasoning=f"Single dominant strategy: {self.dist.dominant_strategy} ({dominant_pct:.1f}% best). "
                              f"Routing is unlikely to provide value. Consider different dataset or problem formulation.",
                    confidence="high"
                )
        
        # Check 2: Balanced distribution + multiple on frontier
        if is_balanced and num_frontier >= 2:
            return DecisionReport(
                proceed=True,
                reasoning=f"Balanced distribution of best strategies: {self.dist.best_strategy_pct}. "
                         f"Multiple strategies on Pareto frontier: {self.pareto.frontier_strategies}. "
                         f"Routing can provide meaningful value.",
                confidence="high"
            )
        
        # Check 3: At least 2 strategies are never dominated
        if num_frontier >= 2:
            return DecisionReport(
                proceed=True,
                reasoning=f"Multiple strategies on Pareto frontier: {self.pareto.frontier_strategies}. "
                         f"Routing can balance accuracy vs latency/cost.",
                confidence="medium"
            )
        
        # Check 4: Weak signal
        if num_frontier >= 1 and dominant_pct <= 60:
            return DecisionReport(
                proceed=True,
                reasoning=f"No clear dominant strategy (best={dominant_pct:.1f}%). "
                         f"At least one non-dominated strategy ({self.pareto.frontier_strategies}). "
                         f"Routing may provide modest value.",
                confidence="medium"
            )
        
        # Default: don't proceed
        return DecisionReport(
            proceed=False,
            reasoning=f"Weak evidence for routing. Dominant strategy: {self.dist.dominant_strategy} ({dominant_pct:.1f}%). "
                     f"Only {num_frontier} strategy(ies) on Pareto frontier. "
                     f"Consider reformulating problem or using different dataset.",
            confidence="low"
        )


class ResultsPrinter:
    """Pretty-print analysis results."""
    
    @staticmethod
    def print_distribution(report: DistributionReport):
        print("\n" + "="*70)
        print("1️⃣  DISTRIBUTION TEST: Which strategy is best?")
        print("="*70)
        
        for strat in sorted(report.best_strategy_pct.keys()):
            pct = report.best_strategy_pct[strat]
            acc_pct = report.best_strategy_accuracy_pct.get(strat, 0)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {strat:20s} {bar} {pct:5.1f}% (EM>0.5: {acc_pct:5.1f}%)")
        
        print(f"\n  Balanced? {report.is_balanced}")
        print(f"  Dominant strategy: {report.dominant_strategy} ({report.dominant_pct:.1f}%)")
    
    @staticmethod
    def print_pareto(report: ParetoReport):
        print("\n" + "="*70)
        print("2️⃣  PARETO FRONTIER: Accuracy vs Latency")
        print("="*70)
        
        for strat in sorted(report.on_frontier.keys()):
            status = "✓ ON FRONTIER" if report.on_frontier[strat] else "✗ DOMINATED"
            print(f"  {strat:20s} {status}")
            
            if report.dominated_by.get(strat):
                print(f"      Dominated by: {', '.join(report.dominated_by[strat])}")
        
        print(f"\n  Frontier strategies: {report.frontier_strategies}")
    
    @staticmethod
    def print_decision(decision: DecisionReport):
        print("\n" + "="*70)
        print("3️⃣  DECISION: Proceed to Phase 2?")
        print("="*70)
        
        proceed_emoji = "✅" if decision.proceed else "❌"
        print(f"  {proceed_emoji} {decision.proceed}")
        print(f"  Confidence: {decision.confidence.upper()}")
        print(f"\n  Reasoning:")
        for line in decision.reasoning.split(". "):
            if line.strip():
                print(f"    • {line.strip()}")
