"""
Post-pilot analysis: Visualize and inspect Phase 1 results.
Creates summary tables, plots, and detailed error analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from tabulate import tabulate


class ResultsAnalyzer:
    """Analyze pilot results in detail."""
    
    def __init__(self, results_file: str):
        """Load results from JSON."""
        with open(results_file) as f:
            self.results = json.load(f)
        
        print(f"✓ Loaded {len(self.results)} results")
    
    def print_summary_table(self):
        """Print summary stats for each strategy."""
        strategies = ["simple", "long_context", "agentic", "multimodal"]
        
        table_data = []
        
        for strat in strategies:
            ems = []
            f1s = []
            latencies = []
            tokens = []
            errors = 0
            
            for result in self.results:
                if strat not in result["strategies"]:
                    continue
                
                s = result["strategies"][strat]
                
                if "error" in s:
                    errors += 1
                    continue
                
                ems.append(s.get("em", 0))
                f1s.append(s.get("f1", 0))
                lats = s.get("latency", -1)
                if lats > 0:
                    latencies.append(lats)
                tokens.append(s.get("token_count", 0))
            
            if not ems:
                continue
            
            table_data.append([
                strat.replace("_", " "),
                f"{np.mean(ems):.3f} ± {np.std(ems):.3f}",
                f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
                f"{np.median(latencies) if latencies else '-':.2f}s",
                f"{int(np.median(tokens))}",
                errors
            ])
        
        print("\n" + "="*70)
        print("SUMMARY: Strategy Performance Across 100 Questions")
        print("="*70)
        
        headers = ["Strategy", "EM (mean ± std)", "F1 (mean ± std)", "Latency (median)", "Tokens (median)", "Errors"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def print_best_strategy_distribution(self):
        """Print % of questions where each strategy is best."""
        strategies = ["simple", "long_context", "agentic", "multimodal"]
        best_counts = {s: 0 for s in strategies}
        
        for result in self.results:
            best_strat = None
            best_em = -1
            
            for strat in strategies:
                if strat not in result["strategies"]:
                    continue
                
                s = result["strategies"][strat]
                if "error" in s:
                    continue
                
                em = s.get("em", 0)
                if em > best_em:
                    best_em = em
                    best_strat = strat
            
            if best_strat:
                best_counts[best_strat] += 1
        
        total = len(self.results)
        
        print("\n" + "="*70)
        print("DISTRIBUTION: % of Questions Where Each Strategy is Best")
        print("="*70)
        
        for strat in strategies:
            pct = 100 * best_counts[strat] / total
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {strat:20s} {bar} {pct:5.1f}%")
    
    def print_pareto_analysis(self):
        """Identify strategies on Pareto frontier."""
        strategies = ["simple", "long_context", "agentic", "multimodal"]
        
        # Aggregate stats per strategy
        stats = {s: {"ems": [], "latencies": []} for s in strategies}
        
        for result in self.results:
            for strat in strategies:
                if strat not in result["strategies"]:
                    continue
                
                s = result["strategies"][strat]
                if "error" in s:
                    continue
                
                stats[strat]["ems"].append(s.get("em", 0))
                lat = s.get("latency", -1)
                if lat > 0:
                    stats[strat]["latencies"].append(lat)
        
        # Compute aggregates
        agg = {}
        for strat in strategies:
            if stats[strat]["ems"]:
                agg[strat] = {
                    "em_mean": np.mean(stats[strat]["ems"]),
                    "latency_median": np.median(stats[strat]["latencies"]) if stats[strat]["latencies"] else -1
                }
        
        # Find frontier
        frontier = {}
        for s_a in agg.keys():
            is_dominated = False
            for s_b in agg.keys():
                if s_a == s_b:
                    continue
                
                # s_b dominates s_a if: accuracy_b >= accuracy_a AND latency_b <= latency_a
                if (agg[s_b]["em_mean"] >= agg[s_a]["em_mean"] and
                    agg[s_b]["latency_median"] <= agg[s_a]["latency_median"]):
                    
                    if (agg[s_b]["em_mean"] > agg[s_a]["em_mean"] or
                        agg[s_b]["latency_median"] < agg[s_a]["latency_median"]):
                        is_dominated = True
            
            frontier[s_a] = not is_dominated
        
        print("\n" + "="*70)
        print("PARETO FRONTIER: Accuracy vs Latency Trade-off")
        print("="*70)
        
        for strat in strategies:
            if strat in agg:
                status = "✓ ON FRONTIER" if frontier[strat] else "✗ DOMINATED"
                print(f"  {strat:20s} {status} | EM={agg[strat]['em_mean']:.3f}, "
                      f"Lat={agg[strat]['latency_median']:.2f}s")
    
    def print_error_analysis(self):
        """Find and categorize errors."""
        print("\n" + "="*70)
        print("ERROR ANALYSIS")
        print("="*70)
        
        errors_by_strategy = {}
        
        for result in self.results:
            for strat, s in result["strategies"].items():
                if "error" in s:
                    if strat not in errors_by_strategy:
                        errors_by_strategy[strat] = []
                    
                    errors_by_strategy[strat].append({
                        "question_id": result["question_id"],
                        "query": result["query"][:60],
                        "error": s["error"][:100]
                    })
        
        if not errors_by_strategy:
            print("  ✓ No errors detected")
            return
        
        for strat in sorted(errors_by_strategy.keys()):
            errors = errors_by_strategy[strat]
            print(f"\n  {strat.upper()} ({len(errors)} errors):")
            
            for i, err in enumerate(errors[:3]):  # Show first 3
                print(f"    [{i+1}] Q{err['question_id']}: {err['error']}")
            
            if len(errors) > 3:
                print(f"    ... and {len(errors) - 3} more")
    
    def print_question_winners(self, top_k: int = 10):
        """Show which strategy won for each question (sample)."""
        print(f"\n" + "="*70)
        print(f"QUESTION-BY-QUESTION ANALYSIS (sample of {top_k})")
        print("="*70)
        
        for i, result in enumerate(self.results[:top_k]):
            strats = result["strategies"]
            
            # Find best
            best_strat = None
            best_em = -1
            for strat, s in strats.items():
                if "error" not in s and s.get("em", 0) > best_em:
                    best_em = s.get("em", 0)
                    best_strat = strat
            
            print(f"\n  Q{i+1} [{result['question_id']}]: {result['query'][:50]}...")
            print(f"    Best strategy: {best_strat} (EM={best_em:.2f})")
            
            for strat in ["simple", "long_context", "agentic", "multimodal"]:
                if strat in strats:
                    s = strats[strat]
                    if "error" in s:
                        print(f"      {strat:15s}: ERROR")
                    else:
                        em = s.get("em", 0)
                        f1 = s.get("f1", 0)
                        lat = s.get("latency", 0)
                        print(f"      {strat:15s}: EM={em:.2f}, F1={f1:.2f}, Lat={lat:.2f}s")
    
    def export_csv(self, output_path: str = "results.csv"):
        """Export as CSV for Excel."""
        import csv
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            header = ["Q_ID", "Query", "Simple_EM", "Simple_F1", "Simple_Lat",
                     "Long_EM", "Long_F1", "Long_Lat",
                     "Agentic_EM", "Agentic_F1", "Agentic_Lat",
                     "Multimodal_EM", "Multimodal_F1", "Multimodal_Lat", "Best"]
            writer.writerow(header)
            
            # Rows
            for result in self.results:
                strats = result["strategies"]
                
                best_strat = None
                best_em = -1
                for strat, s in strats.items():
                    if "error" not in s and s.get("em", 0) > best_em:
                        best_em = s.get("em", 0)
                        best_strat = strat
                
                row = [result["question_id"], result["query"][:50]]
                
                for strat in ["simple", "long_context", "agentic", "multimodal"]:
                    if strat in strats and "error" not in strats[strat]:
                        s = strats[strat]
                        row.extend([s.get("em", 0), s.get("f1", 0), s.get("latency", 0)])
                    else:
                        row.extend(["-", "-", "-"])
                
                row.append(best_strat or "NONE")
                writer.writerow(row)
        
        print(f"✓ Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 1 pilot results")
    parser.add_argument("--results", default="./phase1_results/pilot_results_full.json",
                       help="Path to results JSON")
    parser.add_argument("--export-csv", default=None, help="Export as CSV")
    
    args = parser.parse_args()
    
    if not Path(args.results).exists():
        print(f"✗ Results file not found: {args.results}")
        return
    
    analyzer = ResultsAnalyzer(args.results)
    
    analyzer.print_summary_table()
    analyzer.print_best_strategy_distribution()
    analyzer.print_pareto_analysis()
    analyzer.print_error_analysis()
    analyzer.print_question_winners(top_k=10)
    
    if args.export_csv:
        analyzer.export_csv(args.export_csv)


if __name__ == "__main__":
    main()
