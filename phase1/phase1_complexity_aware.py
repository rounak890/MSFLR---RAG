"""
PHASE 1 PILOT STUDY: Reasoning Complexity-Aware Sampling
Uses Reasoning Complexity Score (RCS) to sample hard questions where routing matters.

Key insight: Routing shows value on complex questions that require reasoning,
evidence aggregation, and multi-section synthesis.
"""

import json
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

# Import from HuggingFace
try:
    from datasets import load_dataset
except ImportError:
    print("⚠️ Install datasets: pip install datasets")
    exit(1)

# Import strategy implementations
from strategies import SimpleRAG, LongContextRAG, AgenticRAG
from evaluation import Evaluator
from analysis import DistributionAnalyzer, ParetoAnalyzer, DecisionAnalyzer, ResultsPrinter


@dataclass
class PilotConfig:
    """Configuration for Phase 1 pilot."""
    num_questions: int = 100
    ollama_model: str = "qwen3:0.6b"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed: int = 42
    output_dir: str = "./phase1_results_complexity_aware"
    
    # Complexity sampling distribution
    # sample counts: (simple, medium, complex, very_complex)
    sample_distribution: Tuple[int, int, int, int] = (10, 20, 40, 30)


class ReasoningComplexityScorer:
    """Compute Reasoning Complexity Score (RCS) for questions."""
    
    # Keyword patterns for different reasoning types
    COMPARISON = {
        "compare", "difference", "differ", "versus", "vs",
        "between", "better", "worse", "similar", "contrast",
        "distinguish", "differentiate"
    }
    
    CAUSAL = {
        "why", "how", "cause", "effect", "impact",
        "lead to", "result", "because", "reason", "consequence",
        "leads to", "causes", "resulting"
    }
    
    SYNTHESIS = {
        "summarize", "conclusion", "overall", "summary",
        "main findings", "main contribution", "key findings",
        "evidence", "according to the paper", "paper argues",
        "paper proposes", "authors conclude"
    }
    
    MULTI_ENTITY = {
        "both", "all", "multiple", "various", "different",
        "several", "each", "both of", "each of", "pair of"
    }
    
    LONG_CONTEXT = {
        "throughout", "across", "throughout the paper",
        "paper", "entire", "whole", "sections", "parts",
        "beginning to end", "from", "to", "range"
    }
    
    @staticmethod
    def compute_rcs(question: str, answer_list: List[str], evidence_spans: List[str]) -> float:
        """
        Compute Reasoning Complexity Score (RCS).
        
        Components:
        1. Comparison keywords: +4
        2. Causal keywords: +3
        3. Synthesis keywords: +3
        4. Multi-entity keywords: +2
        5. Long-context keywords: +2
        6. Question length: +1 per 8 words (max +3)
        7. Answer length: +0.5 per 10 words (max +2)
        8. Number of evidence spans: +1 per span (max +5)
        
        Total range: 0-25
        """
        score = 0.0
        q_lower = question.lower()
        
        # 1. Comparison (+4)
        if any(kw in q_lower for kw in ReasoningComplexityScorer.COMPARISON):
            score += 4.0
        
        # 2. Causal (+3)
        if any(kw in q_lower for kw in ReasoningComplexityScorer.CAUSAL):
            score += 3.0
        
        # 3. Synthesis (+3)
        if any(kw in q_lower for kw in ReasoningComplexityScorer.SYNTHESIS):
            score += 3.0
        
        # 4. Multi-entity (+2)
        if any(kw in q_lower for kw in ReasoningComplexityScorer.MULTI_ENTITY):
            score += 2.0
        
        # 5. Long-context (+2)
        if any(kw in q_lower for kw in ReasoningComplexityScorer.LONG_CONTEXT):
            score += 2.0
        
        # 6. Question length (longer questions = more reasoning)
        q_len = len(question.split())
        score += min(q_len // 8, 3.0)
        
        # 7. Answer length (longer answers = more reasoning)
        if answer_list:
            ans_len = sum(len(ans.split()) for ans in answer_list) / len(answer_list)
            score += min(ans_len / 10, 2.0)
        
        # 8. Number of evidence spans (best indicator of multi-hop)
        # Most important: if 2+ evidence spans, likely multi-hop
        if evidence_spans and len(evidence_spans) >= 2:
            score += min(len(evidence_spans) * 1.5, 5.0)
        
        return score
    
    @staticmethod
    def categorize_by_rcs(rcs: float) -> str:
        """Categorize question by RCS."""
        if rcs >= 12.0:
            return "very_complex"
        elif rcs >= 7.0:
            return "complex"
        elif rcs >= 3.0:
            return "medium"
        else:
            return "simple"


class QASPERComplexityAwareLoader:
    """Load QASPER with complexity-aware sampling."""
    
    @staticmethod
    def load_and_score(num_questions: int = 100) -> List[Dict]:
        """
        Load QASPER from HF, compute RCS for all questions,
        and sample from complexity buckets.
        """
        print("\n" + "="*70)
        print("LOADING QASPER WITH COMPLEXITY-AWARE SAMPLING")
        print("="*70)
        
        # Load dataset
        print("\n1. Loading QASPER from HuggingFace...")
        try:
            dataset = load_dataset("allenai/qasper", split="train")
        except Exception as e:
            print(f"⚠️ Error loading QASPER: {e}")
            exit(1)
        
        print(f"✓ Loaded {len(dataset)} papers from QASPER")
        
        # Extract and score questions
        print("\n2. Computing Reasoning Complexity Score (RCS)...")
        
        scored_questions = []
        
        # for paper in dataset:
        #     paper_id = paper["id"]
        #     title = paper["title"]
        #     abstract = paper["abstract"]
        #     full_text = paper.get("full_text", [])
            
        #     qas_list = paper.get("qas", [])
            
        #     for qa in qas_list:
        #         question = qa.get("question", "")
        #         answers = qa.get("answers", [])
                
        #         # Skip unanswerable
        #         if qa.get("unanswerable", False):
        #             continue
                
        #         # Extract answer texts and evidence
        #         answer_texts = []
        #         all_evidence = []
                
        #         if isinstance(answers, list):
        #             for ans in answers:
        #                 if isinstance(ans, dict):
        #                     answer_texts.append(ans.get("text", "").strip())
        #                     # Extract evidence spans
        #                     evidence = ans.get("evidence", [])
        #                     if isinstance(evidence, list):
        #                         all_evidence.extend(evidence)
        #                 elif isinstance(ans, str):
        #                     answer_texts.append(ans.strip())
                
        #         answer_texts = [a for a in answer_texts if a]
                
        #         if not answer_texts:
        #             continue
                
        #         # Compute RCS
        #         rcs = ReasoningComplexityScorer.compute_rcs(
        #             question=question,
        #             answer_list=answer_texts,
        #             evidence_spans=all_evidence
        #         )
                
        #         category = ReasoningComplexityScorer.categorize_by_rcs(rcs)
                
        #         scored_questions.append({
        #             "question_id": f"{paper_id}-{question[:30]}",
        #             "question": question,
        #             "answers": answer_texts,
        #             "title": title,
        #             "abstract": abstract,
        #             "full_text": full_text,
        #             "paper_id": paper_id,
        #             "rcs": rcs,
        #             "category": category,
        #             "num_evidence": len(all_evidence)
        #         })

        for paper in dataset:
            paper_id = paper["id"]
            title = paper["title"]
            abstract = paper["abstract"]
            full_text = paper.get("full_text", [])

            # HF stores QAS column-wise
            qas = paper.get("qas", {})

            questions = qas.get("question", [])
            question_ids = qas.get("question_id", [])
            answers_all = qas.get("answers", [])

            # Iterate through every question
            for question, qid, answer_group in zip(
                questions,
                question_ids,
                answers_all,
            ):

                annotations = answer_group.get("answer", [])

                answer_texts = []
                all_evidence = []

                for ann in annotations:

                    # Skip unanswerable annotations
                    if ann.get("unanswerable", False):
                        continue

                    # Free-form answer
                    free_form = ann.get("free_form_answer", "").strip()
                    if free_form:
                        answer_texts.append(free_form)

                    # Extractive spans
                    extractive_spans = ann.get("extractive_spans", [])
                    for span in extractive_spans:
                        span = span.strip()
                        if span:
                            answer_texts.append(span)

                    # Evidence spans (used by RCS)
                    evidence = ann.get("evidence", [])
                    if isinstance(evidence, list):
                        all_evidence.extend(evidence)

                # Remove duplicates
                answer_texts = list(dict.fromkeys(answer_texts))
                all_evidence = list(dict.fromkeys(all_evidence))

                # Skip if no valid answers
                if not answer_texts:
                    continue

                # Compute RCS
                rcs = ReasoningComplexityScorer.compute_rcs(
                    question=question,
                    answer_list=answer_texts,
                    evidence_spans=all_evidence
                )

                category = ReasoningComplexityScorer.categorize_by_rcs(rcs)

                scored_questions.append({
                    "question_id": qid if qid else f"{paper_id}-{question[:30]}",
                    "question": question,
                    "answers": answer_texts,
                    "title": title,
                    "abstract": abstract,
                    "full_text": full_text,
                    "paper_id": paper_id,
                    "rcs": rcs,
                    "category": category,
                    "num_evidence": len(all_evidence)
                })
        
        print(f"✓ Scored {len(scored_questions)} questions")
        
        # Analyze distribution
        print("\n3. Complexity Distribution Analysis")
        print("-" * 70)
        
        categories = {}
        for q in scored_questions:
            cat = q["category"]
            if cat not in categories:
                categories[cat] = {"count": 0, "avg_rcs": 0, "questions": []}
            categories[cat]["count"] += 1
            categories[cat]["avg_rcs"] += q["rcs"]
            categories[cat]["questions"].append(q)
        
        for cat in ["simple", "medium", "complex", "very_complex"]:
            if cat in categories:
                cnt = categories[cat]["count"]
                avg = categories[cat]["avg_rcs"] / cnt
                pct = 100 * cnt / len(scored_questions)
                print(f"  {cat:15s}: {cnt:4d} ({pct:5.1f}%) | Avg RCS: {avg:5.2f}")
        
        # Stratified sampling
        print("\n4. Stratified Sampling")
        print("-" * 70)
        
        sample_dist = (10, 20, 40, 30)  # simple, medium, complex, very_complex
        sampled = []
        
        for i, (cat, target_count) in enumerate([
            ("simple", sample_dist[0]),
            ("medium", sample_dist[1]),
            ("complex", sample_dist[2]),
            ("very_complex", sample_dist[3])
        ]):
            if cat in categories:
                available = categories[cat]["questions"]
                to_sample = min(target_count, len(available))
                sampled.extend(random.sample(available, to_sample))
                print(f"  Sampled {to_sample:2d}/{target_count} {cat:15s}")
            else:
                print(f"  Sampled  0/{target_count} {cat:15s} (not available)")
        
        print(f"\n✓ Total sampled: {len(sampled)} questions")
        
        # Show RCS distribution of sample
        sample_rcs = [q["rcs"] for q in sampled]
        print(f"\n  Sample RCS statistics:")
        print(f"    Mean:   {np.mean(sample_rcs):.2f}")
        print(f"    Median: {np.median(sample_rcs):.2f}")
        print(f"    Std:    {np.std(sample_rcs):.2f}")
        print(f"    Range:  {np.min(sample_rcs):.2f} - {np.max(sample_rcs):.2f}")
        
        return sampled


class Phase1PilotStudy:
    """Orchestrates Phase 1 with complexity-aware sampling."""
    
    def __init__(self, config: PilotConfig):
        self.config = config
        self.config.output_dir = Path(config.output_dir)
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Initialize strategies (3 only)
        self.strategies = {
            "simple": SimpleRAG(
                model=config.ollama_model,
                ollama_url=config.ollama_url,
                embedding_model=config.embedding_model
            ),
            "long_context": LongContextRAG(
                model=config.ollama_model,
                ollama_url=config.ollama_url,
                embedding_model=config.embedding_model
            ),
            "agentic": AgenticRAG(
                model=config.ollama_model,
                ollama_url=config.ollama_url,
                embedding_model=config.embedding_model
            )
        }
        
        self.evaluator = Evaluator()
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    def run_single_question(self, question_data: Dict) -> Dict:
        """Run all 3 strategies on a single question."""
        q_id = question_data["question_id"]
        query = question_data["question"]
        ground_truth = question_data["answers"]
        
        results = {
            "question_id": q_id,
            "query": query,
            "ground_truth": ground_truth,
            "rcs": question_data.get("rcs", 0),
            "category": question_data.get("category", "unknown"),
            "strategies": {}
        }
        
        for strategy_name, strategy in self.strategies.items():
            try:
                start_time = time.time()
                
                answer, metadata = strategy.run(
                    query=query,
                    question_data=question_data
                )
                
                latency = time.time() - start_time
                
                em, f1 = self.evaluator.compute_metrics(
                    prediction=answer,
                    ground_truth=ground_truth
                )
                
                token_count = metadata.get("token_count", 0)
                
                results["strategies"][strategy_name] = {
                    "answer": answer,
                    "em": em,
                    "f1": f1,
                    "latency": latency,
                    "token_count": token_count,
                    "metadata": metadata
                }
                
            except Exception as e:
                print(f"  ✗ {strategy_name} failed for Q{q_id}: {str(e)[:100]}")
                results["strategies"][strategy_name] = {
                    "error": str(e),
                    "em": 0.0,
                    "f1": 0.0,
                    "latency": -1,
                    "token_count": -1
                }
        
        return results
    
    def run_pilot(self) -> List[Dict]:
        """Run full pilot study on complexity-aware sample."""
        print("\n" + "="*70)
        print("PHASE 1: PILOT STUDY - COMPLEXITY-AWARE ROUTING VALIDATION")
        print("="*70)
        
        questions = QASPERComplexityAwareLoader.load_and_score(self.config.num_questions)
        all_results = []
        
        print(f"\nRunning {len(questions)} complexity-aware questions × 3 strategies...\n")
        
        for i, q_data in enumerate(tqdm(questions, desc="Questions"), 1):
            result = self.run_single_question(q_data)
            all_results.append(result)
            
            # Checkpoint every 20 questions
            if i % 20 == 0:
                self.save_checkpoint(all_results, i)
        
        return all_results
    
    def save_checkpoint(self, results: List[Dict], batch_num: int):
        """Save intermediate results."""
        ckpt_file = self.config.output_dir / f"checkpoint_batch{batch_num}.json"
        with open(ckpt_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  → Checkpoint saved: {ckpt_file.name}")
    
    def analyze_results(self, all_results: List[Dict]) -> Dict:
        """Analyze results by complexity."""
        print("\n" + "="*70)
        print("ANALYSIS: Performance by Reasoning Complexity")
        print("="*70)
        
        # Build results matrix
        matrix = self._build_results_matrix(all_results)
        
        # Analyze by complexity category
        print("\n📊 Performance by Question Complexity")
        print("-" * 70)
        
        categories = {}
        for result in all_results:
            cat = result.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        for cat in ["simple", "medium", "complex", "very_complex"]:
            if cat not in categories:
                continue
            
            cat_results = categories[cat]
            print(f"\n{cat.upper()} Questions (n={len(cat_results)}):")
            
            # Average F1 per strategy for this category
            for strat in self.strategies.keys():
                f1_scores = []
                for result in cat_results:
                    if strat in result["strategies"] and "f1" in result["strategies"][strat]:
                        f1_scores.append(result["strategies"][strat]["f1"])
                
                if f1_scores:
                    avg_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores)
                    print(f"  {strat:15s}: F1={avg_f1:.3f}±{std_f1:.3f}")
        
        # Distribution analysis
        dist_analyzer = DistributionAnalyzer(matrix)
        dist_report = dist_analyzer.analyze()
        
        # Pareto frontier
        pareto_analyzer = ParetoAnalyzer(matrix)
        pareto_report = pareto_analyzer.analyze()
        
        # Decision logic
        decision_analyzer = DecisionAnalyzer(dist_report, pareto_report)
        decision = decision_analyzer.decide()
        
        # Print summary
        print("\n" + "="*70)
        print("1️⃣  DISTRIBUTION TEST (Overall)")
        print("-" * 70)
        ResultsPrinter.print_distribution(dist_report)
        
        print("\n" + "="*70)
        print("2️⃣  PARETO FRONTIER")
        print("-" * 70)
        ResultsPrinter.print_pareto(pareto_report)
        
        print("\n" + "="*70)
        print("3️⃣  DECISION")
        print("-" * 70)
        ResultsPrinter.print_decision(decision)
        
        # Aggregate report
        analysis = {
            "distribution": {
                "best_strategy_pct": dist_report.best_strategy_pct,
                "is_balanced": dist_report.is_balanced,
                "dominant_strategy": dist_report.dominant_strategy,
                "dominant_pct": dist_report.dominant_pct
            },
            "pareto": {
                "on_frontier": pareto_report.on_frontier,
                "frontier_strategies": pareto_report.frontier_strategies
            },
            "decision": {
                "proceed": decision.proceed,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence
            },
            "matrix_summary": self._summarize_matrix(matrix),
            "complexity_analysis": self._analyze_by_complexity(categories)
        }
        
        return analysis
    
    def _analyze_by_complexity(self, categories: Dict) -> Dict:
        """Analyze performance stratified by complexity."""
        result = {}
        
        for cat in ["simple", "medium", "complex", "very_complex"]:
            if cat not in categories:
                continue
            
            cat_results = categories[cat]
            cat_data = {"n": len(cat_results)}
            
            for strat in self.strategies.keys():
                f1_scores = []
                for res in cat_results:
                    if strat in res["strategies"] and "f1" in res["strategies"][strat]:
                        f1_scores.append(res["strategies"][strat]["f1"])
                
                if f1_scores:
                    cat_data[f"{strat}_f1_mean"] = float(np.mean(f1_scores))
                    cat_data[f"{strat}_f1_std"] = float(np.std(f1_scores))
            
            result[cat] = cat_data
        
        return result
    
    def _build_results_matrix(self, all_results: List[Dict]) -> Dict:
        matrix = {}
        strategies = list(self.strategies.keys())
        
        for result in all_results:
            q_id = result["question_id"]
            matrix[q_id] = {}
            
            for strat in strategies:
                if strat in result["strategies"]:
                    strat_result = result["strategies"][strat]
                    matrix[q_id][strat] = {
                        "em": strat_result.get("em", 0.0),
                        "f1": strat_result.get("f1", 0.0),
                        "latency": strat_result.get("latency", -1),
                        "tokens": strat_result.get("token_count", 0)
                    }
        
        return matrix
    
    def _summarize_matrix(self, matrix: Dict) -> Dict:
        strategies = ["simple", "long_context", "agentic"]
        summary = {}
        
        for strat in strategies:
            ems = [matrix[q][strat]["em"] for q in matrix if strat in matrix[q]]
            f1s = [matrix[q][strat]["f1"] for q in matrix if strat in matrix[q]]
            lats = [matrix[q][strat]["latency"] for q in matrix if strat in matrix[q] and matrix[q][strat]["latency"] > 0]
            
            summary[strat] = {
                "em_mean": np.mean(ems) if ems else 0,
                "f1_mean": np.mean(f1s) if f1s else 0,
                "latency_median": np.median(lats) if lats else -1,
                "count": len(ems)
            }
        
        return summary
    
    def save_results(self, all_results: List[Dict], analysis: Dict):
        """Save results."""
        # Full results
        results_file = self.config.output_dir / "pilot_results_full.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved full results: {results_file}")
        
        # Analysis
        analysis_file = self.config.output_dir / "pilot_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ Saved analysis: {analysis_file}")
        
        # CSV matrix
        self._save_csv_matrix(all_results)
    
    def _save_csv_matrix(self, all_results: List[Dict]):
        """Save results as CSV."""
        import csv
        
        csv_file = self.config.output_dir / "pilot_results_matrix.csv"
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            header = ["Q_ID", "Category", "RCS", "Query"] + [
                f"{strat}_{metric}"
                for strat in self.strategies.keys()
                for metric in ["EM", "F1", "Latency", "Tokens"]
            ]
            writer.writerow(header)
            
            # Rows
            for result in all_results:
                q_id = result["question_id"]
                category = result.get("category", "")
                rcs = result.get("rcs", 0)
                query = result["query"][:40]
                
                row = [q_id, category, f"{rcs:.2f}", query]
                
                strats = result["strategies"]
                for strat in self.strategies.keys():
                    if strat in strats:
                        s = strats[strat]
                        em = s.get("em", 0)
                        f1 = s.get("f1", 0)
                        lat = s.get("latency", -1)
                        tok = s.get("token_count", 0)
                        row.extend([f"{em:.2f}", f"{f1:.3f}", f"{lat:.2f}", tok])
                    else:
                        row.extend(["ERROR", "ERROR", "-1", "-1"])
                
                writer.writerow(row)
        
        print(f"✓ Saved CSV matrix: {csv_file}")


def main():
    """Entry point."""
    config = PilotConfig(
        num_questions=100,
        ollama_model="qwen3:0.6b",
        ollama_url="http://localhost:11434",
        output_dir="./phase1_results_complexity_aware"
    )
    
    # Check Ollama
    print("\n" + "="*70)
    print("CHECKING OLLAMA CONNECTION")
    print("="*70)
    
    try:
        import requests
        response = requests.get(f"{config.ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama is running!")
            print(f"  Available models: {[m['name'].split(':')[0] for m in models]}")
        else:
            print(f"✗ Ollama returned error: {response.status_code}")
            exit(1)
    except Exception as e:
        print(f"✗ Cannot connect to Ollama at {config.ollama_url}")
        print(f"  Error: {e}")
        exit(1)
    
    # Run pilot
    pilot = Phase1PilotStudy(config)
    all_results = pilot.run_pilot()
    
    # Analyze
    analysis = pilot.analyze_results(all_results)
    
    # Save
    pilot.save_results(all_results, analysis)
    
    print("\n" + "="*70)
    print("✓ PHASE 1 COMPLETE (COMPLEXITY-AWARE SAMPLING)")
    print("="*70)
    print(f"Results saved to: {config.output_dir}")
    print(f"\nDecision: {'PROCEED TO PHASE 2 ✅' if analysis['decision']['proceed'] else 'NEED DIFFERENT APPROACH'}")


if __name__ == "__main__":
    main()
