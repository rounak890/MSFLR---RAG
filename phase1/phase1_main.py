"""
PHASE 1 PILOT STUDY: Routing Validation on QASPER (HuggingFace Version)
Uses HuggingFace load_dataset("allenai/qasper") and Ollama (qwen2.5:7b)

Runs 3 strategies on 100 QASPER questions:
1. Simple RAG (fast, cheap)
2. Long-Context RAG (accurate, expensive)
3. Agentic RAG (complex reasoning, slow)

Then analyzes: distribution test, Pareto frontier, proceed to Phase 2?
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


def stratified_sample_qasper(dataset, num_samples=100):
    """Sample questions by type: factoid, multi-hop, comparison, etc."""
    
    # Categorize by question patterns
    factoid = []         # "What is X?", "Where is Y?"
    multi_hop = []       # "Compare X and Y", "How does X lead to Y?"
    comparison = []      # Requires multiple passages
    other = []
    
    for qa in dataset:
        q = qa['question'].lower()
        
        if any(x in q for x in ["what", "where", "which", "when", "who"]):
            if any(x in q for x in ["compare", "differ", "versus", "vs", "between"]):
                comparison.append(qa)
            elif any(x in q for x in ["how", "why", "implication", "result", "effect"]):
                multi_hop.append(qa)
            else:
                factoid.append(qa)
        else:
            other.append(qa)
    
    # Sample evenly: 25 factoid, 35 multi-hop, 25 comparison, 15 other
    sampled = []
    sampled.extend(random.sample(factoid, min(25, len(factoid))))
    sampled.extend(random.sample(multi_hop, min(35, len(multi_hop))))
    sampled.extend(random.sample(comparison, min(25, len(comparison))))
    sampled.extend(random.sample(other, min(15, len(other))))
    
    return sampled[:100]

@dataclass
class PilotConfig:
    """Configuration for Phase 1 pilot."""
    num_questions: int = 100
    ollama_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed: int = 42
    output_dir: str = "./phase1_results"


# class HFQASPERProcessor:
#     """Convert HuggingFace QASPER to pilot format."""
    
#     @staticmethod
#     def load_qasper_hf(num_samples: int = 100) -> List[Dict]:
#         """
#         Load QASPER from HuggingFace and extract (question, answers) pairs.
#         HF format:
#         - id: paper_id
#         - title: paper title
#         - abstract: abstract
#         - full_text: list of text sections
#         - qas: list of {'question': str, 'answers': [...], 'unanswerable': bool}
#         - figures_and_tables: metadata
#         """
#         print("Loading QASPER from HuggingFace...")
        
#         # Load dataset (will auto-cache)
#         try:
#             dataset = load_dataset("allenai/qasper", split="train")
#         except Exception as e:
#             print(f"⚠️ Error loading QASPER: {e}")
#             print("Try: pip install datasets huggingface_hub")
#             exit(1)
        
#         print(f"✓ Loaded {len(dataset)} papers from QASPER")
        
#         # Extract (paper, question, answers) triples
#         question_data = []
        
#         for paper in dataset:
#             paper_id = paper["id"]
#             title = paper["title"]
#             abstract = paper["abstract"]
#             full_text = paper.get("full_text", [])
            
#             # Each paper has multiple questions
#             qas_list = paper.get("qas", [])
            
#             for qa in qas_list:
#                 question = qa.get("question", "")
#                 answers = qa.get("answers", [])
                
#                 # Skip unanswerable questions
#                 if qa.get("unanswerable", False):
#                     continue
                
#                 # Skip if no answers
#                 if not answers:
#                     continue
                
#                 # Normalize answers (HF format: list of {answer_start, text})
#                 answer_texts = []
#                 if isinstance(answers, list):
#                     for ans in answers:
#                         if isinstance(ans, dict):
#                             answer_texts.append(ans.get("text", "").strip())
#                         elif isinstance(ans, str):
#                             answer_texts.append(ans.strip())
                
#                 answer_texts = [a for a in answer_texts if a]  # Remove empties
                
#                 if not answer_texts:
#                     continue
                
#                 # Create question record
#                 question_data.append({
#                     "question_id": f"{paper_id}-{question[:30]}",
#                     "question": question,
#                     "answers": answer_texts,
#                     "title": title,
#                     "abstract": abstract,
#                     "full_text": full_text,
#                     "paper_id": paper_id
#                 })
        
#         print(f"✓ Extracted {len(question_data)} (paper, question, answers) triples")
        
#         # Sample
#         random.seed(42)
#         sampled = random.sample(question_data, min(num_samples, len(question_data)))
#         print(f"✓ Sampled {len(sampled)} questions")
        
#         return sampled

class HFQASPERProcessor:
    """Convert HuggingFace QASPER to pilot format."""

    @staticmethod
    def load_qasper_hf(num_samples: int = 100) -> List[Dict]:
        """
        Load QASPER from HuggingFace and extract (question, answers) pairs.

        HF format:
        - id: paper_id
        - title: paper title
        - abstract: abstract
        - full_text: paper sections
        - qas:
            {
                "question": [...],
                "question_id": [...],
                "answers": [...],
                ...
            }

        Each question contains multiple human annotations.
        """
        print("Loading QASPER from HuggingFace...")

        # Load dataset (auto cached)
        try:
            dataset = load_dataset("allenai/qasper", split="train")
        except Exception as e:
            print(f"⚠️ Error loading QASPER: {e}")
            print("Try: pip install datasets huggingface_hub")
            exit(1)

        print(f"✓ Loaded {len(dataset)} papers from QASPER")

        question_data = []

        for paper in dataset:

            paper_id = paper["id"]
            title = paper["title"]
            abstract = paper["abstract"]
            full_text = paper.get("full_text", [])

            # QAS is stored column-wise instead of list-of-dicts
            qas = paper.get("qas", {})

            questions = qas.get("question", [])
            question_ids = qas.get("question_id", [])
            answers_all = qas.get("answers", [])

            # Iterate over every question in this paper
            for question, qid, answer_group in zip(
                questions,
                question_ids,
                answers_all,
            ):

                annotations = answer_group.get("answer", [])

                answer_texts = []

                for ann in annotations:

                    # Skip unanswerable annotations
                    if ann.get("unanswerable", False):
                        continue

                    # Prefer free-form answer
                    free_form = ann.get("free_form_answer", "").strip()
                    if free_form:
                        answer_texts.append(free_form)

                    # Otherwise use extractive spans
                    extractive_spans = ann.get("extractive_spans", [])
                    for span in extractive_spans:
                        span = span.strip()
                        if span:
                            answer_texts.append(span)

                # Remove duplicates while preserving order
                answer_texts = list(dict.fromkeys(answer_texts))

                # Skip questions with no valid answers
                if not answer_texts:
                    continue

                # Create question record
                question_data.append({
                    "question_id": f"{paper_id}-{question[:30]}",
                    "question": question,
                    "answers": answer_texts,
                    "title": title,
                    "abstract": abstract,
                    "full_text": full_text,
                    "paper_id": paper_id
                })

        print(f"✓ Extracted {len(question_data)} (paper, question, answers) triples")

        # Sample
        random.seed(42)
        sampled = random.sample(
            question_data,
            min(num_samples, len(question_data))
        )

        print(f"✓ Sampled {len(sampled)} questions")

        return sampled

class Phase1PilotStudy:
    """Orchestrates Phase 1: load data, run strategies, evaluate, analyze."""
    
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
        """
        Run all 3 strategies on a single question.
        Returns dict with strategy results and metadata.
        """
        q_id = question_data["question_id"]
        query = question_data["question"]
        ground_truth = question_data["answers"]
        
        results = {
            "question_id": q_id,
            "query": query,
            "ground_truth": ground_truth,
            "strategies": {}
        }
        
        for strategy_name, strategy in self.strategies.items():
            try:
                start_time = time.time()
                
                # Run strategy
                answer, metadata = strategy.run(
                    query=query,
                    question_data=question_data
                )
                
                latency = time.time() - start_time
                
                # Evaluate
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
        """Run full pilot study on 100 questions."""
        print("\n" + "="*70)
        print("PHASE 1: PILOT STUDY - ROUTING VALIDATION (3 Strategies)")
        print("="*70)
        
        # questions = HFQASPERProcessor.load_qasper_hf(self.config.num_questions)


        # NEW (stratified by complexity)
        questions = stratified_sample_qasper(
            HFQASPERProcessor.load_qasper_hf(500),  # Load 500 first
            num_samples=100
        )

        all_results = []
        
        print(f"\nRunning {len(questions)} questions × 3 strategies...\n")
        
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
        """Analyze and visualize pilot results."""
        print("\n" + "="*70)
        print("ANALYSIS: Distribution & Pareto Frontier")
        print("="*70)
        
        # Build results matrix: (question, strategy) -> (em, f1, latency, tokens)
        matrix = self._build_results_matrix(all_results)
        
        # Distribution analysis
        dist_analyzer = DistributionAnalyzer(matrix)
        dist_report = dist_analyzer.analyze()
        
        # Pareto frontier
        pareto_analyzer = ParetoAnalyzer(matrix)
        pareto_report = pareto_analyzer.analyze()
        
        # Decision logic
        decision_analyzer = DecisionAnalyzer(dist_report, pareto_report)
        decision = decision_analyzer.decide()
        
        # Print results
        ResultsPrinter.print_distribution(dist_report)
        ResultsPrinter.print_pareto(pareto_report)
        ResultsPrinter.print_decision(decision)
        
        # Aggregate report
        analysis = {
            "distribution": {
                "best_strategy_pct": dist_report.best_strategy_pct,
                "best_strategy_accuracy_pct": dist_report.best_strategy_accuracy_pct,
                "is_balanced": dist_report.is_balanced,
                "dominant_strategy": dist_report.dominant_strategy,
                "dominant_pct": dist_report.dominant_pct
            },
            "pareto": {
                "on_frontier": pareto_report.on_frontier,
                "frontier_strategies": pareto_report.frontier_strategies,
                "dominated_by": pareto_report.dominated_by
            },
            "decision": {
                "proceed": decision.proceed,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence
            },
            "matrix_summary": self._summarize_matrix(matrix)
        }
        
        return analysis
    
    def _build_results_matrix(self, all_results: List[Dict]) -> Dict[str, Dict]:
        """Convert flat results to (question, strategy) -> metrics dict."""
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
        """Summarize metrics: mean, std, median per strategy."""
        strategies = ["simple", "long_context", "agentic"]
        summary = {}
        
        for strat in strategies:
            ems = [matrix[q][strat]["em"] for q in matrix if strat in matrix[q]]
            f1s = [matrix[q][strat]["f1"] for q in matrix if strat in matrix[q]]
            lats = [matrix[q][strat]["latency"] for q in matrix if strat in matrix[q] and matrix[q][strat]["latency"] > 0]
            
            summary[strat] = {
                "em_mean": np.mean(ems) if ems else 0,
                "em_std": np.std(ems) if ems else 0,
                "f1_mean": np.mean(f1s) if f1s else 0,
                "f1_std": np.std(f1s) if f1s else 0,
                "latency_median": np.median(lats) if lats else -1,
                "count": len(ems)
            }
        
        return summary
    
    def save_results(self, all_results: List[Dict], analysis: Dict):
        """Save all results to JSON and CSV for inspection."""
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
        
        # CSV matrix for easy inspection in Excel
        self._save_csv_matrix(all_results)
    
    def _save_csv_matrix(self, all_results: List[Dict]):
        """Save results as CSV for spreadsheet inspection."""
        import csv
        
        csv_file = self.config.output_dir / "pilot_results_matrix.csv"
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            header = ["Q_ID", "Query"] + [
                f"{strat}_{metric}"
                for strat in self.strategies.keys()
                for metric in ["EM", "F1", "Latency", "Tokens", "Best"]
            ]
            writer.writerow(header)
            
            # Rows
            for result in all_results:
                q_id = result["question_id"]
                query = result["query"][:50]
                
                row = [q_id, query]
                
                strats = result["strategies"]
                best_strat = max(
                    strats.keys(),
                    key=lambda s: strats[s].get("em", 0)
                )
                
                for strat in self.strategies.keys():
                    if strat in strats:
                        s = strats[strat]
                        em = s.get("em", 0)
                        f1 = s.get("f1", 0)
                        lat = s.get("latency", -1)
                        tok = s.get("token_count", 0)
                        is_best = "✓" if strat == best_strat else ""
                        
                        row.extend([f"{em:.2f}", f"{f1:.2f}", f"{lat:.2f}", tok, is_best])
                    else:
                        row.extend(["ERROR", "ERROR", "-1", "-1", ""])
                
                writer.writerow(row)
        
        print(f"✓ Saved CSV matrix: {csv_file}")


def main():
    """Entry point: Configure, run pilot, analyze."""
    config = PilotConfig(
        num_questions=100,
        ollama_model="qwen2.5:7b",
        ollama_url="http://localhost:11434",
        output_dir="./phase1_results"
    )
    
    # Check Ollama connection
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
        print(f"  Make sure Ollama is running: ollama serve")
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
    print("✓ PHASE 1 COMPLETE")
    print("="*70)
    print(f"Results saved to: {config.output_dir}")
    print(f"\nDecision: {'PROCEED TO PHASE 2' if analysis['decision']['proceed'] else 'STOP - ROUTING NOT VALUABLE'}")


if __name__ == "__main__":
    main()