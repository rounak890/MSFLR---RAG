"""
PHASE 3 PRODUCTION: LEARNED ROUTING vs BASELINE ROUTING
========================================================

QASPER VALIDATION split evaluation:
- Compare learned routing vs Simple RAG baseline
- Real EM/F1 metrics (no mocks)
- Draw detection: when F1 scores are within tolerance
- Pareto-ready results

Production-ready, crystal clear outputs.
"""

import json
import time
import numpy as np
import pickle
from pathlib import Path
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except:
    DATASETS_AVAILABLE = False
    print("⚠️  Install: pip install datasets")

from phase2a_final_features import Phase2ACompleteFeatureExtractor

# Real strategy implementations (no mocks)
from strategies import SimpleRAG, LongContextRAG, AgenticRAG

# Real evaluation (no mocks)
from evals_production_ready import exact_match, f1_score, compare_results


@dataclass
class Phase3ProdConfig:
    """Production Phase 3 configuration."""
    phase2_models_dir: str = "./phase2_models_and_results/models"
    phase2_results_dir: str = "./phase2_models_and_results"
    output_dir: str = "./phase3_results_validation_draw_full_set"
    
    test_set_size: int = 241
    random_seed: int = 42
    
    retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:0.6b"
    # ollama_model: str = "qwen2.5:7b"
    
    best_iteration: int = 5
    
    # Draw tolerance: within 1% F1 = draw
    draw_tolerance: float = 0.01



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

class QASPERValidationLoader:
    """Load QASPER validation split (real new data)."""
    
    def __init__(self, config: Phase3ProdConfig):
        self.config = config

    def load_validation_set(self, num_questions: int = 100) -> List[Dict]:
        """Load QASPER validation set."""
        print("\n📥 LOADING QASPER VALIDATION SET")
        print("-" * 80)

        if not DATASETS_AVAILABLE:
            print("⚠️  HuggingFace datasets not available")
            return []

        try:
            dataset = load_dataset("allenai/qasper")
            dataset = dataset["validation"]

            print(f"  QASPER validation split: {len(dataset)} papers")

            questions = []
            scored_questions= []
            np.random.seed(self.config.random_seed)

            for paper in dataset:
                paper_id = paper["id"]
                title = paper["title"]
                abstract = paper["abstract"]
                full_text = paper.get("full_text", [])

                # NEEDED FOR THE WHOLE IN WHOLE CODE @CFBR
                paragraphs = full_text.get("paragraphs", [])
                section_names = full_text.get("section_name", [])

                all_passages = []

                if section_names and len(section_names) == len(paragraphs):
                    for sec_name, sec_paras in zip(section_names, paragraphs):
                        for para in sec_paras:
                            if para.strip():
                                all_passages.append(f"[{sec_name}] {para}")
                else:
                    for sec_paras in paragraphs:
                        for para in sec_paras:
                            if para.strip():
                                all_passages.append(para)

                if not all_passages:
                    continue

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
                        "id": f"{paper_id}_q{qid}",
                        "paper_id": paper_id,

                        "title": title,
                        "query": question,
                        "ground_truth": answer_texts,
                        "passages": all_passages,
                        "source": "qasper_validation",

                        "question_id": qid if qid else f"{paper_id}-{question[:30]}",
                        "question": question,
                        "answers": answer_texts,
                        # "title": title,
                        "abstract": abstract,
                        "full_text": full_text,
                        # "paper_id": paper_id,
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
            
            sample_dist = (10, 40, 20, 30)  # simple, medium, complex, very_complex
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

            # for paper_idx, paper in enumerate(tqdm(val_split, desc="  Loading papers")):
            #     if len(questions) >= num_questions:
            #         break

            #     paper_id = paper.get("id", f"paper_{paper_idx}")
            #     title = paper.get("title", "")

            #     full_text = paper.get("full_text", {})
            #     paragraphs = full_text.get("paragraphs", [])
            #     section_names = full_text.get("section_name", [])

            #     all_passages = []

            #     if section_names and len(section_names) == len(paragraphs):
            #         for sec_name, sec_paras in zip(section_names, paragraphs):
            #             for para in sec_paras:
            #                 if para.strip():
            #                     all_passages.append(f"[{sec_name}] {para}")
            #     else:
            #         for sec_paras in paragraphs:
            #             for para in sec_paras:
            #                 if para.strip():
            #                     all_passages.append(para)

            #     if not all_passages:
            #         continue

            #     qas = paper.get("qas", {})
            #     question_texts = qas.get("question", [])
            #     answers_list = qas.get("answers", [])

            #     for q_idx, (q_text, answer_record) in enumerate(zip(question_texts, answers_list)):
            #         if len(questions) >= num_questions:
            #             break

            #         if not q_text:
            #             continue

            #         # Extract all acceptable answers
            #         answer_texts = []

            #         if isinstance(answer_record, dict):
            #             annotations = answer_record.get("answer", [])

            #             for ann in annotations:
            #                 # Skip unanswerable
            #                 if ann.get("unanswerable", False):
            #                     continue

            #                 # Free-form answer
            #                 free_form = ann.get("free_form_answer", "").strip()
            #                 if free_form:
            #                     answer_texts.append(free_form)

            #                 # Extractive spans
            #                 extractive_spans = ann.get("extractive_spans", [])
            #                 for span in extractive_spans:
            #                     span = span.strip()
            #                     if span:
            #                         answer_texts.append(span)

            #         # Remove duplicates
            #         answer_texts = list(dict.fromkeys(answer_texts))

            #         if not answer_texts:
            #             continue

            #         questions.append({
            #             "id": f"{paper_id}_q{q_idx}",
            #             "paper_id": paper_id,
            #             "title": title,
            #             "query": q_text,
            #             "ground_truth": answer_texts,
            #             "passages": all_passages,
            #             "source": "qasper_validation"
            #         })

            # print(f"✓ Loaded {len(questions)} questions from QASPER validation")
            # return questions

        except Exception as e:
            print(f"⚠️  Error: {str(e)[:80]}")
            return []


class Phase3StrategyRunner:
    """Run RAG strategies and get REAL metrics."""
    
    def __init__(self, config: Phase3ProdConfig):
        self.config = config
        self.feature_extractor = Phase2ACompleteFeatureExtractor(
            ollama_url=config.ollama_url,
            ollama_model=config.ollama_model
        )
        self.strategies = {
            "simple": SimpleRAG(
                model=config.ollama_model,
                ollama_url=config.ollama_url,
                embedding_model=config.retriever_model
            ),
            "long_context": LongContextRAG(
                model=config.ollama_model,
                ollama_url=config.ollama_url,
                embedding_model=config.retriever_model
            ),
            "agentic": AgenticRAG(
                model=config.ollama_model,
                ollama_url=config.ollama_url,
                embedding_model=config.retriever_model
            )
        }
    
    def run_strategy(self, strategy: str, question: Dict) -> Dict:
        """Run strategy and compute REAL EM/F1."""
        query = question["query"]
        passages = question["passages"]
        ground_truth = question["ground_truth"]
        
        question_data = {
            "full_text": passages,
            "abstract": passages[0] if passages else ""
        }
        
        # Run strategy (real implementation from strategies.py)
        strat_obj = self.strategies[strategy]
        answer, metadata = strat_obj.run(query=query, question_data=question_data)
        
        # Compute REAL EM/F1 metrics
        em = 1.0 if exact_match(answer, ground_truth) else 0.0
        f1 = f1_score(answer, ground_truth)
        
        # Get retrieved passages and similarities
        retrieved_passages = metadata.get("retrieved_passages", [])
        if isinstance(retrieved_passages, list) and len(retrieved_passages) > 0:
            if isinstance(retrieved_passages[0], str):
                retrieved_passages = retrieved_passages
            else:
                retrieved_passages = [str(p) for p in retrieved_passages]
        
        similarities = metadata.get("similarities", [])
        similarities_padded = list(similarities) if similarities else []
        while len(similarities_padded) < 10:
            similarities_padded.append(0.3)
            
        return {
            "strategy": strategy,
            "answer": answer,
            "retrieved_passages": retrieved_passages,
            "similarities": similarities_padded[:10],
            "em_score": float(em),
            "f1_score": float(f1),
            "retrieval_time": float(metadata.get("retrieval_time", 0.0)),
            "generation_time": float(metadata.get("generation_time", 0.0)),
            "total_time": float(metadata.get("total_time", 0.0)),
            "token_count": int(metadata.get("token_count", 0)),
            "context_tokens": int(metadata.get("context_tokens", 0)),
            "metadata": metadata
        }


class Phase3Evaluator:
    """Evaluate learned routing vs baseline with draw detection."""
    
    def __init__(self, config: Phase3ProdConfig):
        self.config = config
    
    def evaluate(self, predictions: List[Dict]) -> Dict:
        """Evaluate with draw detection."""
        print("\n📊 EVALUATING LEARNED vs BASELINE (WITH DRAW DETECTION)")
        print("-" * 80)
        
        if not predictions:
            return {}
        
        learned_f1s = [p["learned_f1"] for p in predictions]
        baseline_f1s = [p["baseline_f1"] for p in predictions]
        
        learned_ems = [p["learned_em"] for p in predictions]
        baseline_ems = [p["baseline_em"] for p in predictions]
        
        learned_times = [p["learned_time"] for p in predictions]
        baseline_times = [p["baseline_time"] for p in predictions]
        
        # Compute averages
        learned_f1_avg = np.mean(learned_f1s)
        baseline_f1_avg = np.mean(baseline_f1s)
        f1_improvement = (learned_f1_avg - baseline_f1_avg) / baseline_f1_avg if baseline_f1_avg > 0 else 0
        
        learned_em_avg = np.mean(learned_ems)
        baseline_em_avg = np.mean(baseline_ems)
        em_improvement = (learned_em_avg - baseline_em_avg) / baseline_em_avg if baseline_em_avg > 0 else 0
        
        learned_time_avg = np.mean(learned_times)
        baseline_time_avg = np.mean(baseline_times)
        time_improvement = (baseline_time_avg - learned_time_avg) / baseline_time_avg if baseline_time_avg > 0 else 0
        
        # Count wins/draws/losses
        learned_wins = 0
        draws = 0
        baseline_wins = 0
        
        for l_f1, b_f1 in zip(learned_f1s, baseline_f1s):
            result = compare_results(b_f1, l_f1, tolerance=self.config.draw_tolerance)
            if result == "learned_wins":
                learned_wins += 1
            elif result == "draw":
                draws += 1
            else:
                baseline_wins += 1
        
        learned_wins_pct = 100 * learned_wins / len(predictions) if predictions else 0
        draws_pct = 100 * draws / len(predictions) if predictions else 0
        baseline_wins_pct = 100 * baseline_wins / len(predictions) if predictions else 0
        
        # Strategy distribution
        strategy_dist = defaultdict(int)
        for p in predictions:
            strategy_dist[p["learned_strategy"]] += 1
        
        print(f"\n  Questions evaluated: {len(predictions)}")
        print(f"\n  F1 SCORE:")
        print(f"    Baseline (Simple):   {baseline_f1_avg:.3f}")
        print(f"    Learned routing:     {learned_f1_avg:.3f}")
        print(f"    Improvement:         {f1_improvement:+.1%}")
        
        print(f"\n  EXACT MATCH:")
        print(f"    Baseline (Simple):   {baseline_em_avg:.1%}")
        print(f"    Learned routing:     {learned_em_avg:.1%}")
        print(f"    Improvement:         {em_improvement:+.1%}")
        
        print(f"\n  LATENCY:")
        print(f"    Baseline (Simple):   {baseline_time_avg:.2f}s")
        print(f"    Learned routing:     {learned_time_avg:.2f}s")
        print(f"    Improvement:         {time_improvement:+.1%} faster")
        
        print(f"\n  HEAD-TO-HEAD COMPARISON:")
        print(f"    Learned WINS:        {learned_wins:3d} ({learned_wins_pct:.1f}%)")
        print(f"    DRAWS:               {draws:3d} ({draws_pct:.1f}%)")
        print(f"    Baseline WINS:       {baseline_wins:3d} ({baseline_wins_pct:.1f}%)")
        
        print(f"\n  STRATEGY DISTRIBUTION (Learned):")
        for strategy, count in sorted(strategy_dist.items()):
            pct = 100 * count / len(predictions)
            print(f"    {strategy:15s}: {count:3d} ({pct:.1f}%)")
        
        return {
            "num_predictions": len(predictions),
            "f1_baseline": float(baseline_f1_avg),
            "f1_learned": float(learned_f1_avg),
            "f1_improvement": float(f1_improvement),
            "em_baseline": float(baseline_em_avg),
            "em_learned": float(learned_em_avg),
            "em_improvement": float(em_improvement),
            "latency_baseline": float(baseline_time_avg),
            "latency_learned": float(learned_time_avg),
            "latency_improvement": float(time_improvement),
            "learned_wins": int(learned_wins),
            "learned_wins_pct": float(learned_wins_pct),
            "draws": int(draws),
            "draws_pct": float(draws_pct),
            "baseline_wins": int(baseline_wins),
            "baseline_wins_pct": float(baseline_wins_pct),
            "strategy_distribution": dict(strategy_dist),
            "evaluation_type": "learned_vs_baseline_with_draws"
        }


class Phase3ProdPipeline:
    """Main Phase 3 production pipeline."""
    
    def __init__(self, config: Phase3ProdConfig):
        self.config = config
        self.config.output_dir = Path(config.output_dir)
        self.config.output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("PHASE 3 PRODUCTION: LEARNED ROUTING vs BASELINE")
        print("="*80)
        print("QASPER VALIDATION split (new data)")
        print("Draw detection enabled (within 1% F1)")
        print("Real EM/F1 metrics (no mocks)")
        print("="*80)
    
    def run(self):
        """Execute Phase 3 pipeline."""
        
        # Load validation data
        loader = QASPERValidationLoader(self.config)
        validation_questions = loader.load_validation_set(self.config.test_set_size)
        
        if not validation_questions:
            print("❌ No validation questions loaded!")
            return
        
        # Load Phase 2 models
        print("\n🔄 LOADING PHASE 2 MODELS")
        print("-" * 80)
        
        models_dir = Path(self.config.phase2_models_dir)
        model_file = models_dir / f"model_iter{self.config.best_iteration}.pkl"
        scaler_file = models_dir / f"scaler_iter{self.config.best_iteration}.pkl"
        
        if not model_file.exists() or not scaler_file.exists():
            print(f"❌ Phase 2 models not found!")
            return
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"  ✓ Loaded model (Iteration {self.config.best_iteration})")
        print(f"  ✓ Loaded scaler")
        
        runner = Phase3StrategyRunner(self.config)
        
        # Evaluate
        print("\n🎯 EVALUATING ON VALIDATION SET")
        print("-" * 80)
        
        predictions = []
        strategy_names = ["simple", "long_context", "agentic"]

        wins = 0
        losses = 0
        draws = 0

        baseline_f1_sum = 0.0
        learned_f1_sum = 0.0

        pbar = tqdm(validation_questions, desc="Evaluating", dynamic_ncols=True)

        
        for question in pbar:
            q_id = question["id"]
            query = question["query"]
            
            try:
                # 1. RUN BASELINE
                baseline_result = runner.run_strategy("simple", question)
                baseline_f1 = baseline_result["f1_score"]
                baseline_em = baseline_result["em_score"]
                baseline_time = baseline_result["total_time"]
                
                # 2. EXTRACT FEATURES
                features_dict = runner.feature_extractor.extract_all(
                    query=query,
                    answer=baseline_result["answer"],
                    passages=baseline_result["retrieved_passages"],
                    similarities=baseline_result["similarities"],
                    retrieval_time=baseline_result["retrieval_time"],
                    generation_time=baseline_result["generation_time"],
                    strategy_name="simple"
                )
                
                # 3. PREDICT STRATEGY
                feature_vector = self._build_feature_vector(features_dict)
                feature_vector_scaled = scaler.transform([feature_vector])
                
                predicted_idx = model.predict(feature_vector_scaled)[0]
                predicted_strategy = strategy_names[predicted_idx]
                
                probabilities = model.predict_proba(feature_vector_scaled)[0]
                confidence = float(np.max(probabilities))

                # 4. RUN LEARNED STRATEGY
                if predicted_strategy == "simple":
                    learned_result = baseline_result
                else:
                    learned_result = runner.run_strategy(predicted_strategy, question)

                learned_f1 = learned_result["f1_score"]
                learned_em = learned_result["em_score"]
                learned_time = learned_result["total_time"]
                
                # Determine outcome (win/draw/loss)
                outcome = compare_results(baseline_f1, learned_f1, tolerance=self.config.draw_tolerance)

                # APPLYING CHNAGE STO PBAR
                baseline_f1_sum += baseline_f1
                learned_f1_sum += learned_f1

                if outcome == "learned_wins":
                    wins += 1
                elif outcome == "baseline_wins":
                    losses += 1
                else:
                    draws += 1

                processed = wins + losses + draws

                current_improvement = (
                    (learned_f1_sum - baseline_f1_sum) / baseline_f1_sum
                    if baseline_f1_sum > 0
                    else 0.0
                )

                pbar.set_postfix({
                    "Wins": wins,
                    "Losses": losses,
                    "Draws": draws,
                    "F1": f"{current_improvement:+.1%}",
                    "Route": predicted_strategy
                })
                
                predictions.append({
                    "question_id": q_id,
                    "query": query[:100],
                    "rcs" : question["rcs"],
                    "category": question["category"],
                    
                    "baseline_strategy": "simple",
                    "baseline_f1": baseline_f1,
                    "baseline_em": baseline_em,
                    "baseline_time": baseline_time,
                    
                    "learned_strategy": predicted_strategy,
                    "learned_f1": learned_f1,
                    "learned_em": learned_em,
                    "learned_time": learned_time,
                    "learned_confidence": confidence,
                    "learned_probabilities": {
                        strategy_names[i]: float(probabilities[i])
                        for i in range(3)
                    },
                    
                    "f1_delta": learned_f1 - baseline_f1,
                    "em_delta": learned_em - baseline_em,
                    "time_delta": learned_time - baseline_time,
                    
                    "outcome": outcome,
                    "learned_wins_f1": learned_f1 > baseline_f1,
                    "is_draw": outcome == "draw",
                    "learned_loses": outcome == "baseline_wins"
                })
            
            except Exception as e:
                print(f"  ❌ Q{q_id}: {str(e)[:50]}")
                continue
        
        print(f"✓ Evaluated {len(predictions)} questions")
        
        # Evaluate
        evaluator = Phase3Evaluator(self.config)
        evaluation = evaluator.evaluate(predictions)
        
        # Save results
        print("\n💾 SAVING RESULTS")
        print("-" * 80)
        
        pred_file = self.config.output_dir / "phase3_predictions.json"
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"  ✓ Predictions: {pred_file}")
        
        eval_file = self.config.output_dir / "phase3_evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"  ✓ Evaluation: {eval_file}")
        
        # Summary
        print("\n" + "="*80)
        print("PHASE 3 FINAL RESULTS")
        print("="*80)
        
        print(f"\n✅ F1 IMPROVEMENT: {evaluation.get('f1_improvement', 0):+.1%}")
        print(f"✅ EM IMPROVEMENT: {evaluation.get('em_improvement', 0):+.1%}")
        print(f"\n📊 COMPETITION RESULTS:")
        print(f"   Learned WINS:  {evaluation.get('learned_wins_pct', 0):.1f}%")
        print(f"   DRAWS:         {evaluation.get('draws_pct', 0):.1f}%")
        print(f"   Baseline WINS: {evaluation.get('baseline_wins_pct', 0):.1f}%")
        print(f"\n⏱️  LATENCY: {evaluation.get('latency_improvement', 0):+.1%}")
        
        print("\n" + "="*80)
        
        return predictions, evaluation
    
    @staticmethod
    def _build_feature_vector(features_dict: Dict) -> np.ndarray:
        """Build feature vector (27 features, alphabetically sorted per group)."""
        features = []
        for group in ["A", "B", "C", "D", "E"]:
            for key in sorted(features_dict.keys()):
                if key.startswith(f"{group}_"):
                    features.append(features_dict[key])
        return np.array(features, dtype=np.float32)


def main():
    """Entry point."""
    config = Phase3ProdConfig(
        phase2_models_dir="./phase2_models_and_results/models",
        phase2_results_dir="./phase2_models_and_results",
        output_dir="./phase3_results_jatt_chle_gye_finally",
        test_set_size=100,
        best_iteration=5,
        draw_tolerance=0.0  # 1% tolerance for draws
    )
    
    pipeline = Phase3ProdPipeline(config)
    predictions, evaluation = pipeline.run()


if __name__ == "__main__":
    main()