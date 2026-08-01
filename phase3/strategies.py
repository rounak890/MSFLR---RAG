"""
RAG Strategy Implementations for Phase 1 Pilot - OLLAMA VERSION
Uses local Ollama model (qwen2.5:7b or any model you have)
- Simple RAG: Dense retrieval + direct generation
- Long-Context RAG: Retrieve top-20, concatenate, single LLM call
- Agentic RAG: Decompose → retrieve → synthesize
"""

import os
import json
import time
import requests
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Any
from pathlib import Path

import numpy as np

# For embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️ Install sentence-transformers: pip install sentence-transformers")

# For FAISS retrieval
try:
    import faiss
except ImportError:
    print("⚠️ Install faiss: pip install faiss-cpu")


class OllamaClient:
    """Wrapper for local Ollama model."""
    
    def __init__(self, model: str = "qwen3:0.6b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/generate"
        self.token_counter = TokenCounter()
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> Tuple[str, Dict]:
        """
        Call Ollama API locally.
        Returns: (response_text, metadata with token counts)
        """
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": max_tokens,
                },
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} {response.text}")
            
            data = response.json()
            text = data.get("response", "").strip()
            
            # Estimate token counts
            prompt_tokens = self.token_counter.count(prompt)
            response_tokens = self.token_counter.count(text)
            
            metadata = {
                "token_count": prompt_tokens + response_tokens,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "model": self.model
            }
            
            return text, metadata
        
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"❌ Cannot connect to Ollama at {self.base_url}\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"And pull the model: ollama pull {self.model}"
            )


class BaseRAGStrategy(ABC):
    """Abstract base class for all strategies."""
    
    def __init__(self, model: str = "qwen3:0.6b", ollama_url: str = "http://localhost:11434"):
        self.ollama = OllamaClient(model=model, base_url=ollama_url)
        self.model = model
        self.token_counter = TokenCounter()
    
    @abstractmethod
    def run(self, query: str, question_data: Dict) -> Tuple[str, Dict]:
        """
        Run strategy on query.
        Returns: (answer: str, metadata: Dict with tokens, retrieval info, etc.)
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.token_counter.count(text)
    
    def call_ollama(self, system: str, user_msg: str, max_tokens: int = 500) -> Tuple[str, Dict]:
        """
        Call Ollama and return response + metadata.
        """
        prompt = f"{system}\n\n{user_msg}"
        return self.ollama.generate(prompt, max_tokens=max_tokens)


class SimpleRAG(BaseRAGStrategy):
    """
    Strategy A: Simple RAG
    Dense retrieval (FAISS) → top-5 passages → direct LLM answer
    Fast, cheap, but fails on multi-hop queries.
    """
    
    def __init__(self, model: str = "qwen3:0.6b", ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(model=model, ollama_url=ollama_url)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.faiss_index = None
        self.passage_store = []
    
    def build_index(self, passages: List[str]):
        """Build FAISS index from passages."""
        if not passages:
            return
        
        embeddings = self.embedding_model.encode(passages, convert_to_numpy=True)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings.astype('float32'))
        self.passage_store = passages
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k passages via dense search."""
        if self.faiss_index is None or not self.passage_store:
            return []
        
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_emb.astype('float32'), min(top_k, len(self.passage_store)))
        
        return [self.passage_store[i] for i in indices[0] if i < len(self.passage_store)]
    
    def run(self, query: str, question_data: Dict) -> Tuple[str, Dict]:
        """
        Dense retrieval (FAISS top-3) → direct LLM call.
        """
        passages = self._extract_passages(question_data)
        if not passages:
            passages = [question_data.get("abstract", "No context available")]
        
        self.build_index(passages)
        
        # Retrieve
        retrieved = self.retrieve(query, top_k=3)
        if not retrieved:
            retrieved = passages[:3]
        
        context = "\n\n".join(retrieved)
        
        # Generate answer
        system = "You are a research assistant. Answer the following question based ONLY on the provided passages. Be concise and accurate."
        user_msg = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        
        answer, metadata = self.call_ollama(system, user_msg, max_tokens=300)
        
        metadata["strategy"] = "simple"
        metadata["retrieved_passages"] = len(retrieved)
        metadata["context_tokens"] = self.count_tokens(context)
        
        return answer.strip(), metadata
    
    def _extract_passages(self, question_data: Dict) -> List[str]:
        """Extract text passages from question data."""
        passages = []
        
        # From full_text (list of sections)
        if "full_text" in question_data and question_data["full_text"]:
            full_text = question_data["full_text"]
            if isinstance(full_text, list):
                passages.extend([str(item).strip() for item in full_text if item])
            else:
                # Split by period if it's a single string
                passages.extend([s.strip() for s in str(full_text).split(". ") if s.strip()])
        
        # Add abstract as fallback
        if "abstract" in question_data and question_data["abstract"]:
            passages.append(question_data["abstract"])
        
        return passages[:100] if passages else []


class LongContextRAG(BaseRAGStrategy):
    """
    Strategy B: Long-Context RAG
    Dense retrieval → top-20 passages → concatenate → single LLM call.
    Better reasoning, but expensive and slow.
    """
    
    def __init__(self, model: str = "qwen3:0.6b", ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(model=model, ollama_url=ollama_url)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.faiss_index = None
        self.passage_store = []
    
    def build_index(self, passages: List[str]):
        """Build FAISS index from passages."""
        if not passages:
            return
        
        embeddings = self.embedding_model.encode(passages, convert_to_numpy=True)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings.astype('float32'))
        self.passage_store = passages
    
    def retrieve(self, query: str, top_k: int = 20) -> List[str]:
        """Retrieve top-k passages."""
        if self.faiss_index is None or not self.passage_store:
            return []
        
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_emb.astype('float32'), min(top_k, len(self.passage_store)))
        
        return [self.passage_store[i] for i in indices[0] if i < len(self.passage_store)]
    
    def run(self, query: str, question_data: Dict) -> Tuple[str, Dict]:
        """
        Retrieve top-20 passages, concatenate, single LLM call.
        """
        passages = self._extract_passages(question_data)
        if not passages:
            passages = [question_data.get("abstract", "No context available")]
        
        self.build_index(passages)
        
        retrieved = self.retrieve(query, top_k=20)
        if not retrieved:
            retrieved = passages[:20]
        
        context = "\n\n".join(retrieved)
        
        # Limit context for local model (avoid token explosion)
        max_context_tokens = 8000
        current_tokens = self.count_tokens(context)
        if current_tokens > max_context_tokens:
            ratio = max_context_tokens / current_tokens
            context = context[:int(len(context) * ratio)]
        
        system = """You are a research assistant. Read the full context and answer the question. You may infer relationships between different parts of the context.
        Instructions:

        1. Identify all relevant evidence.
        2. Explain how each piece contributes.
        3. Resolve contradictions if any.
        4. Combine the evidence.
        5. Produce the final answer.

        """
        user_msg = f"Question: {query}\n\nFull Context:\n{context}\n\nAnswer:"
        
        answer, metadata = self.call_ollama(system, user_msg, max_tokens=300)
        
        metadata["strategy"] = "long_context"
        metadata["retrieved_passages"] = len(retrieved)
        metadata["context_tokens"] = self.count_tokens(context)
        
        return answer.strip(), metadata
    
    def _extract_passages(self, question_data: Dict) -> List[str]:
        """Extract text passages from question data."""
        passages = []
        
        # From full_text (list of sections)
        if "full_text" in question_data and question_data["full_text"]:
            full_text = question_data["full_text"]
            if isinstance(full_text, list):
                passages.extend([str(item).strip() for item in full_text if item])
            else:
                passages.extend([s.strip() for s in str(full_text).split(". ") if s.strip()])
        
        # Add abstract
        if "abstract" in question_data and question_data["abstract"]:
            passages.append(question_data["abstract"])
        
        return passages[:200] if passages else []


class AgenticRAG(BaseRAGStrategy):
    """
    Strategy C: Agentic RAG
    Decompose query → retrieve for each sub-question → synthesize.
    Handles complex reasoning but slow and prone to errors.
    """
    
    def __init__(self, model: str = "qwen3:0.6b", ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(model=model, ollama_url=ollama_url)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.faiss_index = None
        self.passage_store = []
    
    def build_index(self, passages: List[str]):
        if not passages:
            return
        
        embeddings = self.embedding_model.encode(passages, convert_to_numpy=True)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings.astype('float32'))
        self.passage_store = passages
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.faiss_index is None or not self.passage_store:
            return []
        
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_emb.astype('float32'), min(top_k, len(self.passage_store)))
        
        return [self.passage_store[i] for i in indices[0] if i < len(self.passage_store)]
    
    def decompose_query(self, query: str) -> List[str]:
        """Use LLM to break query into sub-questions."""
        # system = "Break down this query into 2-3 independent sub-questions that together answer the main query. Return only the sub-questions, one per line. No numbering."
        # user_msg = f"Query: {query}"
        system = "Break down this query into 2-3 independent sub-questions that together answer the main query. Return only the sub-questions, one per line. No numbering."
        system = """
            You are a question decomposition system.

            Split the question into at most 2 independent factual questions.

            Rules:
            - Produce only questions.
            - One question per line.
            - No numbering.
            - No explanations.
            - Keep each question under 15 words.
            - Do not rewrite if decomposition is unnecessary.

            """
        user_msg = f"Query: {query}"
        
        response, _ = self.call_ollama(system, user_msg, max_tokens=200)
        
        sub_qs = [line.strip() for line in response.split("\n") if line.strip() and not line.startswith("#")]
        return sub_qs[:3]  # Max 3 sub-questions (to save time)
    
    def run(self, query: str, question_data: Dict) -> Tuple[str, Dict]:
        """
        Decompose → retrieve for each sub-question → synthesize.
        """
        passages = self._extract_passages(question_data)
        if not passages:
            passages = [question_data.get("abstract", "No context available")]
        
        self.build_index(passages)
        
        # Decompose
        sub_questions = self.decompose_query(query)
        if not sub_questions:
            # Fallback: just use original query
            sub_questions = [query]
        
        # Retrieve and answer each sub-question
        sub_answers = []
        for sub_q in sub_questions:
            retrieved = self.retrieve(sub_q, top_k=3)
            if not retrieved:
                retrieved = passages[:3]
            
            context = "\n\n".join(retrieved)

            
            # system = "Answer this sub-question based on the context. Be concise and factual."
            # user_msg = f"Sub-question: {sub_q}\n\nContext:\n{context}\n\nAnswer:"

            system = """
                    You answer questions using ONLY the provided context.

            Rules:
            - Use only information from the context.
            - Copy names, numbers and dates exactly.
            - Do not explain your reasoning.
            - If the answer is missing, reply exactly:
            Insufficient information.
            - Keep the answer under 25 words.
            """
            user_msg = f"Sub-question: {sub_q}\n\nContext:\n{context}\n\nAnswer:"


            sub_answer, _ = self.call_ollama(system, user_msg, max_tokens=200)
            sub_answers.append(f"Q: {sub_q}\nA: {sub_answer.strip()}")
        
        # Synthesize
        synthesis_context = "\n\n".join(sub_answers)
        # system = "Synthesize the following sub-answers into a single coherent final answer to the main query."
        # user_msg = f"Main Query: {query}\n\nSub-answers:\n{synthesis_context}\n\nFinal Answer:"

        system = """You are combining answers to answer the original question.

        Rules:
        - Use ONLY the sub-answers.
        - Copy important entities exactly.
        - Do not paraphrase names.
        - If multiple facts are needed, combine them into one sentence.
        - If some sub-answer says "Insufficient information", ignore it.
        - Output ONLY the final answer.
        - Maximum 40 words.
        """

        user_msg = f"Original Question: {query}\n\nSub-answers:\n{synthesis_context}\n\nFinal Answer:"

        final_answer, metadata = self.call_ollama(system, user_msg, max_tokens=300)
        
        metadata["strategy"] = "agentic"
        metadata["num_sub_questions"] = len(sub_questions)
        metadata["sub_questions"] = sub_questions
        
        return final_answer.strip(), metadata
    
    def _extract_passages(self, question_data: Dict) -> List[str]:
        passages = []
        
        # From full_text (list of sections)
        if "full_text" in question_data and question_data["full_text"]:
            full_text = question_data["full_text"]
            if isinstance(full_text, list):
                passages.extend([str(item).strip() for item in full_text if item])
            else:
                passages.extend([s.strip() for s in str(full_text).split(". ") if s.strip()])
        
        # Add abstract
        if "abstract" in question_data and question_data["abstract"]:
            passages.append(question_data["abstract"])
        
        return passages[:200] if passages else []



class TokenCounter:
    """Utility to estimate token counts."""
    
    def count(self, text: str) -> int:
        """Rough token count (avg 4 chars per token for English)."""
        return len(text) // 4