"""
Data preparation: Convert raw QASPER to pilot-ready format.
This script:
1. Loads raw QASPER JSON
2. Validates questions/answers
3. Associates PDFs
4. Saves in pilot format
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def load_raw_qasper(path: str) -> List[Dict]:
    """Load raw QASPER dataset."""
    with open(path) as f:
        data = json.load(f)
    
    return data if isinstance(data, list) else list(data.values())


def validate_question(q: Dict) -> bool:
    """Check if question has required fields."""
    required = ["question_id", "question", "answers"]
    return all(field in q for field in required)


def associate_pdf(question: Dict, pdf_dir: Path) -> Dict:
    """Try to find associated PDF."""
    if "paper_id" in question:
        pdf_path = pdf_dir / f"{question['paper_id']}.pdf"
        if pdf_path.exists():
            question["pdf_path"] = str(pdf_path)
    
    return question


def prepare_pilot_data(
    raw_qasper_path: str,
    pdf_dir: str = "./arxiv_pdfs",
    output_path: str = "./qasper_data/qasper.json",
    seed: int = 42
):
    """
    Prepare QASPER data for pilot.
    
    Args:
        raw_qasper_path: Path to raw QASPER JSON
        pdf_dir: Directory containing PDF files
        output_path: Where to save pilot-ready data
        seed: Random seed for reproducibility
    """
    print("="*70)
    print("PREPARING DATA FOR PHASE 1 PILOT")
    print("="*70)
    
    random.seed(seed)
    pdf_dir = Path(pdf_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    print(f"\n1. Loading raw QASPER from {raw_qasper_path}...")
    data = load_raw_qasper(raw_qasper_path)
    print(f"   ✓ Loaded {len(data)} items")
    
    # Validate
    print(f"\n2. Validating questions...")
    valid = [q for q in data if validate_question(q)]
    print(f"   ✓ {len(valid)} valid questions")
    
    # Filter: keep only questions with gold answers
    print(f"\n3. Filtering: questions with ground-truth answers...")
    with_answers = [q for q in valid if q.get("answers")]
    print(f"   ✓ {len(with_answers)} questions have answers")
    
    # Associate PDFs
    print(f"\n4. Associating PDFs...")
    for q in with_answers:
        associate_pdf(q, pdf_dir)
    
    has_pdf = sum(1 for q in with_answers if "pdf_path" in q)
    print(f"   ✓ {has_pdf}/{len(with_answers)} have associated PDFs")
    
    # Normalize answer format
    print(f"\n5. Normalizing answer format...")
    for q in with_answers:
        answers = q["answers"]
        
        # If answers is a dict (HF format), extract values
        if isinstance(answers, dict):
            if "text" in answers:
                # HuggingFace SQuAD-like format
                q["answers"] = answers["text"] if isinstance(answers["text"], list) else [answers["text"]]
            else:
                # Assume dict values are answers
                q["answers"] = list(answers.values())
        
        # Ensure answers is a list of strings
        if isinstance(q["answers"], list):
            q["answers"] = [str(a).strip() for a in q["answers"]]
        else:
            q["answers"] = [str(q["answers"]).strip()]
    
    print(f"   ✓ Normalized")
    
    # Extract text from PDFs (optional, for faster startup)
    print(f"\n6. Extracting text from PDFs (optional)...")
    extracted = 0
    try:
        import pymupdf
        for q in with_answers:
            if "pdf_path" in q and "full_text" not in q:
                try:
                    doc = pymupdf.open(q["pdf_path"])
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    q["full_text"] = text[:10000]  # First 10k chars
                    extracted += 1
                except Exception as e:
                    print(f"   ⚠️ Could not extract {q.get('paper_id')}: {e}")
    except ImportError:
        print("   ⚠️ PyMuPDF not available, skipping text extraction")
    
    print(f"   ✓ Extracted text from {extracted} PDFs")
    
    # Save
    print(f"\n7. Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(with_answers, f, indent=2)
    
    print(f"   ✓ Saved {len(with_answers)} questions")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total questions: {len(with_answers)}")
    print(f"With PDF: {has_pdf}")
    print(f"With extracted text: {extracted}")
    print(f"Output: {output_path}")
    
    return with_answers


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare QASPER data for pilot")
    parser.add_argument("--raw", default="./qasper_raw.json", help="Raw QASPER path")
    parser.add_argument("--pdfs", default="./arxiv_pdfs", help="PDF directory")
    parser.add_argument("--output", default="./qasper_data/qasper.json", help="Output path")
    
    args = parser.parse_args()
    
    prepare_pilot_data(
        raw_qasper_path=args.raw,
        pdf_dir=args.pdfs,
        output_path=args.output
    )
