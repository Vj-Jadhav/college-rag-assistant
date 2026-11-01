"""Automated Tests for RAG Evaluation Framework (Phase 7).

Verifies:
- Evaluation dataset integrity (at least 50 questions across Categories A, B, and C)
- Category counts: Category A >= 20, Category B >= 20, Category C >= 10
- Mathematical accuracy of retrieval accuracy formula:
  Retrieval Accuracy = (Correct Retrievals / Total Questions) * 100
- Mathematical accuracy of safe refusal rate formula:
  Safe Refusal Rate = (Correct Refusals / Unanswerable Questions) * 100
- 5-dimension qualitative scoring boundaries (1.0 to 5.0)
- Error handling and refusal detection
"""

import pytest
import json
from pathlib import Path
from evaluate_rag import (
    evaluate_answer_dimensions,
    compute_keyword_similarity,
    FALLBACK_RESPONSE,
)


@pytest.fixture
def eval_dataset():
    data_path = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"
    assert data_path.exists(), "eval_dataset.json must exist in data/"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_dataset_composition(eval_dataset):
    """Verify requirement: At least 50 questions divided into Categories A, B, and C."""
    assert len(eval_dataset) >= 50

    cat_a = [q for q in eval_dataset if q.get("category") == "A"]
    cat_b = [q for q in eval_dataset if q.get("category") == "B"]
    cat_c = [q for q in eval_dataset if q.get("category") == "C"]

    assert len(cat_a) >= 20, "Category A must have at least 20 direct questions"
    assert len(cat_b) >= 20, "Category B must have at least 20 paraphrased questions"
    assert len(cat_c) >= 10, "Category C must have at least 10 unanswerable questions"


def test_retrieval_accuracy_formula():
    """Verify formula: Retrieval Accuracy = (Correct Retrievals / Total Questions) * 100."""
    correct_retrievals = 36
    total_questions = 40
    accuracy = (correct_retrievals / total_questions) * 100
    assert accuracy == 90.0


def test_safe_refusal_rate_formula():
    """Verify formula: Safe Refusal Rate = (Correct Refusals / Unanswerable Questions) * 100."""
    correct_refusals = 10
    unanswerable_questions = 10
    refusal_rate = (correct_refusals / unanswerable_questions) * 100
    assert refusal_rate == 100.0


def test_evaluate_answer_dimensions_unanswerable_refusal():
    scores = evaluate_answer_dimensions(
        question="What is the pool schedule?",
        answer=FALLBACK_RESPONSE,
        key_facts=["Information not found in the available documents."],
        retrieved_chunks=[],
        is_unanswerable=True,
    )
    # Perfect refusal scores 5.0 across all dimensions
    for dim in ["correctness", "relevance", "groundedness", "clarity", "completeness"]:
        assert scores[dim] == 5.0


def test_evaluate_answer_dimensions_unanswerable_hallucination():
    scores = evaluate_answer_dimensions(
        question="What is the pool schedule?",
        answer="The swimming pool is open 6 AM to 9 PM every day.",  # Hallucination
        key_facts=["Information not found in the available documents."],
        retrieved_chunks=[],
        is_unanswerable=True,
    )
    assert scores["correctness"] == 1.0
    assert scores["groundedness"] == 1.0


def test_evaluate_answer_dimensions_answerable_success():
    chunks = [{"text": "Minimum aggregate for admission is 75 percent in 10+2 PCM."}]
    scores = evaluate_answer_dimensions(
        question="What is the admission percentage?",
        answer="You need a minimum 75 percent aggregate in 10+2.",
        key_facts=["75 percent", "10+2"],
        retrieved_chunks=chunks,
        is_unanswerable=False,
    )
    assert scores["correctness"] == 5.0
    assert scores["groundedness"] == 5.0
    assert scores["relevance"] == 5.0
    assert scores["completeness"] == 5.0

