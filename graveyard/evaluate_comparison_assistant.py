from __future__ import annotations

import json
from pathlib import Path

from graveyard.comparison_assistant import HybridRetriever, build_document_chunks, load_product_records, safe_answer_question


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVAL_PATH = DATA_DIR / "comparison_questions.json"
RESULTS_PATH = DATA_DIR / "comparison_evaluation_results.json"


def evaluate() -> dict:
    records = load_product_records(DATA_DIR)
    documents = build_document_chunks(records)
    retriever = HybridRetriever(documents)

    with EVAL_PATH.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)

    results = []
    retrieval_hits = 0
    citation_pass = 0
    unsupported_pass = 0

    for item in questions:
        question = item["question"]
        hits = retriever.search(question, top_k=5)
        answer = safe_answer_question(question, retriever)

        retrieval_ok = bool(hits) == bool(item.get("expected_hit", True))
        citation_ok = ("citation" in answer.lower() or "source" in answer.lower()) == bool(item.get("expect_citation", True))
        refusal_ok = ("not found in sources" in answer.lower() or "cannot provide" in answer.lower()) == bool(item.get("expect_refusal", False))

        if hits:
            retrieval_hits += 1
        if "citation" in answer.lower() or "source" in answer.lower():
            citation_pass += 1
        if "not found in sources" in answer.lower() or "cannot provide" in answer.lower():
            unsupported_pass += 1

        results.append(
            {
                "question": question,
                "retrieval_ok": retrieval_ok,
                "citation_ok": citation_ok,
                "refusal_ok": refusal_ok,
                "retrieval_count": len(hits),
                "answer_preview": answer[:200],
            }
        )

    summary = {
        "total_questions": len(questions),
        "retrieval_hit_rate": retrieval_hits / len(questions) if questions else 0.0,
        "citation_coverage": citation_pass / len(questions) if questions else 0.0,
        "unsupported_claim_refusal_rate": unsupported_pass / len(questions) if questions else 0.0,
        "refusal_quality_score": (
            sum(1 for item in results if item["refusal_ok"]) / len(results) if results else 0.0
        ),
        "results": results,
    }
    return summary


if __name__ == "__main__":
    report = evaluate()
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved evaluation report to {RESULTS_PATH}")
