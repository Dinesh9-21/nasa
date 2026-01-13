#!/usr/bin/env python3
"""
Batch Evaluation Script for NASA RAG System
"""

import os
import json
import statistics
from typing import List, Dict

import rag_client
import llm_client
import ragas_evaluator


# ===================== CONFIGURATION =====================

TEST_QUESTIONS_FILE = "evaluation_dataset.txt"
N_RESULTS = 3
MODEL_NAME = "gpt-3.5-turbo"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable")


# ===================== HELPERS =====================

def load_test_questions(path: str) -> List[Dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append({"question": line})

    if len(questions) < 5:
        raise ValueError("Evaluation dataset must contain at least 5 questions")

    return questions


def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    aggregates = {}

    for result in results:
        for metric, value in result["metrics"].items():
            if isinstance(value, (int, float)):
                aggregates.setdefault(metric, []).append(value)

    return {
        metric: statistics.mean(values)
        for metric, values in aggregates.items()
        if values
    }


# ===================== MAIN =====================

def main():
    print("🚀 Starting batch evaluation")

    questions = load_test_questions(TEST_QUESTIONS_FILE)
    print(f"Loaded {len(questions)} evaluation questions")

    backends = rag_client.discover_chroma_backends()
    if not backends:
        raise RuntimeError("No ChromaDB backends found")

    backend = list(backends.values())[0]
    print(f"Using backend: {backend['display_name']}")

    collection = rag_client.initialize_rag_system(
        backend["path"],
        backend["collection"]
    )

    per_question_results = []

    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"\n🔍 Question {idx}: {question}")

        retrieval = rag_client.retrieve_documents(
            collection,
            question,
            n_results=N_RESULTS
        )

        if not retrieval or not retrieval.get("documents"):
            print("⚠️ No documents retrieved, skipping")
            continue

        documents = retrieval["documents"][0]
        metadatas = retrieval["metadatas"][0]
        context = rag_client.format_context(documents, metadatas)

        answer = llm_client.generate_response(
            openai_key=OPENAI_API_KEY,
            user_message=question,
            context=context,
            conversation_history=[],
            model=MODEL_NAME
        )

        try:
            metrics = ragas_evaluator.evaluate_response_quality(
                question=question,
                answer=answer,
                contexts=documents
            )
        except Exception as e:
            metrics = {"error": str(e)}

        per_question_results.append({
            "question": question,
            "answer": answer,
            "metrics": metrics
        })

        print("Metrics:", metrics)

    aggregate = aggregate_metrics(per_question_results)

    print("\n📊 Aggregate Metrics")
    for metric, score in aggregate.items():
        print(f"{metric}: {score:.4f}")

    with open("batch_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_question": per_question_results,
                "aggregate": aggregate
            },
            f,
            indent=2
        )

    print("\n✅ Batch evaluation complete")


if __name__ == "__main__":
    main()
