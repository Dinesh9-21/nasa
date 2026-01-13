#!/usr/bin/env python3
"""
Batch Evaluation Script for NASA RAG System

Runs end-to-end evaluation on a test question set and computes
per-question and aggregate RAGAS metrics.
"""

import json
import os
import statistics
from typing import List, Dict

import rag_client
import llm_client
import ragas_evaluator


# ===================== CONFIGURATION =====================

TEST_QUESTIONS_FILE = "test_questions.json"
N_RESULTS = 3
MODEL_NAME = "gpt-3.5-turbo"

# Set OpenAI key via env variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable")


# ===================== HELPERS =====================

def load_test_questions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute mean score per metric"""
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


# ===================== MAIN EVALUATION =====================

def main():
    print("🚀 Starting batch evaluation...")

    # Load test questions
    questions = load_test_questions(TEST_QUESTIONS_FILE)
    print(f"Loaded {len(questions)} test questions")

    # Discover and initialize first available Chroma backend
    backends = rag_client.discover_chroma_backends()
    if not backends:
        raise RuntimeError("No ChromaDB backends found")

    backend_key = list(backends.keys())[0]
    backend = backends[backend_key]

    print(f"Using backend: {backend['display_name']}")

    collection = rag_client.initialize_rag_system(
        backend["path"],
        backend["collection"]
    )

    per_question_results = []

    # ===================== LOOP =====================
    for item in questions:
        qid = item.get("id", "unknown")
        question = item["question"]

        print(f"\n🔍 Question {qid}: {question}")

        # Retrieve documents
        retrieval = rag_client.retrieve_documents(
            collection,
            question,
            n_results=N_RESULTS
        )

        documents = retrieval["documents"][0]
        metadatas = retrieval["metadatas"][0]

        context = rag_client.format_context(documents, metadatas)

        # Generate answer
        answer = llm_client.generate_response(
            openai_key=OPENAI_API_KEY,
            user_message=question,
            context=context,
            conversation_history=[],
            model=MODEL_NAME
        )

        # Evaluate answer
        metrics = ragas_evaluator.evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=documents
        )

        per_question_results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "metrics": metrics
        })

        print("Metrics:", metrics)

    # ===================== AGGREGATION =====================

    aggregate = aggregate_metrics(per_question_results)

    print("\n📊 Aggregate Metrics")
    for metric, score in aggregate.items():
        print(f"{metric}: {score:.4f}")

    # Save results
    output = {
        "per_question": per_question_results,
        "aggregate": aggregate
    }

    with open("batch_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Batch evaluation complete")
    print("Results saved to batch_evaluation_results.json")


if __name__ == "__main__":
    main()
