"""
Main evaluation script - Run full system evaluation.
"""

import os
import json
from dotenv import load_dotenv
from src.matching_engine import TwoStageMatchingEngine
from src.utils import (
    load_job_description,
    load_resumes,
    load_ground_truth,
    save_results,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    print_ranking,
    print_metrics
)


def main():
    """Run complete evaluation pipeline."""
    
    # Load environment
    load_dotenv()
    
    # Initialize engine
    engine = TwoStageMatchingEngine(
        api_key=os.getenv("GROQ_API_KEY"),
        backup_key=os.getenv("GROQ_API_KEY_2")
    )
    
    # Load data
    print("📥 Loading data...")
    job_description = load_job_description()
    resumes = load_resumes()
    ground_truth = load_ground_truth()
    
    print(f"✅ Loaded {len(resumes)} resumes")
    print(f"✅ Loaded ground truth for {len(ground_truth)} candidates\n")
    
    # Evaluate all resumes
    results = engine.evaluate_batch(job_description, resumes)
    
    # Save results
    save_results(results, "results.json")
    
    # Calculate metrics
    # Map results to ground truth order
    predicted_scores = []
    true_relevance = []
    
    for result in results:
        resume_id = result["id"]
        if resume_id in ground_truth:
            predicted_scores.append(result["final_score"])
            true_relevance.append(ground_truth[resume_id])
    
    ndcg_3 = calculate_ndcg_at_k(predicted_scores, true_relevance, k=3)
    precision_1 = calculate_precision_at_k(predicted_scores, true_relevance, k=1)
    recall_3 = calculate_recall_at_k(predicted_scores, true_relevance, k=3)
    
    # Print metrics
    print_metrics(ndcg_3, precision_1, recall_3)
    
    # Print ranking
    print_ranking(results)
    
    # Save metrics
    metrics = {
        "nDCG@3": round(ndcg_3, 3),
        "Precision@1": round(precision_1, 3),
        "Recall@3": round(recall_3, 3),
        "weights": engine.weights,
        "total_resumes": len(results)
    }
    
    with open("metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Evaluation complete! Results saved to results.json and metrics.json")
    
    return results, metrics


if __name__ == "__main__":
    main()
