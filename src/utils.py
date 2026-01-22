import numpy as np
from sklearn.metrics import ndcg_score
from typing import List, Dict, Any
from pathlib import Path
import json
import os

def load_job_description(path: str = "data/job_descriptions/ema_ai_apps_engineer.txt") -> str:
    """Load the job description from file."""
    with open(path, 'r') as f:
        return f.read()

def load_resumes(directory: str = "data/resumes") -> List[Dict[str, str]]:
    """Load all resumes from directory."""
    resumes = []
    resume_dir = Path(directory)
    
    for file_path in sorted(resume_dir.glob("*.txt")):
        if "edge" in file_path.name or "test" in file_path.name:
            continue
            
        with open(file_path, 'r') as f:
            resumes.append({
                "id": file_path.stem,
                "text": f.read(),
                "filename": file_path.name
            })
    return resumes

def load_ground_truth(path: str = "data/ground_truth.json") -> Dict[str, float]:
    """Load ground truth labels."""
    with open(path, 'r') as f:
        return json.load(f)

def save_results(results: List[Dict], path: str):
    """Save results to JSON file."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

def calculate_ndcg_at_k(predicted_scores: List[float], true_scores: List[float], k: int = 3) -> float:
    """Calculate nDCG@K."""
    try:
        return ndcg_score([true_scores], [predicted_scores], k=k)
    except Exception:
        return 0.0

def calculate_precision_at_k(predicted_scores: List[float], true_scores: List[float], k: int = 1) -> float:
    """Calculate Precision@K (is top predicted in top true)."""
    # Get indices of top-k predicted
    pred_top_k = set(np.argsort(predicted_scores)[-k:][::-1])
    # Get indices of true positives (score >= 0.8)
    true_positives = set(i for i, s in enumerate(true_scores) if s >= 0.8)
    
    if not true_positives:
        return 0.0
    
    # Check if any of top-k predicted are true positives
    return len(pred_top_k & true_positives) / k

def calculate_recall_at_k(predicted_scores: List[float], true_scores: List[float], k: int = 3) -> float:
    """Calculate Recall@K."""
    pred_top_k = set(np.argsort(predicted_scores)[-k:][::-1])
    true_positives = set(i for i, s in enumerate(true_scores) if s >= 0.8)
    
    if not true_positives:
        return 0.0
    
    return len(pred_top_k & true_positives) / len(true_positives)

def print_ranking(results: List[Dict[str, Any]]):
    """Print ranked results."""
    print("\n📊 Final Rankings:")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        name = r.get("candidate_name", r.get("id", "Unknown"))
        score = r.get("final_score", 0)
        print(f"  {i}. {name}: {score:.3f}")
    print("-" * 60)
    
def print_metrics(ndcg: float, precision: float, recall: float):
    """Print evaluation metrics."""
    print("\n📈 Evaluation Metrics:")
    print("-" * 40)
    print(f"  nDCG@3:      {ndcg:.3f}")
    print(f"  Precision@1: {precision:.3f}")
    print(f"  Recall@3:    {recall:.3f}")
    print("-" * 40)

def calculate_metrics(rankings: List[str], ground_truth: Dict[str, float], k: int = 3) -> Dict:
    # Legacy wrapper for calculate_metrics if needed by other scripts
    # For evaluate_v2/v3 compatibility
    sorted_ids = sorted(ground_truth.keys())
    y_true = [ground_truth[cid] for cid in sorted_ids]
    
    id_to_rank = {cid: i for i, cid in enumerate(rankings)}
    default_rank = len(rankings)
    y_score = [1.0 / (id_to_rank.get(cid, default_rank) + 1) for cid in sorted_ids]
    
    try:
        ndcg_val = ndcg_score([y_true], [y_score], k=k)
    except:
        ndcg_val = 0.0
        
    qualified = [cid for cid, score in ground_truth.items() if score >= 0.8]
    p_at_1 = 1.0 if rankings[0] in qualified else 0.0
    
    top_k_pred = set(rankings[:k])
    recall_at_k = len(set(qualified) & top_k_pred) / len(qualified) if qualified else 0.0
    
    return {
        "ndcg_at_3": round(ndcg_val, 3),
        "p_at_1": round(p_at_1, 3),
        "recall_at_3": round(recall_at_k, 3)
    }
