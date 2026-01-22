import numpy as np
from sklearn.metrics import ndcg_score, mean_absolute_error
from scipy.stats import kendalltau, spearmanr
from typing import List, Dict

def calculate_metrics(rankings: List[str], ground_truth: Dict[str, float], k: int = 3) -> Dict:
    """
    Calculate evaluation metrics for a ranked list of candidates.
    
    Args:
        rankings: List of candidate IDs in ranked order (highest score first)
        ground_truth: Dict mapping candidate ID to relevance score (0.0-1.0)
        k: The rank at which to calculate metrics
        
    Returns:
        Dict containing nDCG@K, P@1, R@K, etc.
    """
    # Create vectors for ndcg_score
    # We need to map the rankings back to the ground truth order
    sorted_ids = sorted(ground_truth.keys())
    
    # True relevance scores in sorted_ids order
    y_true = [ground_truth[cid] for cid in sorted_ids]
    
    # Predicted relevance scores (we need to assign scores to each ID based on their rank)
    # Give highest rank (0) a score of N, next N-1, etc.
    # Or better: use the inverse of the rank index as the 'score'
    id_to_rank = {cid: i for i, cid in enumerate(rankings)}
    
    # If a candidate is not in rankings, they get the worst rank
    default_rank = len(rankings)
    y_score = [1.0 / (id_to_rank.get(cid, default_rank) + 1) for cid in sorted_ids]
    
    # nDCG@K
    try:
        ndcg_val = ndcg_score([y_true], [y_score], k=k)
    except Exception:
        ndcg_val = 0.0
        
    # Precision@1 (Is the #1 ranked candidate in the top-K of ground truth?)
    # Get ground truth top candidates (those with score >= 0.8)
    qualified = [cid for cid, score in ground_truth.items() if score >= 0.8]
    p_at_1 = 1.0 if rankings[0] in qualified else 0.0
    
    # Recall@K (What fraction of qualified candidates are in the predicted top-K?)
    top_k_pred = set(rankings[:k])
    recall_at_k = len(set(qualified) & top_k_pred) / len(qualified) if qualified else 0.0
    
    return {
        "ndcg_at_3": round(ndcg_val, 3),
        "p_at_1": round(p_at_1, 3),
        "recall_at_3": round(recall_at_k, 3)
    }
