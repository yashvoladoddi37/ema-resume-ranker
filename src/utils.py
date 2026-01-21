"""
Utility functions for data loading and metric calculation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np


def load_job_description(filepath: str = "data/job_descriptions/ema_ai_apps_engineer.txt") -> str:
    """Load job description from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_resumes(directory: str = "data/resumes/") -> List[Dict[str, str]]:
    """
    Load all resumes from directory.
    
    Returns:
        List of {"id": "resume_001", "text": "resume content"}
    """
    resumes = []
    resume_dir = Path(directory)
    
    for filepath in sorted(resume_dir.glob("*.txt")):
        resume_id = filepath.stem  # Filename without extension
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        resumes.append({
            "id": resume_id,
            "text": text
        })
    
    print(f"📁 Loaded {len(resumes)} resumes from {directory}")
    return resumes


def load_ground_truth(filepath: str = "data/ground_truth.json") -> Dict[str, float]:
    """
    Load manual labels for evaluation.
    
    Format:
    {
        "resume_001": 1.0,  // Perfect match
        "resume_002": 0.5,  // Partial match
        "resume_003": 0.0   // Poor match
    }
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results: List[Dict], filepath: str = "results.json"):
    """Save evaluation results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to {filepath}")


def calculate_ndcg_at_k(
    predicted_scores: List[float],
    true_relevance: List[float],
    k: int = 3
) -> float:
    """
    Calculate nDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        predicted_scores: Model's scores (0.0-1.0)
        true_relevance: Ground truth relevance (0.0, 0.5, 1.0)
        k: Number of top results to consider
        
    Returns:
        nDCG score (0.0-1.0), where 1.0 is perfect ranking
    """
    # Get indices sorted by predicted scores (descending)
    predicted_order = np.argsort(predicted_scores)[::-1][:k]
    
    # Get indices sorted by true relevance (descending) for ideal DCG
    ideal_order = np.argsort(true_relevance)[::-1][:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, idx in enumerate(predicted_order):
        rel = true_relevance[idx]
        dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate Ideal DCG
    idcg = 0.0
    for i, idx in enumerate(ideal_order):
        rel = true_relevance[idx]
        idcg += (2 ** rel - 1) / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_precision_at_k(
    predicted_scores: List[float],
    true_relevance: List[float],
    k: int = 1,
    threshold: float = 0.7
) -> float:
    """
    Calculate Precision@k.
    
    Args:
        predicted_scores: Model's scores
        true_relevance: Ground truth (0.0, 0.5, 1.0)
        k: Number of top results to consider
        threshold: Relevance threshold for "good" candidate
        
    Returns:
        Precision (0.0-1.0)
    """
    # Get top-k indices by predicted score
    top_k_indices = np.argsort(predicted_scores)[::-1][:k]
    
    # Check if top-k are actually relevant
    relevant_count = sum(
        1 for idx in top_k_indices 
        if true_relevance[idx] >= threshold
    )
    
    return relevant_count / k


def calculate_recall_at_k(
    predicted_scores: List[float],
    true_relevance: List[float],
    k: int = 3,
    threshold: float = 0.7
) -> float:
    """
    Calculate Recall@k.
    
    Args:
        predicted_scores: Model's scores
        true_relevance: Ground truth (0.0, 0.5, 1.0)
        k: Number of top results to consider
        threshold: Relevance threshold for "good" candidate
        
    Returns:
        Recall (0.0-1.0)
    """
    # Get top-k indices
    top_k_indices = np.argsort(predicted_scores)[::-1][:k]
    
    # Total relevant candidates
    total_relevant = sum(1 for rel in true_relevance if rel >= threshold)
    
    if total_relevant == 0:
        return 0.0
    
    # Relevant candidates in top-k
    relevant_in_top_k = sum(
        1 for idx in top_k_indices 
        if true_relevance[idx] >= threshold
    )
    
    return relevant_in_top_k / total_relevant


def print_ranking(results: List[Dict]):
    """Pretty print ranking results."""
    print("\n" + "="*80)
    print("RANKING RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n#{i} | {result['candidate_name']} ({result['id']})")
        print(f"     Score: {result['final_score']:.3f}")
        dims = result.get('dimension_scores', {})
        print(f"     Skills: {dims.get('skill_match', {}).get('score', 'N/A')} | "
              f"Experience: {dims.get('experience_depth', {}).get('score', 'N/A')} | "
              f"Domain: {dims.get('domain_fit', {}).get('score', 'N/A')}")
        matched = result.get('matched_skills', [])[:5]
        print(f"     Matched: {', '.join(matched) if matched else 'None'}")
        
    print("\n" + "="*80)


def print_metrics(
    ndcg: float, 
    precision: float, 
    recall: float,
    targets: Dict[str, float] = None
):
    """Pretty print evaluation metrics."""
    if targets is None:
        targets = {"ndcg": 0.85, "precision": 1.0, "recall": 0.9}
    
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    print(f"\nnDCG@3:        {ndcg:.3f}  {'✅' if ndcg >= targets['ndcg'] else '❌'} (target: ≥{targets['ndcg']})")
    print(f"Precision@1:   {precision:.3f}  {'✅' if precision >= targets['precision'] else '❌'} (target: ≥{targets['precision']})")
    print(f"Recall@3:      {recall:.3f}  {'✅' if recall >= targets['recall'] else '❌'} (target: ≥{targets['recall']})")
    
    print("="*80)
