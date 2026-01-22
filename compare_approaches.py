#!/usr/bin/env python3
"""
Comprehensive Comparison of Three Resume Matching Approaches
============================================================

Approach 1: Deterministic (Embeddings + Rules only)
Approach 2: Hybrid (Embeddings + Rules + LLM)
Approach 3: Two-Stage LLM Pipeline

Metrics:
- nDCG@K (Normalized Discounted Cumulative Gain)
- Precision@K
- Kendall Tau correlation
- Spearman correlation
- Mean Absolute Error (MAE)
- Execution time
- Cost estimate
"""

import os
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score, mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

# Import our implementations
from src.scorers.semantic import SemanticScorer
from src.deterministic import DeterministicExtractor
from src.scorers.llm import LLMScorer
from src.matching_engine import TwoStageMatchingEngine

print("=" * 80)
print("RESUME MATCHING APPROACH COMPARISON")
print("=" * 80)

# ============================================================================
# 1. APPROACH 1: DETERMINISTIC (Embeddings + Rules Only, NO LLM)
# ============================================================================

class DeterministicMatchingEngine:
    """Pure deterministic: Embeddings (60%) + Rules (40%), NO LLM"""
    
    def __init__(self):
        self.semantic_scorer = SemanticScorer()
        self.rule_extractor = DeterministicExtractor()
        self.weights = {'semantic': 0.60, 'rule': 0.40}
    
    def score(self, job_desc: str, resume_text: str) -> Dict:
        # Semantic similarity
        semantic_score = self.semantic_scorer.score(job_desc, resume_text)
        
        # Rule-based scoring
        years_exp = self.rule_extractor.extract_years_of_experience(resume_text)
        skill_profile = self.rule_extractor.extract_skills(resume_text)
        domain_relevance = self.rule_extractor.calculate_domain_relevance(resume_text)
        
        rule_score, breakdown = self.rule_extractor.calculate_deterministic_score(
            years_exp, skill_profile, domain_relevance
        )
        
        # Weighted combination
        final_score = (
            self.weights['semantic'] * semantic_score +
            self.weights['rule'] * rule_score
        )
        
        return {
            'final_score': round(final_score, 3),
            'semantic_score': semantic_score,
            'rule_score': rule_score,
            'years_experience': years_exp,
            'skill_coverage': skill_profile.skill_coverage_score
        }

# ============================================================================
# 2. APPROACH 2: HYBRID (Embeddings + Rules + LLM)
# ============================================================================

class HybridMatchingEngine:
    """Three-component hybrid: Embeddings (35%) + Rules (25%) + LLM (40%)"""
    
    def __init__(self):
        self.semantic_scorer = SemanticScorer()
        self.rule_extractor = DeterministicExtractor()
        self.llm_scorer = LLMScorer()
        self.weights = {'semantic': 0.35, 'rule': 0.25, 'llm': 0.40}
    
    def score(self, job_desc: str, resume_text: str) -> Dict:
        # Semantic similarity
        semantic_score = self.semantic_scorer.score(job_desc, resume_text)
        
        # Rule-based scoring
        years_exp = self.rule_extractor.extract_years_of_experience(resume_text)
        skill_profile = self.rule_extractor.extract_skills(resume_text)
        domain_relevance = self.rule_extractor.calculate_domain_relevance(resume_text)
        
        rule_score, breakdown = self.rule_extractor.calculate_deterministic_score(
            years_exp, skill_profile, domain_relevance
        )
        
        # LLM scoring
        llm_result = self.llm_scorer.score(job_desc, resume_text)
        llm_score = llm_result['score']
        
        # Weighted combination
        final_score = (
            self.weights['semantic'] * semantic_score +
            self.weights['rule'] * rule_score +
            self.weights['llm'] * llm_score
        )
        
        return {
            'final_score': round(final_score, 3),
            'semantic_score': semantic_score,
            'rule_score': rule_score,
            'llm_score': llm_score,
            'llm_reasoning': llm_result.get('reasoning', ''),
            'years_experience': years_exp
        }

# ============================================================================
# 3. APPROACH 3: TWO-STAGE LLM PIPELINE
# ============================================================================

# Already implemented in TwoStageMatchingEngine

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_ground_truth() -> Dict[str, float]:
    """Load ground truth labels"""
    with open('data/ground_truth.json', 'r') as f:
        return json.load(f)

def load_job_description() -> str:
    """Load job description"""
    with open('data/job_descriptions/ema_ai_apps_engineer.txt', 'r') as f:
        return f.read()

def load_resumes() -> List[Dict[str, str]]:
    """Load all resumes"""
    resumes = []
    resume_dir = Path('data/resumes')
    
    for filepath in sorted(resume_dir.glob('*.txt')):
        resume_id = filepath.stem
        with open(filepath, 'r') as f:
            text = f.read()
        
        resumes.append({
            'id': resume_id,
            'text': text
        })
    
    return resumes

def calculate_metrics(predicted_scores: List[float], true_scores: List[float], k: int = 3) -> Dict:
    """Calculate comprehensive evaluation metrics"""
    
    # nDCG@K
    try:
        ndcg_at_k = ndcg_score([true_scores], [predicted_scores], k=k)
    except:
        ndcg_at_k = 0.0
    
    # Kendall Tau (ranking correlation)
    kendall_tau, kendall_p = kendalltau(predicted_scores, true_scores)
    
    # Spearman correlation
    spearman_r, spearman_p = spearmanr(predicted_scores, true_scores)
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(true_scores, predicted_scores)
    
    # Precision@K (top-K accuracy)
    # Get top-K indices from predictions
    pred_top_k_indices = np.argsort(predicted_scores)[-k:][::-1]
    true_top_k_indices = np.argsort(true_scores)[-k:][::-1]
    
    # Count how many predictions match ground truth in top-K
    precision_at_k = len(set(pred_top_k_indices) & set(true_top_k_indices)) / k
    
    return {
        'ndcg_at_3': round(ndcg_at_k, 4),
        'kendall_tau': round(kendall_tau, 4),
        'spearman_r': round(spearman_r, 4),
        'precision_at_3': round(precision_at_k, 4),
        'mae': round(mae, 4)
    }

def evaluate_approach(engine, name: str, job_desc: str, resumes: List[Dict], ground_truth: Dict) -> Dict:
    """Evaluate a single approach"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {name}")
    print(f"{'='*80}")
    
    results = []
    start_time = time.time()
    
    for i, resume in enumerate(resumes, 1):
        print(f"  [{i}/{len(resumes)}] Scoring {resume['id']}...", end='\r')
        
        try:
            result = engine.score(job_desc, resume['text'])
            results.append({
                'id': resume['id'],
                'score': result['final_score'],
                'details': result
            })
        except Exception as e:
            print(f"\n  ❌ Error scoring {resume['id']}: {e}")
            results.append({
                'id': resume['id'],
                'score': 0.0,
                'details': {'error': str(e)}
            })
    
    elapsed_time = time.time() - start_time
    
    # Extract scores for metrics
    predicted_scores = []
    true_scores = []
    
    for result in results:
        resume_id = result['id']
        predicted_scores.append(result['score'])
        true_scores.append(ground_truth.get(resume_id, 0.0))
    
    # Calculate metrics
    metrics = calculate_metrics(predicted_scores, true_scores)
    
    # Sort results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n\n✅ Complete!")
    print(f"   Time: {elapsed_time:.1f}s ({elapsed_time/len(resumes):.2f}s per resume)")
    
    return {
        'name': name,
        'results': results,
        'metrics': metrics,
        'execution_time': elapsed_time,
        'time_per_resume': elapsed_time / len(resumes)
    }

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    # Load data
    print("\n📁 Loading data...")
    job_desc = load_job_description()
    resumes = load_resumes()
    ground_truth = load_ground_truth()
    
    print(f"   ✅ Loaded {len(resumes)} resumes")
    print(f"   ✅ Loaded {len(ground_truth)} ground truth labels")
    
    # Initialize engines
    print("\n⚙️  Initializing engines...")
    
    approach1 = DeterministicMatchingEngine()
    approach2 = HybridMatchingEngine()
    approach3 = TwoStageMatchingEngine()
    
    print("   ✅ All engines initialized")
    
    # Evaluate each approach
    results = {}
    
    # Approach 1: Deterministic
    results['deterministic'] = evaluate_approach(
        approach1, 
        "Approach 1: Deterministic (Embeddings + Rules)",
        job_desc, 
        resumes, 
        ground_truth
    )
    
    # Approach 2: Hybrid
    results['hybrid'] = evaluate_approach(
        approach2,
        "Approach 2: Hybrid (Embeddings + Rules + LLM)",
        job_desc,
        resumes,
        ground_truth
    )
    
    # Approach 3: Two-Stage LLM
    print(f"\n{'='*80}")
    print(f"EVALUATING: Approach 3: Two-Stage LLM Pipeline")
    print(f"{'='*80}")
    
    start_time = time.time()
    two_stage_results = approach3.evaluate_batch(job_desc, resumes)
    elapsed_time = time.time() - start_time
    
    # Extract scores
    predicted_scores = []
    true_scores = []
    formatted_results = []
    
    for result in two_stage_results:
        resume_id = result['id']
        predicted_scores.append(result['final_score'])
        true_scores.append(ground_truth.get(resume_id, 0.0))
        formatted_results.append({
            'id': resume_id,
            'score': result['final_score'],
            'details': result
        })
    
    metrics = calculate_metrics(predicted_scores, true_scores)
    
    results['two_stage'] = {
        'name': "Approach 3: Two-Stage LLM Pipeline",
        'results': formatted_results,
        'metrics': metrics,
        'execution_time': elapsed_time,
        'time_per_resume': elapsed_time / len(resumes)
    }
    
    print(f"\n✅ Complete!")
    print(f"   Time: {elapsed_time:.1f}s ({elapsed_time/len(resumes):.2f}s per resume)")
    
    # ========================================================================
    # COMPARISON TABLE
    # ========================================================================
    
    print(f"\n\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    # Metrics comparison
    print("📊 METRICS COMPARISON:\n")
    print(f"{'Metric':<20} | {'Deterministic':>15} | {'Hybrid':>15} | {'Two-Stage LLM':>15} | {'Winner':>15}")
    print("-" * 93)
    
    metrics_to_compare = ['ndcg_at_3', 'precision_at_3', 'kendall_tau', 'spearman_r', 'mae']
    
    for metric in metrics_to_compare:
        det_val = results['deterministic']['metrics'][metric]
        hyb_val = results['hybrid']['metrics'][metric]
        two_val = results['two_stage']['metrics'][metric]
        
        # Determine winner (lower is better for MAE, higher for others)
        if metric == 'mae':
            winner = min([(det_val, 'Deterministic'), (hyb_val, 'Hybrid'), (two_val, 'Two-Stage')])[1]
        else:
            winner = max([(det_val, 'Deterministic'), (hyb_val, 'Hybrid'), (two_val, 'Two-Stage')])[1]
        
        print(f"{metric:<20} | {det_val:>15.4f} | {hyb_val:>15.4f} | {two_val:>15.4f} | {winner:>15}")
    
    # Performance comparison
    print(f"\n⏱️  PERFORMANCE COMPARISON:\n")
    print(f"{'Metric':<20} | {'Deterministic':>15} | {'Hybrid':>15} | {'Two-Stage LLM':>15}")
    print("-" * 78)
    
    det_time = results['deterministic']['time_per_resume']
    hyb_time = results['hybrid']['time_per_resume']
    two_time = results['two_stage']['time_per_resume']
    
    print(f"{'Time per resume':<20} | {det_time:>14.2f}s | {hyb_time:>14.2f}s | {two_time:>14.2f}s")
    
    # Cost estimation
    print(f"\n💰 ESTIMATED COST (per resume):\n")
    print(f"{'Approach':<30} | {'Cost':>15}")
    print("-" * 48)
    print(f"{'Deterministic (No LLM)':<30} | {'$0.0001':>15}")
    print(f"{'Hybrid (1 LLM call)':<30} | {'$0.005':>15}")
    print(f"{'Two-Stage (2 LLM calls)':<30} | {'$0.010':>15}")
    
    # Rankings comparison
    print(f"\n🏆 TOP 5 RANKINGS (by each approach):\n")
    
    for approach_key, approach_data in results.items():
        print(f"\n{approach_data['name']}:")
        top_5 = approach_data['results'][:5]
        for i, result in enumerate(top_5, 1):
            gt_score = ground_truth.get(result['id'], 0.0)
            print(f"  {i}. {result['id']:<20} Score: {result['score']:.3f}  (GT: {gt_score:.1f})")
    
    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'comparison': {
            'deterministic': {
                'metrics': results['deterministic']['metrics'],
                'time_per_resume': det_time,
                'estimated_cost': 0.0001,
                'rankings': [(r['id'], r['score']) for r in results['deterministic']['results'][:10]]
            },
            'hybrid': {
                'metrics': results['hybrid']['metrics'],
                'time_per_resume': hyb_time,
                'estimated_cost': 0.005,
                'rankings': [(r['id'], r['score']) for r in results['hybrid']['results'][:10]]
            },
            'two_stage': {
                'metrics': results['two_stage']['metrics'],
                'time_per_resume': two_time,
                'estimated_cost': 0.010,
                'rankings': [(r['id'], r['score']) for r in results['two_stage']['results'][:10]]
            }
        }
    }
    
    with open('results_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\n💾 Full results saved to: results_comparison.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
