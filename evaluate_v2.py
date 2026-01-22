import os
import json
import time
from datetime import datetime
from src.deterministic_engine import DeterministicEngine
from src.utils import calculate_metrics, load_resumes, load_job_description

def run_evaluation():
    print("🎯 Starting V2 Deterministic Evaluation...")
    engine = DeterministicEngine()
    
    # Load data
    jd = load_job_description()
    resumes = load_resumes()
                
    # Evaluate
    start_time = time.time()
    results = engine.evaluate_batch(jd, resumes)
    duration = time.time() - start_time
    
    # Create run directory
    run_id = f"v2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = f"runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save results
    with open(f"{run_dir}/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Calculate metrics (using same ground truth)
    with open("data/ground_truth.json", "r") as f:
        ground_truth = json.load(f)
        
    # Prepare rankings for metrics
    # Format: list of candidate IDs in ranked order
    rankings = [r['candidate_id'] for r in results]
    
    metrics = calculate_metrics(rankings, ground_truth)
    
    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n✅ V2 Evaluation Complete!")
    print(f"⏱️  Duration: {duration:.2f}s")
    print(f"📊 nDCG@3: {metrics['ndcg_at_3']:.3f}")
    print(f"📊 Precision@1: {metrics['p_at_1']:.3f}")
    print(f"📂 Results saved to: {run_dir}")

if __name__ == "__main__":
    run_evaluation()
