import os
import json
import time
from datetime import datetime
from src.hybrid_engine import HybridEngine
from src.utils import calculate_metrics

def run_evaluation():
    print("🚀 Starting V3 Sequential Hybrid Evaluation...")
    engine = HybridEngine()
    
    # Load data
    with open("data/job_descriptions/ema_ai_apps_engineer.txt", "r") as f:
        jd = f.read()
        
    resumes_dir = "data/resumes"
    resumes = []
    for filename in os.listdir(resumes_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(resumes_dir, filename), "r") as f:
                resumes.append({
                    "id": filename.replace(".txt", ""),
                    "text": f.read()
                })
                
    # Evaluate
    start_time = time.time()
    results = engine.evaluate_batch(jd, resumes)
    duration = time.time() - start_time
    
    # Create run directory
    run_id = f"v3_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = f"runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save results
    with open(f"{run_dir}/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Calculate metrics
    with open("data/ground_truth.json", "r") as f:
        ground_truth = json.load(f)
        
    rankings = [r['candidate_id'] for r in results]
    metrics = calculate_metrics(rankings, ground_truth)
    
    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n✅ V3 Evaluation Complete!")
    print(f"⏱️  Duration: {duration:.2f}s")
    print(f"📊 nDCG@3: {metrics['ndcg_at_3']:.3f}")
    print(f"📊 Precision@1: {metrics['p_at_1']:.3f}")
    print(f"💰 Cost: Low (1 LLM call per resume vs 2 in V1)")
    print(f"📂 Results saved to: {run_dir}")

if __name__ == "__main__":
    run_evaluation()
