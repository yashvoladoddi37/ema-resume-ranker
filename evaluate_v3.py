import os
import json
import time
import logging
from datetime import datetime
from src.hybrid_engine import HybridEngine
from src.utils import calculate_metrics, load_resumes, load_job_description
from src.config import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

def run_evaluation():
    logger.info("🚀 Starting V3 Sequential Hybrid Evaluation...")
    engine = HybridEngine()
    
    # Load data
    try:
        jd = load_job_description()
        resumes = load_resumes()
        logger.info(f"Loaded {len(resumes)} resumes and job description.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
                
    # Evaluate
    start_time = time.time()
    try:
        results = engine.evaluate_batch(jd, resumes)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return
        
    duration = time.time() - start_time
    
    # Create run directory
    run_id = f"v3_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = f"runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save results
    with open(f"{run_dir}/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Calculate metrics
    try:
        with open("data/ground_truth.json", "r") as f:
            ground_truth = json.load(f)
            
        rankings = [r['candidate_id'] for r in results]
        metrics = calculate_metrics(rankings, ground_truth)
        
        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"\n✅ V3 Evaluation Complete!")
        logger.info(f"⏱️  Duration: {duration:.2f}s")
        logger.info(f"📊 nDCG@3: {metrics['ndcg_at_3']:.3f}")
        logger.info(f"📊 Precision@1: {metrics['p_at_1']:.3f}")
        logger.info(f"💰 Cost: Low (1 LLM call per resume vs 2 in V1)")
        logger.info(f"📂 Results saved to: {run_dir}")
        
    except FileNotFoundError:
        logger.warning("Ground truth file not found. Metrics skipped.")
    except Exception as e:
        logger.error(f"Metric calculation failed: {e}")

if __name__ == "__main__":
    run_evaluation()
