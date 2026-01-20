import os
import json
import logging
from pathlib import Path
from src.hybrid_scorer import HybridScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    jd_path = "data/job_descriptions/ema_ai_apps_engineer.txt"
    resumes_dir = "data/resumes/"
    
    with open(jd_path, "r") as f:
        jd = f.read()
        
    resumes = []
    for filename in sorted(os.listdir(resumes_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(resumes_dir, filename), "r") as f:
                resumes.append({
                    "id": filename.replace(".txt", ""),
                    "text": f.read()
                })
    return jd, resumes

def main():
    print("🚀 Starting Production Hybrid Resume Matcher...")
    print("   Combining LLM Intelligence + Deterministic Rules\n")
    
    # Initialize hybrid scorer (60% LLM, 40% Deterministic)
    scorer = HybridScorer(llm_weight=0.6, deterministic_weight=0.4)
    
    jd, resumes = load_data()
    print(f"📄 Loaded JD and {len(resumes)} resumes.\n")
    
    results = []
    for res in resumes:
        print(f"🔍 Evaluating: {res['id']}...")
        result = scorer.score(jd, res['text'])
        
        # Add ID for tracking
        result['id'] = res['id']
        results.append(result)
        
        # Show breakdown
        print(f"   LLM Score: {result['llm_component']['score']:.2f} (weight: {result['llm_component']['weight']})")
        print(f"   Deterministic: {result['deterministic_component']['score']:.2f} (weight: {result['deterministic_component']['weight']})")
        print(f"   → Final: {result['final_score']:.2f}\n")
    
    # Rank by final score
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Display rankings
    print("\n" + "="*80)
    print(f"{'RANK':<6} | {'CANDIDATE':<25} | {'FINAL':<8} | {'LLM':<8} | {'RULE':<8}")
    print("-" * 80)
    
    for i, res in enumerate(results, 1):
        print(f"{i:<6} | {res['id']:<25} | {res['final_score']:<8.2f} | "
              f"{res['llm_component']['score']:<8.2f} | "
              f"{res['deterministic_component']['score']:<8.2f}")
        
    print("="*80 + "\n")
    
    # Save detailed results
    with open("results_hybrid.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Full results saved to results_hybrid.json")
    
    # Show top candidate breakdown
    print("\n📊 Top Candidate Detailed Breakdown:\n")
    top = results[0]
    print(f"Candidate: {top['id']}")
    print(f"Final Score: {top['final_score']:.3f}\n")
    
    print("LLM Component:")
    print(f"  Score: {top['llm_component']['score']:.3f}")
    print(f"  Reasoning: {top['llm_component']['reasoning']}\n")
    
    print("Deterministic Component:")
    det = top['deterministic_component']
    print(f"  Score: {det['score']:.3f}")
    print(f"  Years Experience: {det['years_experience']}")
    print(f"  Skill Coverage: {det['skill_coverage']:.1%}")
    print(f"  AI Relevance: {det['ai_relevance']:.1%}")
    print(f"  Support Relevance: {det['support_relevance']:.1%}")
    print(f"  Matched Required: {', '.join(det['matched_required_skills'][:5])}")
    print(f"  Missing Required: {', '.join(det['missing_required_skills'])}")

if __name__ == "__main__":
    main()
