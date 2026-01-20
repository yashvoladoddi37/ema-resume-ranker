import os
import json
from src.engine import ResumeMatchingEngine

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
    print("🚀 Starting Resume Matching Engine Demo...")
    engine = ResumeMatchingEngine()
    
    jd, resumes = load_data()
    print(f"📄 Loaded JD and {len(resumes)} resumes.")
    
    # Process only top 3 for the quick demo to save tokens/time if needed
    # but for full verification we'll do all 10.
    print("🔍 Ranking candidates (this may take a few moments)...")
    results = engine.rank_resumes(jd, resumes)
    
    print("\n" + "="*50)
    print(f"{'RANK':<5} | {'CANDIDATE':<20} | {'SCORE':<8} | {'SEMANTIC'}")
    print("-" * 50)
    
    for i, res in enumerate(results, 1):
        print(f"{i:<5} | {res['id']:<20} | {res['final_score']:<8} | {res['semantic_score']}")
        
    print("="*50 + "\n")
    
    # Save results for analysis
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Full results saved to results.json")

if __name__ == "__main__":
    main()
