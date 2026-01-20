import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimal LLM scorer without embedding dependency
class MinimalLLMScorer:
    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
    def score(self, job_description: str, resume_text: str) -> dict:
        prompt = f"""You are a Principal AI Applications Engineer at Ema. Evaluate this candidate's Resume against the Job Description.

### Job Description:
{job_description}

### Candidate Resume:
{resume_text}

Return ONLY a JSON object with this schema:
{{
  "score": float (0.0-1.0),
  "reasoning": "2-3 sentence technical justification",
  "matched_skills": ["list", "of", "skills"],
  "missing_skills": ["critical", "gaps"]
}}"""

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {"score": 0.0, "reasoning": str(e), "matched_skills": [], "missing_skills": []}

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
    print("🚀 Starting Minimal Resume Matching Demo (LLM-Only)...")
    scorer = MinimalLLMScorer()
    
    jd, resumes = load_data()
    print(f"📄 Loaded JD and {len(resumes)} resumes.\n")
    
    results = []
    for res in resumes:
        print(f"🔍 Evaluating: {res['id']}...")
        llm_result = scorer.score(jd, res['text'])
        results.append({
            "id": res["id"],
            "score": llm_result["score"],
            "reasoning": llm_result["reasoning"],
            "matched_skills": llm_result["matched_skills"],
            "missing_skills": llm_result["missing_skills"]
        })
    
    # Rank by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print("\n" + "="*70)
    print(f"{'RANK':<6} | {'CANDIDATE':<25} | {'SCORE':<8}")
    print("-" * 70)
    
    for i, res in enumerate(results, 1):
        print(f"{i:<6} | {res['id']:<25} | {res['score']:<8.2f}")
        
    print("="*70 + "\n")
    
    # Save detailed results
    with open("results_llm_only.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Full results saved to results_llm_only.json")
    
    # Show top 3 reasoning
    print("\n📊 Top 3 Candidates - Detailed Reasoning:\n")
    for i, res in enumerate(results[:3], 1):
        print(f"{i}. {res['id']} (Score: {res['score']:.2f})")
        print(f"   Reasoning: {res['reasoning']}")
        print(f"   Matched: {', '.join(res['matched_skills'][:5])}")
        print()

if __name__ == "__main__":
    main()
