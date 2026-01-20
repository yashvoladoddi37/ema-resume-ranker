#!/usr/bin/env python3
"""
Hybrid Resume Matching Demo (LLM + Deterministic)
==================================================
Combines two scoring components for robust, auditable rankings:
  - LLM (60%): Nuanced contextual understanding
  - Deterministic (40%): Rule-based skill/experience extraction

This demonstrates that we don't rely purely on LLM - there's a verifiable,
reproducible deterministic baseline that anchors the scoring.
"""

import os
import json
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Component Weights
# ============================================================
WEIGHT_LLM = 0.60
WEIGHT_DETERMINISTIC = 0.40


# ============================================================
# 1. DETERMINISTIC SCORER (Rule-Based, Auditable)
# ============================================================
class DeterministicScorer:
    """
    Extracts verifiable metrics using regex patterns and keyword matching.
    This provides an auditable baseline independent of LLM.
    
    Unlike LLM, this component:
    - Is 100% reproducible (same input = same output, always)
    - Is auditable (you can trace exactly WHY a score was given)
    - Has no API costs or latency variance
    - Provides hard signals: years extracted, exact skill matches
    """
    
    REQUIRED_SKILLS = {
        'python', 'api', 'rest', 'json', 'troubleshooting',
        'production', 'technical support', 'saas'
    }
    
    PREFERRED_SKILLS = {
        'genai', 'llm', 'ml', 'langchain', 'prompt engineering',
        'observability', 'logging', 'dashboard', 'aws', 'gcp',
        'crm', 'ats', 'soap', 'integration'
    }
    
    AI_KEYWORDS = {
        'ai', 'artificial intelligence', 'machine learning', 'ml',
        'llm', 'large language model', 'genai', 'generative ai',
        'langchain', 'langgraph', 'prompt', 'rag', 'embedding'
    }
    
    SUPPORT_KEYWORDS = {
        'support', 'customer success', 'technical support',
        'troubleshooting', 'debugging', 'production issues',
        'incident', 'ticket', 'escalation', 'customer-facing'
    }
    
    def score(self, resume_text: str) -> Dict[str, Any]:
        """Returns deterministic score with full breakdown."""
        import re
        text_lower = resume_text.lower()
        
        # 1. Extract years of experience
        years_exp = 0.0
        
        # Heuristic A: Look for "X years" (the old way)
        year_matches = re.findall(r'(\d+)\+?\s*years?', text_lower)
        if year_matches:
            years_exp = max(float(m) for m in year_matches)
            
        # Heuristic B: Calculate from date ranges (e.g., 2024 - 2025)
        # This is more robust for actual resumes
        date_ranges = re.findall(r'(\d{4})\s*[-–]\s*(\d{4}|present)', text_lower)
        range_years = 0.0
        for start, end in date_ranges:
            end_year = 2026 if end == 'present' else int(end) # Using 2026 as current year per metadata
            range_years += (end_year - int(start))
            
        years_exp = max(years_exp, range_years)
        
        # 2. Skill matching
        matched_required = {s for s in self.REQUIRED_SKILLS if re.search(rf'\b{re.escape(s)}\b', text_lower)}
        matched_preferred = {s for s in self.PREFERRED_SKILLS if re.search(rf'\b{re.escape(s)}\b', text_lower)}
        missing_required = self.REQUIRED_SKILLS - matched_required
        
        required_coverage = len(matched_required) / len(self.REQUIRED_SKILLS)
        preferred_coverage = len(matched_preferred) / len(self.PREFERRED_SKILLS)
        skill_score = (0.7 * required_coverage) + (0.3 * preferred_coverage)
        
        # 3. Domain relevance (AI + Support keyword density)
        words = text_lower.split()
        total_words = max(len(words), 1)
        
        ai_count = sum(1 for w in words if any(kw in w for kw in self.AI_KEYWORDS))
        support_count = sum(1 for w in words if any(kw in w for kw in self.SUPPORT_KEYWORDS))
        
        ai_relevance = min(ai_count / total_words * 10, 1.0)
        support_relevance = min(support_count / total_words * 10, 1.0)
        
        # 4. Calculate final deterministic score
        exp_score = min(years_exp / 3.0, 1.0)  # 3+ years = full points
        final_score = (
            0.20 * exp_score +
            0.40 * skill_score +
            0.20 * ai_relevance +
            0.20 * support_relevance
        )
        
        return {
            'score': round(final_score, 3),
            'breakdown': {
                'experience_component': round(0.20 * exp_score, 3),
                'skill_component': round(0.40 * skill_score, 3),
                'ai_relevance_component': round(0.20 * ai_relevance, 3),
                'support_relevance_component': round(0.20 * support_relevance, 3)
            },
            'extracted_data': {
                'years_experience': years_exp,
                'matched_required': sorted(list(matched_required)),
                'matched_preferred': sorted(list(matched_preferred)),
                'missing_required': sorted(list(missing_required)),
                'required_coverage_pct': round(required_coverage * 100, 1),
                'preferred_coverage_pct': round(preferred_coverage * 100, 1)
            }
        }


# ============================================================
# 3. LLM SCORER (Contextual Intelligence)
# ============================================================
class LLMScorer:
    """
    Uses LLM for nuanced, context-aware evaluation.
    Temperature=0 for deterministic outputs across runs.
    """
    
    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
    def score(self, job_description: str, resume_text: str, deterministic_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate resume with optional deterministic 'ground truth' context.
        """
        context_str = ""
        if deterministic_context:
            ext = deterministic_context.get('extracted_data', {})
            context_str = f"""
### System Detected Facts (Ground Truth):
- Detected Experience: {ext.get('years_experience', 0)} years
- Required Skills Matched: {', '.join(ext.get('matched_required', [])) or 'None'}
- Required Skills Missing: {', '.join(ext.get('missing_required', [])) or 'None'}
- Skill Coverage: {ext.get('required_coverage_pct', 0)}%

NOTE: Use these facts as a baseline for your evaluation. Your goal is to provide technical depth beyond these simple matches.
"""

        prompt = f"""You are a Principal AI Applications Engineer at Ema. Evaluate this candidate's Resume against the Job Description.

### Instructions:
1. Analyze with Nuance: Look beyond keyword matching. Evaluate depth of experience, seniority, and domain relevance.
2. Score (0.0 - 1.0):
   - 0.8-1.0: Excellent fit. Meets nearly all requirements with specific relevant experience.
   - 0.5-0.7: Partial fit. Has significant experience but lacks some core skills.
   - 0.2-0.4: Weak fit. Some transferable skills but major gaps.
   - 0.0-0.1: No fit. Irrelevant stack or domain entirely.
3. Reasoning: Provide 2-3 sentence technical justification.

### Job Description:
{job_description}

{context_str}

### Candidate Resume:
{resume_text}

### Output Format:
Return ONLY a JSON object:
{{
  "score": float,
  "reasoning": "string",
  "matched_skills": ["list"],
  "missing_skills": ["list"]
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


# ============================================================
# HYBRID ENGINE (Combines Both)
# ============================================================
class HybridEngine:
    """
    Production-grade resume matching with transparent, auditable scoring.
    
    Architecture: Sequential Enrichment
    1. Deterministic extraction provides 'Ground Truth'
    2. LLM evaluates quality/nuance using Ground Truth as context
    3. Final score is a weighted combination
    """
    
    def __init__(self):
        self.llm_scorer = LLMScorer()
        self.deterministic_scorer = DeterministicScorer()
        
    def evaluate(self, job_description: str, resume: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate a single resume against the job description.
        Returns comprehensive scoring breakdown.
        """
        resume_text = resume['text']
        
        # 1. Deterministic (verifiable, auditable) - ALWAYS RUNS FIRST
        det_result = self.deterministic_scorer.score(resume_text)
        det_score = det_result['score']
        
        # 2. LLM (contextual intelligence) - Sequential Enrichment
        llm_result = self.llm_scorer.score(job_description, resume_text, deterministic_context=det_result)
        llm_score = llm_result['score']
        
        # Weighted combination
        final_score = (WEIGHT_LLM * llm_score) + (WEIGHT_DETERMINISTIC * det_score)
        
        return {
            'id': resume['id'],
            'final_score': round(final_score, 4),
            'score_formula': f"({WEIGHT_LLM} × {llm_score:.2f}) + ({WEIGHT_DETERMINISTIC} × {det_score:.3f}) = {final_score:.4f}",
            'components': {
                'llm': {
                    'score': llm_score,
                    'weight': WEIGHT_LLM,
                    'weighted_contribution': round(WEIGHT_LLM * llm_score, 4),
                    'reasoning': llm_result['reasoning'],
                    'matched_skills': llm_result['matched_skills'],
                    'missing_skills': llm_result['missing_skills']
                },
                'deterministic': {
                    'score': det_score,
                    'weight': WEIGHT_DETERMINISTIC,
                    'weighted_contribution': round(WEIGHT_DETERMINISTIC * det_score, 4),
                    **det_result['breakdown'],
                    **det_result['extracted_data']
                }
            }
        }
    
    def rank_all(self, job_description: str, resumes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Evaluate and rank all resumes."""
        results = []
        for i, resume in enumerate(resumes, 1):
            logger.info(f"[{i}/{len(resumes)}] Evaluating: {resume['id']}")
            result = self.evaluate(job_description, resume)
            results.append(result)
        
        return sorted(results, key=lambda x: x['final_score'], reverse=True)


# ============================================================
# DATA LOADING
# ============================================================
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


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*80)
    print("🚀 HYBRID RESUME MATCHING ENGINE (LLM + Deterministic)")
    print(f"   Formula: Final = ({int(WEIGHT_LLM*100)}% × LLM) + ({int(WEIGHT_DETERMINISTIC*100)}% × Deterministic)")
    print("="*80 + "\n")
    
    # Initialize engine
    engine = HybridEngine()
    
    # Load data
    jd, resumes = load_data()
    print(f"📄 Loaded JD and {len(resumes)} resumes.\n")
    
    # Rank all resumes
    results = engine.rank_all(jd, resumes)
    
    # Display rankings table
    print("\n" + "="*90)
    print(f"{'RANK':<5} | {'CANDIDATE':<25} | {'FINAL':<8} | {'LLM':<6} | {'DETERM':<7} | {'YRS EXP':<7}")
    print("-" * 90)
    
    for i, res in enumerate(results, 1):
        c = res['components']
        yrs = c['deterministic']['years_experience']
        print(f"{i:<5} | {res['id']:<25} | {res['final_score']:<8.4f} | "
              f"{c['llm']['score']:<6.2f} | {c['deterministic']['score']:<7.3f} | {yrs:<7.0f}")
    
    print("="*90)
    
    # Save detailed results
    output_path = "results_hybrid.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Full results saved to {output_path}")
    
    # Show detailed breakdown for top 3
    print("\n" + "="*80)
    print("📊 TOP 3 CANDIDATES - DETAILED BREAKDOWN")
    print("="*80)
    
    for i, res in enumerate(results[:3], 1):
        c = res['components']
        det = c['deterministic']
        llm = c['llm']
        
        print(f"\n#{i}: {res['id']}")
        print(f"    Final Score: {res['final_score']:.4f}")
        print(f"    {res['score_formula']}")
        print("-" * 60)
        
        # LLM Component
        print(f"  📝 LLM Analysis (weight: {WEIGHT_LLM}):")
        print(f"     Score: {llm['score']:.2f}")
        print(f"     Reasoning: {llm['reasoning']}")
        print(f"     Matched: {', '.join(llm['matched_skills'][:5])}")
        
        # Deterministic Component
        print(f"\n  🔧 Deterministic Analysis (weight: {WEIGHT_DETERMINISTIC}):")
        print(f"     Score: {det['score']:.3f}")
        print(f"     Years Experience: {det['years_experience']:.0f}")
        print(f"     Required Skills: {det['required_coverage_pct']}% ({len(det['matched_required'])}/8 matched)")
        print(f"     Matched Required: {', '.join(det['matched_required']) or 'None'}")
        print(f"     Missing Required: {', '.join(det['missing_required']) or 'None'}")
        print(f"     Preferred Skills: {det['preferred_coverage_pct']}% matched")
    
    print("\n" + "="*80)
    print("💡 KEY INSIGHT: The deterministic component ensures verifiable signals")
    print("   (years extracted, skill coverage) anchor the LLM's contextual assessment.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
