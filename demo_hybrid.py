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
import time
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from sklearn.metrics import ndcg_score

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
    
    RELEVANT_DEGREES = {
        'computer science', 'software engineering', 'information technology',
        'computer engineering', 'electrical engineering', 'data science'
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
        
        # 4. Degree Relevance (Strictly by degree, not institution)
        has_relevant_degree = any(re.search(rf'\b{re.escape(d)}\b', text_lower) for d in self.RELEVANT_DEGREES)
        edu_score = 1.0 if has_relevant_degree else 0.0

        # 5. Calculate final deterministic score
        # Updated weighting based on JD analysis (totals 100%):
        # - Technical Skills: 35% (APIs, Python - foundation)
        # - AI Relevance: 35% (GenAI/LLM - core differentiator)
        # - Experience: 15% (3+ years preferred)
        # - Support Relevance: 5% (preferred, not required)
        # - Education: 10% (relevant degree)
        exp_score = min(years_exp / 3.0, 1.0)  # 3+ years = full points
        final_score = (
            0.15 * exp_score +
            0.35 * skill_score +
            0.35 * ai_relevance +
            0.05 * support_relevance +
            0.10 * edu_score
        )
        
        return {
            'score': round(final_score, 3),
            'breakdown': {
                'experience_component': round(0.15 * exp_score, 3),
                'skill_component': round(0.35 * skill_score, 3),
                'ai_relevance_component': round(0.35 * ai_relevance, 3),
                'support_relevance_component': round(0.05 * support_relevance, 3),
                'education_component': round(0.10 * edu_score, 3)
            },
            'extracted_data': {
                'years_experience': years_exp,
                'has_relevant_degree': has_relevant_degree,
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
        self.primary_key = os.getenv("GROQ_API_KEY")
        self.fallback_key = os.getenv("GROQ_API_KEY_2")
        self.client = Groq(api_key=self.primary_key)
        self.using_fallback = False
        
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
2. Degree over Institution: EVALUATE BY DEGREE RELEVANCE ONLY (e.g., Computer Science, Engineering). IGNORE INSTITUTION PRESTIGE OR RANKINGS. Do not bias scores toward Ivy League or equivalent status colleges.
3. Score (0.0 - 1.0):
   - 0.8-1.0: Excellent fit. Meets nearly all requirements with specific relevant experience.
   - 0.5-0.7: Partial fit. Has significant experience but lacks some core skills.
   - 0.2-0.4: Weak fit. Some transferable skills but major gaps.
   - 0.0-0.1: No fit. Irrelevant stack or domain entirely.
4. Detailed Reasoning: Provide a technical justification that includes a breakdown of:
   - Skill Alignment: How well the core tech stack matches.
   - Experience Depth: Evaluation of the complexity and impact of previous roles.
   - Domain Fit: Relevance to AI support, SaaS, and GenAI workflows.

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
  "technical_breakdown": {{
    "skill_alignment": float,
    "experience_depth": float,
    "domain_fit": float
  }},
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
            error_str = str(e)
            # Check if it's a rate limit error and we have a fallback key
            if "rate_limit" in error_str.lower() or "429" in error_str:
                if self.fallback_key and not self.using_fallback:
                    logger.warning(f"Primary API key hit rate limit. Switching to fallback key...")
                    from groq import Groq
                    self.client = Groq(api_key=self.fallback_key)
                    self.using_fallback = True
                    
                    # Retry with fallback key
                    try:
                        completion = self.client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            response_format={"type": "json_object"}
                        )
                        logger.info("✅ Successfully used fallback API key")
                        return json.loads(completion.choices[0].message.content)
                    except Exception as fallback_error:
                        logger.error(f"Fallback API key also failed: {fallback_error}")
                        return {"score": 0.0, "reasoning": f"Both API keys failed. Primary: {error_str}, Fallback: {str(fallback_error)}", "matched_skills": [], "missing_skills": [], "technical_breakdown": {}}
                else:
                    logger.error(f"Rate limit hit but no fallback key available or already using fallback")
            
            logger.error(f"LLM Error: {e}")
            return {"score": 0.0, "reasoning": str(e), "matched_skills": [], "missing_skills": [], "technical_breakdown": {}}


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
                    'technical_breakdown': llm_result.get('technical_breakdown', {}),
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
            if i > 1:
                time.sleep(2)  # Rate limit prevention
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



def run_formal_evaluation(results: List[Dict[str, Any]], ground_truth_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compare engine results vs ground truth and return metrics."""
    gt_lookup = {}
    for item in ground_truth_data["resumes"]:
        base_id = item["id"]
        gt_lookup[base_id] = item["label"]
    
    comparison = []
    for res in results:
        res_id = res['id']
        # Extract base ID (e.g., res_001_yashpreet -> res_001)
        base_id = '_'.join(res_id.split('_')[:2])
        gt_label = gt_lookup.get(base_id, None)
        if gt_label is not None:
            comparison.append({
                'id': res_id,
                'predicted': res['final_score'],
                'ground_truth': gt_label
            })
    
    metrics = {}
    if len(comparison) > 0:
        comparison_ranked = sorted(comparison, key=lambda x: x['predicted'], reverse=True)
        predicted_scores = [c['predicted'] for c in comparison_ranked]
        ground_truth = [c['ground_truth'] for c in comparison_ranked]
        
        # 1. nDCG@3 (Ranking Quality)
        metrics['ndcg3'] = float(ndcg_score([ground_truth], [predicted_scores], k=3))

        # 2. Precision@1 (Is top result a "Good" match?)
        metrics['precision1'] = 1.0 if ground_truth[0] == 1.0 else 0.0
        
        # 3. Recall@3 (Are all "Good" matches in top 3?)
        good_matches_total = sum(1 for gt in ground_truth if gt == 1.0)
        good_matches_in_top3 = sum(1 for gt in ground_truth[:3] if gt == 1.0)
        metrics['recall3'] = good_matches_in_top3 / good_matches_total if good_matches_total > 0 else 0.0
        
        # 4. Pairwise Accuracy (Correct ordering of pairs)
        correct_pairs = 0
        total_pairs = 0
        for i in range(len(comparison_ranked)):
            for j in range(i + 1, len(comparison_ranked)):
                gt_i, gt_j = comparison_ranked[i]['ground_truth'], comparison_ranked[j]['ground_truth']
                if gt_i != gt_j:  # Only count pairs with different labels
                    total_pairs += 1
                    # Since i < j and sorted by predicted desc, we expect gt_i >= gt_j
                    if gt_i >= gt_j: # This condition is correct for a descending sort
                        correct_pairs += 1
        metrics['pairwise_acc'] = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        
        # 5. Tier Separation (Mean score per tier)
        good_scores = [c['predicted'] for c in comparison if c['ground_truth'] == 1.0]
        partial_scores = [c['predicted'] for c in comparison if c['ground_truth'] == 0.5]
        poor_scores = [c['predicted'] for c in comparison if c['ground_truth'] == 0.0]
        
        mean_good = sum(good_scores) / len(good_scores) if good_scores else 0
        mean_partial = sum(partial_scores) / len(partial_scores) if partial_scores else 0
        mean_poor = sum(poor_scores) / len(poor_scores) if poor_scores else 0
        
        metrics['mean_good_score'] = mean_good
        metrics['mean_partial_score'] = mean_partial
        metrics['mean_poor_score'] = mean_poor
        metrics['tier_separation'] = mean_good > mean_partial > mean_poor
        
        return metrics, comparison_ranked, comparison
    else:
        return metrics, [], []

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
        det = c['deterministic']
        yrs = det.get('years_experience', 0) # Use .get() for safety
        print(f"{i:<5} | {res['id']:<25} | {res['final_score']:<8.4f} | "
              f"{c['llm']['score']:<6.2f} | {c['deterministic']['score']:<7.3f} | {yrs:<7.0f}")
    
    print("="*90)
    
    # Save detailed results
    output_path = "results_hybrid.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Full results saved to {output_path}")
    
    # ============================================================
    # FORMAL EVALUATION: Compare Engine vs Ground Truth
    # ============================================================
    print("\n" + "="*80)
    print("📊 FORMAL EVALUATION: Engine vs Ground Truth")
    print("="*80)
    
    # Load ground truth labels
    with open("data/labeled_dataset.json", "r") as f:
        labeled_data = json.load(f)
    
    metrics, comparison_ranked, comparison = run_formal_evaluation(results, labeled_data)

    if len(comparison) > 0:
        # Print Results
        print(f"\n{'Metric':<25} | {'Value':<10} | {'Target':<10} | {'Status'}")
        print("-" * 65)
        print(f"{'nDCG@3':<25} | {metrics['ndcg3']:<10.3f} | {'≥0.85':<10} | {'✅' if metrics['ndcg3'] >= 0.85 else '❌'}")
        print(f"{'Precision@1':<25} | {metrics['precision1']:<10.0%} | {'100%':<10} | {'✅' if metrics['precision1'] == 1.0 else '❌'}")
        print(f"{'Recall@3 (Good matches)':<25} | {metrics['recall3']:<10.0%} | {'100%':<10} | {'✅' if metrics['recall3'] == 1.0 else '❌'}")
        print(f"{'Pairwise Accuracy':<25} | {metrics['pairwise_acc']:<10.1%} | {'≥85%':<10} | {'✅' if metrics['pairwise_acc'] >= 0.85 else '❌'}")
        print(f"{'Tier Separation':<25} | {'Yes' if metrics['tier_separation'] else 'No':<10} | {'Yes':<10} | {'✅' if metrics['tier_separation'] else '❌'}")
        
        print(f"\n📈 Tier Mean Scores:")
        print(f"   Good (1.0):    {metrics['mean_good_score']:.3f}  ({sum(1 for c in comparison if c['ground_truth'] == 1.0)} candidates)")
        print(f"   Partial (0.5): {metrics['mean_partial_score']:.3f}  ({sum(1 for c in comparison if c['ground_truth'] == 0.5)} candidates)")
        print(f"   Poor (0.0):    {metrics['mean_poor_score']:.3f}  ({sum(1 for c in comparison if c['ground_truth'] == 0.0)} candidates)")
        
        print("\n📋 Full Comparison Table:")
        print(f"{'Rank':<5} | {'Candidate':<25} | {'Predicted':<10} | {'Ground Truth':<12} | {'Match?'}")
        print("-" * 70)
        for rank, c in enumerate(comparison_ranked, 1):
            gt_label = "Good" if c['ground_truth'] == 1.0 else "Partial" if c['ground_truth'] == 0.5 else "Poor"
            # Check if ranking is reasonable
            match = "✅" if (c['predicted'] >= 0.6 and c['ground_truth'] >= 0.5) or \
                           (c['predicted'] < 0.6 and c['ground_truth'] < 0.5) or \
                           (c['predicted'] < 0.35 and c['ground_truth'] == 0.0) else "⚠️"
            print(f"{rank:<5} | {c['id']:<25} | {c['predicted']:<10.4f} | {gt_label:<12} | {match}")
    else:
        print("⚠️ Could not match results to ground truth labels.")
    
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
    print("   (years extracted, skill coverage, degree relevance) anchor the LLM's assessment.")
    print("   Zero weight is given to institution prestige to ensure fair, technical evaluation.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
