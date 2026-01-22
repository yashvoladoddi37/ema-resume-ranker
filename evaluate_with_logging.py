"""
Evaluation script with full audit trail logging.
Saves all LLM outputs and intermediate JSON to a timestamped folder.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import time
from src.utils import (
    load_job_description,
    load_resumes,
    load_ground_truth,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    print_ranking,
    print_metrics
)


class AuditedMatchingEngine:
    """
    Matching engine with full audit trail.
    Saves every LLM input/output to a timestamped folder.
    """
    
    def __init__(self, run_folder: Path, api_key: str = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        self.run_folder = run_folder
        
        # Create subfolders
        (run_folder / "01_raw_resumes").mkdir(parents=True, exist_ok=True)
        (run_folder / "02_parser_prompts").mkdir(exist_ok=True)
        (run_folder / "03_parsed_data").mkdir(exist_ok=True)
        (run_folder / "04_scorer_prompts").mkdir(exist_ok=True)
        (run_folder / "05_scorer_outputs").mkdir(exist_ok=True)
        (run_folder / "06_final_results").mkdir(exist_ok=True)
        
        # Priority: Skills > Domain Fit (AI focus) > Experience
        # Support experience is preferred, not required
        self.weights = {
            'skill_match': 0.55,       # Most important: Python, AI/GenAI, APIs
            'domain_fit': 0.30,        # AI/GenAI domain alignment
            'experience_depth': 0.15   # Years matter less than skills
        }
    
    def _save_json(self, folder: str, filename: str, data: dict):
        """Save JSON to the run folder."""
        path = self.run_folder / folder / f"{filename}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path
    
    def _save_text(self, folder: str, filename: str, text: str):
        """Save text to the run folder."""
        path = self.run_folder / folder / f"{filename}.txt"
        with open(path, 'w') as f:
            f.write(text)
        return path
    
    def _call_llm(self, prompt: str, stage: str, resume_id: str) -> dict:
        """Call LLM and log both input and output."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=2048
        )
        
        raw_output = response.choices[0].message.content
        parsed_output = json.loads(raw_output)
        
        return {
            "prompt": prompt,
            "raw_response": raw_output,
            "parsed_response": parsed_output
        }
    
    def parse_resume(self, resume_text: str, resume_id: str) -> dict:
        """Stage 1: Parse resume with full logging."""
        
        # Save raw resume
        self._save_text("01_raw_resumes", resume_id, resume_text)
        
        # Build prompt
        prompt = f"""You are a resume parser. Extract structured information from the following resume.

RESUME TEXT:
{resume_text}

TASK:
Extract and return ONLY a valid JSON object with this EXACT structure:

{{
  "candidate_name": "Full Name",
  "skills": ["skill1", "skill2", "skill3"],
  "total_years_experience": 0.0,
  "experience": [
    {{
      "title": "Job Title",
      "company": "Company Name",
      "duration_years": 0.0,
      "start_date": "Month Year",
      "end_date": "Month Year or Present",
      "key_responsibilities": ["responsibility1", "responsibility2"]
    }}
  ],
  "education": {{
    "degree": "Degree Name",
    "field": "Field of Study",
    "institution": "University Name"
  }},
  "certifications": ["cert1", "cert2"],
  "domains": ["domain1", "domain2"]
}}

CRITICAL INSTRUCTIONS:
1. **Date Handling:**
   - If end_date is "Present", "Current", or missing, use "January 2025"
   - Calculate duration_years accurately (e.g., "Jan 2023 - Jan 2025" = 2.0 years)

2. **Skills Extraction:**
   - Extract ALL technical skills (languages, frameworks, tools, platforms)
   - Include synonyms where applicable

3. **Total Experience:**
   - Sum all duration_years from experience array
   - Do NOT count overlapping periods twice

4. **Domains:**
   - Infer from experience (e.g., "built chatbot" → "Conversational AI")
   - Common domains: "AI/ML", "Customer Support", "SaaS", "Cloud Infrastructure"

5. **Output Format:**
   - Return ONLY valid JSON
   - NO markdown, NO backticks

Begin parsing now:"""
        
        # Save prompt
        self._save_text("02_parser_prompts", resume_id, prompt)
        
        # Call LLM
        result = self._call_llm(prompt, "parser", resume_id)
        
        # Save parsed output
        self._save_json("03_parsed_data", resume_id, {
            "llm_raw_response": result["raw_response"],
            "parsed_data": result["parsed_response"]
        })
        
        return result["parsed_response"]
    
    def score_resume(self, job_description: str, parsed_resume: dict, resume_id: str) -> dict:
        """Stage 2: Score resume with full logging."""
        
        # Extract skills for explicit cross-reference
        candidate_skills = parsed_resume.get("skills", [])
        skills_list = ", ".join(candidate_skills) if candidate_skills else "None listed"
        
        prompt = f"""You are evaluating a candidate for a job opening.

JOB DESCRIPTION:
{job_description}

CANDIDATE PROFILE (extracted from resume):
{json.dumps(parsed_resume, indent=2)}

TASK:
Evaluate the candidate on three dimensions and return ONLY a JSON object.

**CRITICAL: SKILL MATCHING INSTRUCTIONS**

The candidate has these skills: [{skills_list}]

For skill_match, you MUST:
1. Go through EACH skill in the candidate's list above
2. For EACH skill, determine if it satisfies ANY JD requirement (even if wording differs)
3. Use SEMANTIC matching:
   - "Prometheus" satisfies "logging tools" and "alerting tools"
   - "LangChain" satisfies "GenAI workflows"
   - "REST" satisfies "APIs (JSON, REST, SOAP)"
4. matched_skills = skills FROM THE CANDIDATE'S LIST that match JD requirements
5. missing_skills = skills FROM THE JD that the candidate does NOT have

**CRITICAL: PER-EXPERIENCE RELEVANCE SCORING**

For experience_depth, you MUST:
1. Evaluate EACH experience block INDIVIDUALLY
2. For EACH, assign a relevance_score (0.0-1.0) based on:
   - Does the role involve AI/GenAI, SaaS, technical support, or customer success?
   - Does it involve Python, APIs, troubleshooting, or integrations?
3. Calculate relevant_years = SUM of (duration_years * relevance_score)
4. Base the final score on RELEVANT YEARS, not total years
5. Career switchers with 10y total but 1y relevant should score LOW

**CRITICAL: REQUIRED VS PREFERRED QUALIFICATIONS**

If the JD has "Required" and "Preferred" sections:
- Missing REQUIRED skills: heavy penalty (each missing = -0.15 from skill_match)
- Missing PREFERRED skills: light penalty (each missing = -0.05)
- Weight required skills 2x higher than preferred in final calculation

**CRITICAL: FORMULA AND REASONING**

For EACH dimension, include a "formula" key showing exact calculation.

OUTPUT FORMAT:

{{
  "skill_match": {{
    "score": 0.0,
    "reasoning": "Brief explanation",
    "formula": "required_matches/total_required * 0.7 + preferred_matches/total_preferred * 0.3 = X",
    "matched_skills": ["skill1", "skill2"],
    "missing_skills": ["skill3"]
  }},
  "experience_depth": {{
    "score": 0.0,
    "reasoning": "Brief explanation with relevant_years vs total_years",
    "formula": "relevant_years = (role1_years * rel1) + (role2_years * rel2) = X. Score = min(1.0, X/3)",
    "total_years": 0.0,
    "relevant_years": 0.0,
    "experience_breakdown": [
      {{"role": "Title", "years": 0.0, "relevance": 0.0, "relevance_reasoning": "Why"}}
    ]
  }},
  "domain_fit": {{
    "score": 0.0,
    "reasoning": "Brief explanation",
    "formula": "domain_matches/total_required_domains = X"
  }},
  "overall_assessment": "1-2 sentence summary"
}}

SCORING GUIDELINES:

1. **skill_match (0.0-1.0):**
   - 1.0 = ALL required + most preferred
   - 0.8 = ALL required, some preferred
   - 0.6 = MOST required
   - 0.4 = SOME required
   - 0.0 = NO required skills

2. **experience_depth (0.0-1.0):**
   - Based on RELEVANT YEARS ONLY
   - JD requires "3+ years" with 3+ relevant: ≥ 0.7
   - Career switcher (10y total, 1y relevant): ≤ 0.4

3. **domain_fit (0.0-1.0):**
   - AI/ML + customer-facing = high
   - Related domains = medium
   - Unrelated = low

Begin evaluation now:"""
        
        # Save prompt
        self._save_text("04_scorer_prompts", resume_id, prompt)
        
        # Call LLM
        result = self._call_llm(prompt, "scorer", resume_id)
        
        # Save scorer output
        self._save_json("05_scorer_outputs", resume_id, {
            "llm_raw_response": result["raw_response"],
            "dimension_scores": result["parsed_response"]
        })
        
        return result["parsed_response"]
    
    def evaluate(self, job_description: str, resume_text: str, resume_id: str) -> dict:
        """Full evaluation pipeline with logging."""
        
        start_time = time.time()
        print(f"📄 Evaluating {resume_id}...")
        
        # Stage 1
        print(f"  ⚙️  Stage 1: Parsing...")
        parsed_data = self.parse_resume(resume_text, resume_id)
        time.sleep(0.5)
        
        # Stage 2
        print(f"  ⚙️  Stage 2: Scoring...")
        dimension_scores = self.score_resume(job_description, parsed_data, resume_id)
        
        # Stage 3: Aggregate
        print(f"  ⚙️  Stage 3: Aggregating...")
        final_score = sum(
            dimension_scores.get(dim, {}).get("score", 0.5) * weight
            for dim, weight in self.weights.items()
        )
        final_score = max(0.0, min(1.0, final_score))
        
        processing_time = time.time() - start_time
        print(f"  ✅ Score: {final_score:.3f} ({processing_time:.1f}s)")
        
        result = {
            "id": resume_id,
            "candidate_name": parsed_data.get("candidate_name", "Unknown"),
            "final_score": round(final_score, 3),
            "parsed_data": parsed_data,
            "dimension_scores": dimension_scores,
            "weights": self.weights.copy(),
            "matched_skills": dimension_scores.get("skill_match", {}).get("matched_skills", []),
            "missing_skills": dimension_scores.get("skill_match", {}).get("missing_skills", []),
            "processing_time_seconds": round(processing_time, 2)
        }
        
        # Save final result
        self._save_json("06_final_results", resume_id, result)
        
        return result
    
    def evaluate_batch(self, job_description: str, resumes: list) -> list:
        """Evaluate all resumes with full logging."""
        
        results = []
        total = len(resumes)
        
        # Save job description
        self._save_text(".", "job_description", job_description)
        
        print(f"\n🚀 Starting audited evaluation of {total} resumes...")
        print(f"📁 Logs saved to: {self.run_folder}\n")
        
        for i, resume in enumerate(resumes, 1):
            print(f"[{i}/{total}] ", end="")
            result = self.evaluate(job_description, resume["text"], resume["id"])
            results.append(result)
            
            if i < total:
                time.sleep(1)
        
        # Sort by score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Save combined results
        self._save_json(".", "all_results", results)
        
        print(f"\n✅ Complete! Logs in: {self.run_folder}")
        return results


def main():
    """Run evaluation with full audit trail."""
    
    load_dotenv()
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(f"runs/run_{timestamp}")
    run_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Creating audit trail in: {run_folder}")
    
    # Initialize engine
    engine = AuditedMatchingEngine(run_folder)
    
    # Load data
    print("📥 Loading data...")
    job_description = load_job_description()
    resumes = load_resumes()
    ground_truth = load_ground_truth()
    
    # Evaluate
    results = engine.evaluate_batch(job_description, resumes)
    
    # Calculate metrics
    predicted = [r["final_score"] for r in results if r["id"] in ground_truth]
    actual = [ground_truth[r["id"]] for r in results if r["id"] in ground_truth]
    
    ndcg = calculate_ndcg_at_k(predicted, actual, k=3)
    prec = calculate_precision_at_k(predicted, actual, k=1)
    rec = calculate_recall_at_k(predicted, actual, k=3)
    
    # Print metrics
    print_metrics(ndcg, prec, rec)
    print_ranking(results)
    
    # Save metrics
    metrics = {
        "nDCG@3": round(ndcg, 3),
        "Precision@1": round(prec, 3),
        "Recall@3": round(rec, 3),
        "weights": engine.weights,
        "run_folder": str(run_folder),
        "timestamp": timestamp
    }
    
    with open(run_folder / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ All outputs saved to: {run_folder}")
    print(f"📂 Folder structure:")
    print(f"   ├── 01_raw_resumes/     # Original resume text")
    print(f"   ├── 02_parser_prompts/  # LLM prompts for parsing")
    print(f"   ├── 03_parsed_data/     # Parser LLM outputs")
    print(f"   ├── 04_scorer_prompts/  # LLM prompts for scoring")
    print(f"   ├── 05_scorer_outputs/  # Scorer LLM outputs")
    print(f"   ├── 06_final_results/   # Per-candidate final results")
    print(f"   ├── all_results.json    # Combined ranked results")
    print(f"   └── metrics.json        # Evaluation metrics")


if __name__ == "__main__":
    main()
