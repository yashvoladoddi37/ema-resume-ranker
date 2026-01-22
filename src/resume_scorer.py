"""
Resume Scorer - Stage 2 of Two-Stage Pipeline
Scores parsed resume against job description on multiple dimensions.
"""

from groq import Groq
import json
import os
from typing import Dict, Any, List
from dotenv import load_dotenv


class ResumeScorer:
    """
    Scores parsed resume data against job description.
    
    Dimensions:
    - skill_match: How well skills align (0.0-1.0)
    - experience_depth: Years + relevance (0.0-1.0)
    - domain_fit: Background alignment (0.0-1.0)
    """
    
    def __init__(self, api_key: str = None):
        """Initialize scorer with Groq API."""
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        self.temperature = float(os.getenv("TEMPERATURE", 0))
    
    def score(
        self, 
        job_description: str, 
        parsed_resume: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score resume on multiple dimensions.
        
        Args:
            job_description: The job posting text
            parsed_resume: Output from ResumeParser.parse()
            
        Returns:
            {
                "skill_match": {
                    "score": 0.0-1.0,
                    "reasoning": "explanation",
                    "matched_skills": ["skill1", "skill2"],
                    "missing_skills": ["skill3", "skill4"]
                },
                "experience_depth": {
                    "score": 0.0-1.0,
                    "reasoning": "explanation"
                },
                "domain_fit": {
                    "score": 0.0-1.0,
                    "reasoning": "explanation"
                },
                "overall_assessment": "summary"
            }
        """
        
        prompt = self._build_scoring_prompt(job_description, parsed_resume)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=2048
            )
            
            scores = json.loads(response.choices[0].message.content)
            return self._validate_scores(scores)
            
        except Exception as e:
            print(f"❌ Scoring failed: {e}")
            return self._get_default_scores()
    
    def _build_scoring_prompt(
        self, 
        jd: str, 
        parsed: Dict
    ) -> str:
        """Build the LLM prompt for scoring."""
        
        # Extract skills list for explicit cross-reference instruction
        candidate_skills = parsed.get("skills", [])
        skills_list = ", ".join(candidate_skills) if candidate_skills else "None listed"
        
        return f"""You are evaluating a candidate for a job opening.

JOB DESCRIPTION:
{jd}

CANDIDATE PROFILE (extracted from resume):
{json.dumps(parsed, indent=2)}

TASK:
Evaluate the candidate on three dimensions and return ONLY a JSON object.

**CRITICAL: SKILL MATCHING INSTRUCTIONS**

The candidate has these skills: [{skills_list}]

For skill_match, you MUST:
1. Go through EACH skill in the candidate's list above
2. For EACH skill, determine if it satisfies ANY JD requirement (even if wording differs)
3. Use SEMANTIC matching, not exact string matching:
   - "Prometheus" satisfies "logging tools" and "alerting tools"
   - "Kibana" satisfies "dashboard creation"
   - "Splunk" satisfies "logging tools"
   - "LangChain" satisfies "GenAI workflows"
   - "REST" satisfies "APIs (JSON, REST, SOAP)"
4. matched_skills = skills FROM THE CANDIDATE'S LIST that match JD requirements
5. missing_skills = skills FROM THE JD that the candidate does NOT have

DO NOT list a skill as "missing" if the candidate has an equivalent tool.

OUTPUT FORMAT:

{{
  "skill_match": {{
    "score": 0.0,
    "reasoning": "Brief explanation of skill alignment",
    "matched_skills": ["skill1", "skill2"],
    "missing_skills": ["skill3", "skill4"]
  }},
  "experience_depth": {{
    "score": 0.0,
    "reasoning": "Brief explanation including years comparison"
  }},
  "domain_fit": {{
    "score": 0.0,
    "reasoning": "Brief explanation of domain alignment"
  }},
  "overall_assessment": "1-2 sentence summary of candidacy"
}}

SCORING GUIDELINES:

1. **skill_match (0.0-1.0):**
   - 1.0 = Has ALL required skills + most preferred skills
   - 0.8 = Has ALL required skills, some preferred
   - 0.6 = Has MOST required skills
   - 0.4 = Has SOME required skills
   - 0.2 = Has FEW required skills
   - 0.0 = Has NO required skills

2. **experience_depth (0.0-1.0):**
   - Consider BOTH years AND relevance
   - If JD requires "3+ years" and candidate has 3+: score ≥ 0.7
   - If JD requires "3+ years" and candidate has 2 years: score ≤ 0.6
   - Relevant experience is worth more than total years
   - Senior roles with junior experience: score ≤ 0.5

3. **domain_fit (0.0-1.0):**
   - How well does candidate's background align with role domain?
   - For AI Applications Engineer: AI/ML background + customer-facing = high
   - Related domains (e.g., ML Engineer → AI Engineer) = medium-high
   - Unrelated domains = low

CRITICAL RULES:
- Use the FULL 0.0-1.0 scale (not just 0.6-0.9)
- Be critical but fair
- NO markdown formatting
- Return ONLY valid JSON
- Each score must be a float with 2 decimal places
- Reasoning should be 1-2 sentences max per dimension

Begin evaluation now:"""
    
    def _validate_scores(self, scores: Dict) -> Dict:
        """Validate score structure and ranges."""
        
        required_dims = ["skill_match", "experience_depth", "domain_fit"]
        
        for dim in required_dims:
            if dim not in scores:
                print(f"⚠️  Missing dimension: {dim}")
                scores[dim] = {"score": 0.5, "reasoning": "Not evaluated"}
            
            # Ensure score is float between 0 and 1
            if "score" in scores[dim]:
                try:
                    score = float(scores[dim]["score"])
                    scores[dim]["score"] = max(0.0, min(1.0, score))
                except (TypeError, ValueError):
                    scores[dim]["score"] = 0.5
        
        # Ensure skill_match has lists
        if "matched_skills" not in scores.get("skill_match", {}):
            scores["skill_match"]["matched_skills"] = []
        if "missing_skills" not in scores.get("skill_match", {}):
            scores["skill_match"]["missing_skills"] = []
        
        # Ensure overall_assessment exists
        if "overall_assessment" not in scores:
            scores["overall_assessment"] = "Evaluation completed"
        
        return scores
    
    def _get_default_scores(self) -> Dict:
        """Return default scores on failure."""
        return {
            "skill_match": {
                "score": 0.5,
                "reasoning": "Scoring failed",
                "matched_skills": [],
                "missing_skills": []
            },
            "experience_depth": {
                "score": 0.5,
                "reasoning": "Scoring failed"
            },
            "domain_fit": {
                "score": 0.5,
                "reasoning": "Scoring failed"
            },
            "overall_assessment": "Evaluation incomplete due to system error"
        }
