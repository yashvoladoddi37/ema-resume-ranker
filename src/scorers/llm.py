import os
import json

from groq import Groq
from dotenv import load_dotenv
from typing import Dict, Any

class LLMScorer:
    """
    Interfaces with LLM providers to perform nuanced resume evaluation.
    Updated for V3 to support Grounded Reasoning.
    """
    
    def __init__(self, provider: str = "groq"):
        load_dotenv()
        self.provider = provider.lower()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        self.temperature = 0.0

    def score(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        """Standard scoring without context (V1 style)"""
        return self.score_with_context(job_description, resume_text, None)

    def score_with_context(self, job_description: str, resume_text: str, det_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Grounded scoring using deterministic ground truth.
        """
        prompt = self._build_prompt(job_description, resume_text, det_context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM Error: {e}")
            return {"score": 0.5, "reasoning": "Error in LLM scoring"}

    def _build_prompt(self, jd: str, resume: str, context: Dict[str, Any]) -> str:
        context_str = ""
        if context:
            context_str = f"""
VERIFIED FACTS (from deterministic extraction):
- Years of Experience: {context['years_experience']}
- Matched Skills: {', '.join(context['matched_skills'])}
- Missing Required Skills: {', '.join(context['missing_required_skills'])}

USE THESE FACTS AS GROUND TRUTH. Do not contradict them. If the resume has additional evidence for skills not listed above, you may consider them, but stay critical.
"""

        return f"""You are a senior technical recruiter evaluating a candidate for an AI Applications Engineer role.

JOB DESCRIPTION:
{jd}

RESUME TEXT:
{resume}
{context_str}

TASK:
Provide a nuanced evaluation of the candidate. Bridge the gap between raw text and JD requirements.
Example: If the resume lists "Prometheus" and JD asks for "logging tools", confirm this as a match.

Return ONLY a JSON object:
{{
  "score": 0.0-1.0,
  "reasoning": "Nuanced explanation (2-3 sentences)",
  "matched_skills": ["List specific matched skills"],
  "missing_skills": ["List missing critical skills"]
}}
"""
