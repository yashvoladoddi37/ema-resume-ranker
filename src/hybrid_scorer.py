import os
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from src.deterministic import DeterministicExtractor
from src.prompts import RESUME_SCORING_PROMPT

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridScorer:
    """
    Production-grade hybrid scoring system combining:
    1. LLM intelligence (contextual understanding)
    2. Deterministic rules (verifiable metrics)
    3. HITL override capabilities
    """
    
    def __init__(self, llm_weight: float = 0.6, deterministic_weight: float = 0.4):
        from groq import Groq
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.llm_weight = llm_weight
        self.deterministic_weight = deterministic_weight
        self.extractor = DeterministicExtractor()
        
    def score(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        """
        Hybrid scoring with full transparency and auditability.
        
        Returns:
        {
            'final_score': float,
            'llm_component': {...},
            'deterministic_component': {...},
            'hitl_override': None or float,
            'audit_trail': {...}
        }
        """
        # 1. Deterministic Analysis (Always runs first - verifiable)
        years_exp = self.extractor.extract_years_of_experience(resume_text)
        skill_profile = self.extractor.extract_skills(resume_text)
        domain_relevance = self.extractor.calculate_domain_relevance(resume_text)
        
        deterministic_score, score_breakdown = self.extractor.calculate_deterministic_score(
            years_exp, skill_profile, domain_relevance
        )
        
        # 2. LLM Analysis (Contextual understanding with Facts)
        # Format deterministic facts for the LLM
        det_context_str = (
            f"Extracted Years of Experience: {years_exp} (Target: 3+)\n"
            f"Matched Skills: {', '.join(skill_profile.matched_required)}\n"
            f"Missing Required Skills: {', '.join(skill_profile.missing_required)}\n"
            f"Rule-Based Score: {deterministic_score:.2f}/1.0"
        )
        
        llm_result = self._call_llm(job_description, resume_text, det_context_str)
        
        # 3. Hybrid Combination
        final_score = (
            self.llm_weight * llm_result['score'] +
            self.deterministic_weight * deterministic_score
        )
        
        # 4. Build comprehensive result
        return {
            'final_score': round(final_score, 3),
            'llm_component': {
                'score': llm_result['score'],
                'weight': self.llm_weight,
                'reasoning': llm_result['reasoning'],
                'matched_skills': llm_result['matched_skills'],
                'missing_skills': llm_result['missing_skills']
            },
            'deterministic_component': {
                'score': deterministic_score,
                'weight': self.deterministic_weight,
                'years_experience': years_exp,
                'skill_coverage': skill_profile.skill_coverage_score,
                'matched_required_skills': list(skill_profile.matched_required),
                'matched_preferred_skills': list(skill_profile.matched_preferred),
                'missing_required_skills': list(skill_profile.missing_required),
                'ai_relevance': domain_relevance['ai_relevance'],
                'support_relevance': domain_relevance['support_relevance'],
                'score_breakdown': score_breakdown
            },
            'hitl_override': None,  # Placeholder for human override
            'audit_trail': {
                'llm_model': 'llama-3.3-70b-versatile',
                'deterministic_version': '1.0',
                'weights': {
                    'llm': self.llm_weight,
                    'deterministic': self.deterministic_weight
                }
            }
        }
    
    def _call_llm(self, job_description: str, resume_text: str, deterministic_context: str) -> dict:
        """LLM scoring using the centralized prompt with verified facts."""
        prompt = RESUME_SCORING_PROMPT.format(
            job_description=job_description,
            deterministic_context=deterministic_context,
            resume_text=resume_text
        )

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
            return {
                "score": 0.0,
                "reasoning": f"Error: {str(e)}",
                "matched_skills": [],
                "missing_skills": []
            }
    
    def apply_hitl_override(self, result: Dict[str, Any], override_score: float, reason: str) -> Dict[str, Any]:
        """
        Apply human-in-the-loop override for edge cases.
        
        Use cases:
        - Candidate has unique experience not captured by rules
        - LLM hallucinated or missed critical context
        - Manual review reveals discrepancy
        """
        result['hitl_override'] = {
            'score': override_score,
            'reason': reason,
            'original_score': result['final_score']
        }
        result['final_score'] = override_score
        result['audit_trail']['hitl_applied'] = True
        
        return result
