import os
import json
from typing import Dict, Any, List
from src.deterministic_engine import DeterministicEngine
from src.scorers.llm import LLMScorer

class HybridEngine:
    """
    V3 Engine: Sequential Hybrid Approach.
    
    Architecture:
    1. Deterministic Extraction (Regex/Embeddings) provides "Ground Truth".
    2. LLM reasons about candidate fit, grounded by the deterministic facts.
    3. Final score = 60% LLM + 40% Deterministic.
    """
    
    def __init__(self):
        self.det_engine = DeterministicEngine()
        self.llm_scorer = LLMScorer()
        
        # V3 Weights
        self.weight_llm = 0.6
        self.weight_det = 0.4
        
    def evaluate(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        """
        Evaluate a resume using sequential hybrid logic.
        """
        # 1. Stage 1: Deterministic "Ground Truth"
        det_results = self.det_engine.evaluate(job_description, resume_text)
        
        # 2. Stage 2: Grounded LLM Reasoning
        # We enrich the prompt with deterministic facts to prevent hallucinations
        det_context = {
            "years_experience": det_results['extracts']['years_experience'],
            "matched_skills": det_results['extracts']['matched_required_skills'] + det_results['extracts']['matched_preferred_skills'],
            "missing_required_skills": det_results['extracts']['missing_required_skills']
        }
        
        # We need a special score method for LLM that accepts context
        # I'll update src/scorers/llm.py or pass it as part of the resume text
        llm_results = self.llm_scorer.score_with_context(
            job_description, 
            resume_text, 
            det_context
        )
        
        # 3. Aggregation
        llm_score = llm_results.get('score', 0.5)
        det_score = det_results['final_score']
        
        final_score = (self.weight_llm * llm_score) + (self.weight_det * det_score)
        
        return {
            "final_score": round(final_score, 3),
            "llm_score": llm_score,
            "deterministic_score": det_score,
            "reasoning": llm_results.get('reasoning', ''),
            "matched_skills": llm_results.get('matched_skills', []),
            "missing_skills": llm_results.get('missing_skills', []),
            "det_extracts": det_results['extracts'],
            "det_breakdown": det_results['breakdown']
        }

    def evaluate_batch(self, job_description: str, resumes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results = []
        for resume in resumes:
            res = self.evaluate(job_description, resume['text'])
            res['candidate_id'] = resume.get('id', 'unknown')
            res['candidate_name'] = resume.get('name', 'unknown')
            results.append(res)
            
        return sorted(results, key=lambda x: x['final_score'], reverse=True)
