import os
import json
from typing import Dict, Any, List
from src.deterministic import DeterministicExtractor
from src.scorers.semantic import SemanticScorer

class DeterministicEngine:
    """
    V2 Engine: Pure Deterministic Approach (Embeddings + Regex).
    No LLMs used. Targets reliability, speed, and cost-effectiveness.
    """
    
    def __init__(self):
        self.extractor = DeterministicExtractor()
        self.semantic_scorer = SemanticScorer()
        
        # V2 Weights: Focus on verifiable facts (70%) vs semantic gist (30%)
        self.weight_deterministic = 0.7
        self.weight_semantic = 0.3
        
    def evaluate(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        """
        Evaluate a single resume using pure deterministic methods.
        """
        # 1. Deterministic Extraction (Regex/Keywords)
        years_exp = self.extractor.extract_years_of_experience(resume_text)
        skill_profile = self.extractor.extract_skills(resume_text)
        domain_relevance = self.extractor.calculate_domain_relevance(resume_text)
        
        det_score, det_breakdown = self.extractor.calculate_deterministic_score(
            years_exp, skill_profile, domain_relevance
        )
        
        # 2. Semantic Scoring (Embeddings)
        semantic_score = self.semantic_scorer.score(job_description, resume_text)
        
        # 3. Final Aggregation
        final_score = (self.weight_deterministic * det_score) + (self.weight_semantic * semantic_score)
        
        return {
            "final_score": round(final_score, 3),
            "deterministic_score": det_score,
            "semantic_score": semantic_score,
            "breakdown": {
                **det_breakdown,
                "semantic_similarity": round(semantic_score, 3)
            },
            "extracts": {
                "years_experience": years_exp,
                "matched_required_skills": list(skill_profile.matched_required),
                "matched_preferred_skills": list(skill_profile.matched_preferred),
                "missing_required_skills": list(skill_profile.missing_required)
            }
        }

    def evaluate_batch(self, job_description: str, resumes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple resumes.
        """
        results = []
        for resume in resumes:
            res = self.evaluate(job_description, resume['text'])
            res['candidate_id'] = resume.get('id', 'unknown')
            res['candidate_name'] = resume.get('name', 'unknown')
            results.append(res)
            
        # Rank by final score
        return sorted(results, key=lambda x: x['final_score'], reverse=True)
