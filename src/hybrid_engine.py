import os
import json
import logging
from typing import Dict, Any, List
from src.deterministic_engine import DeterministicEngine
from src.matching_engine import TwoStageMatchingEngine  # V1 Logic
from src.config import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class HybridEngine:
    """
    V3 Engine: Parallel Ensemble (Hybrid).
    
    Combines:
    1. V1: LLM Pipeline (TwoStageMatchingEngine) - High reasoning, high hallucination risk.
    2. V2: Deterministic Engine (Regex/Embeddings) - Strict, no hallucination.
    
    Score = Weighted Average (V1, V2).
    """
    
    def __init__(self):
        self.v1_engine = TwoStageMatchingEngine()  # The exact V1 architecture
        self.v2_engine = DeterministicEngine()     # The exact V2 architecture
        
        # V3 Weights from config
        self.weight_v1 = config.WEIGHT_LLM_V3  # e.g. 0.6
        self.weight_v2 = config.WEIGHT_DET_V3  # e.g. 0.4
        logger.info(f"Initialized HybridEngine Ensemble: V1={self.weight_v1}, V2={self.weight_v2}")
        
    def evaluate(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        """
        Evaluate using both engines and ensemble the results.
        """
        # 1. Run V2 (Deterministic)
        try:
            v2_result = self.v2_engine.evaluate(job_description, resume_text)
            v2_score = v2_result['final_score']
        except Exception as e:
            logger.error(f"V2 Evaluation failed: {e}")
            v2_score = 0.5
            v2_result = {}

        # 2. Run V1 (LLM)
        try:
            # We don't have resume_id here easily, passing generic
            v1_result = self.v1_engine.evaluate(job_description, resume_text, resume_id="hybrid_eval")
            v1_score = v1_result['final_score']
        except Exception as e:
            logger.error(f"V1 Evaluation failed: {e}")
            v1_score = 0.5
            v1_result = {}
        
        # 3. Ensemble
        final_score = (self.weight_v1 * v1_score) + (self.weight_v2 * v2_score)
        
        return {
            "final_score": round(final_score, 3),
            "v1_score": v1_score,
            "v2_score": v2_score,
            "v1_breakdown": v1_result.get('score_breakdown', ''),
            "v2_breakdown": v2_result.get('breakdown', {}),
            "reasoning": f"Ensemble Score: {self.weight_v1*100}% V1 ({v1_score}) + {self.weight_v2*100}% V2 ({v2_score})",
            "matched_skills": list(set(v1_result.get('matched_skills', []) + v2_result.get('extracts', {}).get('matched_required_skills', [])))
        }

    def evaluate_batch(self, job_description: str, resumes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results = []
        for resume in resumes:
            res = self.evaluate(job_description, resume['text'])
            res['candidate_id'] = resume.get('id', 'unknown')
            res['candidate_name'] = resume.get('name', 'unknown')
            results.append(res)
            
        return sorted(results, key=lambda x: x['final_score'], reverse=True)
