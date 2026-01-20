from typing import List, Dict, Any
import logging
from src.config import config
from src.preprocessing import TextPreprocessor
from src.scorers.llm import LLMScorer
from src.scorers.semantic import SemanticScorer

logger = logging.getLogger(__name__)

class ResumeMatchingEngine:
    """
    The orchestrator that combines LLM reasoning with Semantic baselines.
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.llm_scorer = LLMScorer()
        self.semantic_scorer = SemanticScorer()

    def rank_resumes(self, job_description: str, resumes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of resumes and returns them ranked by final score.
        Input format: [{"id": "res_001", "text": "..."}, ...]
        """
        results = []
        clean_jd = self.preprocessor.clean_text(job_description)
        
        for res in resumes:
            logger.info(f"Processing candidate: {res['id']}")
            clean_resume = self.preprocessor.clean_text(res['text'])
            
            # 1. LLM Scoring (Structured)
            llm_data = self.llm_scorer.score(clean_jd, clean_resume)
            
            # 2. Semantic Scoring (Baseline)
            semantic_score = self.semantic_scorer.score(clean_jd, clean_resume)
            
            # 3. Hybrid Aggregation
            final_score = (config.WEIGHT_LLM * llm_data["score"]) + \
                          (config.WEIGHT_EMBEDDING * semantic_score)
            
            results.append({
                "id": res["id"],
                "final_score": round(final_score, 4),
                "llm_score": llm_data["score"],
                "semantic_score": round(semantic_score, 4),
                "reasoning": llm_data["reasoning"],
                "matched_skills": llm_data["matched_skills"],
                "missing_skills": llm_data["missing_skills"]
            })
            
        # Rank by final_score descending
        ranked_results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        return ranked_results
