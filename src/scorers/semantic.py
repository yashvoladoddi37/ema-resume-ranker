from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer, util
from src.config import config

class SemanticScorer:
    """
    Computes semantic similarity embeddings using a Bi-Encoder.
    This serves as our deterministic baseline.
    """
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        
    def score(self, job_description: str, resume: str) -> float:
        """
        Calculates cosine similarity between JD and Resume.
        Returns a float between 0 and 1.
        """
        # Encode both texts
        jd_emb = self.model.encode(job_description, convert_to_tensor=True)
        resume_emb = self.model.encode(resume, convert_to_tensor=True)
        
        # Compute cosine similarity
        score = util.cos_sim(jd_emb, resume_emb).item()
        
        # Normalize/Clip to [0, 1] range just in case
        return max(0.0, min(1.0, float(score)))

    def batch_score(self, job_description: str, resumes: List[str]) -> List[float]:
        """
        Batch process multiple resumes for efficiency.
        """
        jd_emb = self.model.encode(job_description, convert_to_tensor=True)
        resume_embs = self.model.encode(resumes, convert_to_tensor=True)
        
        # util.cos_sim returns a matrix, we need diagonal for 1-to-many if matched
        # But here it's 1 JD to N resumes, so it returns [1, N]
        scores = util.cos_sim(jd_emb, resume_embs)[0].tolist()
        return [float(s) for s in scores]
