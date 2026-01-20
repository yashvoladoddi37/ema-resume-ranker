import json
import logging
from typing import Dict, Any, Optional
from src.config import config
from src.prompts import RESUME_SCORING_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMScorer:
    """
    Interfaces with LLM providers (Groq/Gemini) to perform nuanced resume evaluation.
    """
    
    def __init__(self, provider: str = config.LLM_PROVIDER):
        self.provider = provider.lower()
        self.api_key = config.GROQ_API_KEY if self.provider == "groq" else config.GEMINI_API_KEY
        
        if not self.api_key:
            logger.warning(f"No API key found for {self.provider}. Scorer will fail.")

    def _call_groq(self, prompt: str) -> str:
        from groq import Groq
        client = Groq(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.TEMPERATURE,
            response_format={"type": "json_object"}
        )
        return completion.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash") # Fallback to flash for speed
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=config.TEMPERATURE,
                response_mime_type="application/json"
            )
        )
        return response.text

    def score(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        """
        Executes the LLM scoring pipeline.
        """
        prompt = RESUME_SCORING_PROMPT.format(
            job_description=job_description,
            resume_text=resume_text
        )
        
        try:
            if self.provider == "groq":
                raw_response = self._call_groq(prompt)
            elif self.provider == "gemini":
                raw_response = self._call_gemini(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Parse and validate JSON
            data = json.loads(raw_response)
            
            # Ensure required fields exist
            required_keys = ["score", "reasoning", "matched_skills", "missing_skills"]
            for key in required_keys:
                if key not in data:
                    data[key] = 0.0 if key == "score" else "N/A" if key == "reasoning" else []
            
            return data
            
        except Exception as e:
            logger.error(f"LLM Scoring Error: {e}")
            return {
                "score": 0.0,
                "reasoning": f"Error during scoring: {str(e)}",
                "matched_skills": [],
                "missing_skills": []
            }
