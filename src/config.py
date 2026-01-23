import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EngineConfig:
    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "gemini"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model Names
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Validated Weights (V1 verified: 0.55/0.15/0.30)
    WEIGHT_SKILL: float = float(os.getenv("WEIGHT_SKILL", "0.55"))
    WEIGHT_EXPERIENCE: float = float(os.getenv("WEIGHT_EXPERIENCE", "0.15"))
    WEIGHT_DOMAIN: float = float(os.getenv("WEIGHT_DOMAIN", "0.30"))

    # V2/V3 Weights
    WEIGHT_DETERMINISTIC: float = float(os.getenv("WEIGHT_DETERMINISTIC", "0.70"))
    WEIGHT_SEMANTIC: float = float(os.getenv("WEIGHT_SEMANTIC", "0.30"))
    WEIGHT_LLM_V3: float = float(os.getenv("WEIGHT_LLM_V3", "0.60"))
    WEIGHT_DET_V3: float = float(os.getenv("WEIGHT_DET_V3", "0.40"))

    # Processing
    MAX_RESUME_LENGTH: int = int(os.getenv("MAX_RESUME_LENGTH", "8000"))
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Global instance
config = EngineConfig()
