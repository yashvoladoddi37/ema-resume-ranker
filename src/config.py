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
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Weights
    WEIGHT_LLM: float = 0.70
    WEIGHT_EMBEDDING: float = 0.30
    
    # Processing
    MAX_RESUME_LENGTH: int = 8000
    TEMPERATURE: float = 0.0

# Global instance
config = EngineConfig()
