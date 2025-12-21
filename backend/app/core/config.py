from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Data Scientist Agent"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "211e372e3cff13169f6ca76c3eede6672ada9ddbc0998ea016f0d1dae8858589")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data_scientist_agent.db")
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "uploads"
    ALLOWED_EXTENSIONS: set = {
        "csv", "xlsx", "xls", "json", "parquet", "arff", "tsv"
    }
    
    # ML Settings
    MAX_FEATURES_AUTO_SELECTION: int = 50
    DEFAULT_CV_FOLDS: int = 5
    MAX_OPTUNA_TRIALS: int = 100
    
    # OpenAI (optional for advanced analysis)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
