"""
Centralized configuration management.

Loads environment variables and provides typed configuration
access throughout the application.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env file from project root
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # Knowledge base
    knowledge_base_path: str = Field(default="./data", alias="KNOWLEDGE_BASE_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
