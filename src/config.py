"""
Centralized configuration management.

Loads environment variables from ``.env`` and exposes typed, validated
configuration objects used across the application. Supports both standard
OpenAI and Azure OpenAI endpoints.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Load .env from the project root (one level above src/)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
KNOWLEDGE_BASE_FILE = DATA_DIR / "knowledge_base.txt"


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Supports both **standard OpenAI** and **Azure OpenAI** endpoints.
    When ``AZURE_OPENAI_ENDPOINT`` is set, the system will use the Azure
    client; otherwise it falls back to the standard OpenAI API.
    """

    # --- OpenAI (standard) --------------------------------------------------
    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
        description="API key for the standard OpenAI endpoint.",
    )

    # --- Azure OpenAI (optional) ---------------------------------------------
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        alias="AZURE_OPENAI_API_KEY",
        description="API key for Azure OpenAI. Leave blank to use standard OpenAI.",
    )
    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        alias="AZURE_OPENAI_ENDPOINT",
        description="Azure OpenAI endpoint URL (e.g. https://<resource>.openai.azure.com/).",
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01",
        alias="AZURE_OPENAI_API_VERSION",
        description="Azure OpenAI API version.",
    )

    # --- Model configuration -------------------------------------------------
    llm_model: str = Field(
        default="gpt-4o-mini",
        alias="LLM_MODEL",
        description="Model name / deployment name to use for chat completions.",
    )
    llm_temperature: float = Field(
        default=0.0,
        alias="LLM_TEMPERATURE",
        description="Sampling temperature (0.0 = deterministic, 1.0 = creative).",
    )

    # --- Knowledge base ------------------------------------------------------
    knowledge_base_path: str = Field(
        default="./data/knowledge_base.txt",
        alias="KNOWLEDGE_BASE_PATH",
        description="Path to the knowledge base text file.",
    )

    # --- Pydantic settings ---------------------------------------------------
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }

    # --- Computed helpers ----------------------------------------------------

    @property
    def use_azure(self) -> bool:
        """Return True if Azure OpenAI credentials are configured."""
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance.

    The first call reads from the environment / ``.env`` file; subsequent
    calls return the same object without re-parsing.
    """
    return Settings()
