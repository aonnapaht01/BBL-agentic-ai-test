"""
Centralized configuration management.

Loads environment variables from ``.env`` and exposes typed, validated
configuration objects used across the application. Supports **Groq**,
**Google Gemini**, **Azure OpenAI**, and **standard OpenAI** endpoints with 
automatic priority-based selection via the ``get_llm()`` factory.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

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

    Supports four LLM backends with priority-based selection:

    1. **Groq** – fast & free local testing (``GROQ_API_KEY``).
    2. **Google Gemini** – free-tier local testing (``GOOGLE_API_KEY``).
    3. **Azure OpenAI** – enterprise / BBL submission.
    4. **Standard OpenAI** – default fallback.
    """

    # --- Groq (Priority 1) --------------------------------------------------
    groq_api_key: Optional[str] = Field(
        default=None,
        alias="GROQ_API_KEY",
        description="API key for Groq. When set, Groq is used as the LLM.",
    )

    # --- Google Gemini (Priority 2) -----------------------------------------
    google_api_key: Optional[str] = Field(
        default=None,
        alias="GOOGLE_API_KEY",
        description="API key for Google Gemini. When set, Gemini is used as the LLM.",
    )

    # --- OpenAI (standard) (Priority 4) -------------------------------------
    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
        description="API key for the standard OpenAI endpoint.",
    )

    # --- Azure OpenAI (Priority 3) ------------------------------------------
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
        default="gpt-5-mini",
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
    def use_groq(self) -> bool:
        """Return True if Groq credentials are configured."""
        return bool(self.groq_api_key)

    @property
    def use_google(self) -> bool:
        """Return True if Google Gemini credentials are configured."""
        return bool(self.google_api_key)

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


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm() -> BaseChatModel:
    """Dynamically instantiate the appropriate LangChain Chat Model.

    Selection priority:

    1. **Groq** — if ``GROQ_API_KEY`` is set.
       Uses ``llama-3.1-8b-instant`` by default (overridden if
       ``LLM_MODEL`` contains ``"llama"`` or ``"mixtral"``).
    2. **Google Gemini** — if ``GOOGLE_API_KEY`` is set.
       Uses ``gemini-2.0-flash`` by default (overridden if
       ``LLM_MODEL`` contains ``"gemini"``).
    3. **Azure OpenAI** — if ``AZURE_OPENAI_ENDPOINT`` *and*
       ``AZURE_OPENAI_API_KEY`` are both set.
    4. **Standard OpenAI** — fallback using ``OPENAI_API_KEY``.

    Returns:
        A ready-to-use ``BaseChatModel`` instance.

    Raises:
        ImportError: If required packages like ``langchain-groq`` or
            ``langchain-google-genai`` are not installed.
    """
    settings = get_settings()

    # ── Priority 1: Groq ────────────────────────────────────────────────
    if settings.use_groq:
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError(
                "Groq is configured (GROQ_API_KEY is set) but the "
                "required package is missing.\n"
                "Install it with:  pip install langchain-groq"
            ) from exc

        # Use LLM_MODEL if it's a Groq model name; otherwise default
        model_name = (
            settings.llm_model
            if any(x in settings.llm_model.lower() for x in ["llama", "mixtral"])
            else "llama-3.1-8b-instant"
        )
        logger.info("Using Groq model: %s", model_name)
        return ChatGroq(
            model=model_name,
            groq_api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
        )

    # ── Priority 2: Google Gemini ───────────────────────────────────────
    if settings.use_google:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Google Gemini is configured (GOOGLE_API_KEY is set) but the "
                "required package is missing.\n"
                "Install it with:  pip install langchain-google-genai"
            ) from exc

        # Use LLM_MODEL if it's a Gemini model name; otherwise default
        model_name = (
            settings.llm_model
            if "gemini" in settings.llm_model.lower()
            else "gemini-2.0-flash"
        )
        logger.info("Using Google Gemini model: %s", model_name)
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
        )

    # ── Priority 3: Azure OpenAI ────────────────────────────────────────
    if settings.use_azure:
        from langchain_openai import AzureChatOpenAI

        logger.info(
            "Using Azure OpenAI | endpoint=%s | deployment=%s",
            settings.azure_openai_endpoint,
            settings.llm_model,
        )
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.llm_model,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            temperature=settings.llm_temperature,
        )

    # ── Priority 4: Standard OpenAI (fallback) ──────────────────────────
    from langchain_openai import ChatOpenAI

    logger.info("Using standard OpenAI model: %s", settings.llm_model)
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )