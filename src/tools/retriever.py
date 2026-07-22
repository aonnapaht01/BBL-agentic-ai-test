"""
Knowledge Base Retrieval Tool.

Provides a keyword-based retrieval mechanism over a plain-text knowledge base.
Paragraphs are ranked by term-frequency overlap with the user query and the
top-k results are returned. Exposed as a LangChain tool for agent invocation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Default knowledge base path (relative to project root)
# ---------------------------------------------------------------------------
_DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base.txt"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_paragraphs(filepath: Path) -> list[str]:
    """Read a text file and split it into non-empty paragraphs.

    Paragraphs are delimited by one or more blank lines.

    Args:
        filepath: Absolute or relative path to the knowledge base file.

    Returns:
        A list of paragraph strings with leading/trailing whitespace stripped.

    Raises:
        FileNotFoundError: If the knowledge base file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {filepath}")

    raw_text = filepath.read_text(encoding="utf-8")

    # Split on two or more consecutive newlines to separate paragraphs
    paragraphs = re.split(r"\n{2,}", raw_text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _score_paragraph(query_tokens: list[str], paragraph: str) -> float:
    """Score a paragraph against query tokens using TF-based overlap.

    The scoring formula rewards:
      1. **Coverage** – the fraction of unique query terms found in the paragraph.
      2. **Frequency** – the sum of log(1 + tf) for each matching query term,
         normalised by paragraph length to avoid bias toward longer text.

    Final score = coverage_ratio * normalised_frequency_sum

    Args:
        query_tokens: Tokenised query terms.
        paragraph: Raw paragraph text.

    Returns:
        A non-negative relevance score (0.0 means no match).
    """
    if not query_tokens:
        return 0.0

    para_tokens = _tokenize(paragraph)
    if not para_tokens:
        return 0.0

    para_freq = Counter(para_tokens)
    unique_query = set(query_tokens)

    matched_terms = 0
    freq_sum = 0.0

    for term in unique_query:
        tf = para_freq.get(term, 0)
        if tf > 0:
            matched_terms += 1
            freq_sum += math.log(1 + tf)

    coverage = matched_terms / len(unique_query)
    normalised_freq = freq_sum / len(para_tokens)

    return coverage * normalised_freq


def search_knowledge_base_impl(
    query: str,
    filepath: str | None = None,
    top_k: int = 2,
) -> str:
    """Search the knowledge base and return the most relevant paragraphs.

    Args:
        query: The natural-language search query.
        filepath: Path to the knowledge base text file.
                  Defaults to ``data/knowledge_base.txt`` in the project root.
        top_k: Number of top-ranked paragraphs to return.

    Returns:
        A string containing the top-k paragraphs separated by ``---``,
        or ``"NO_RELEVANT_DATA_FOUND"`` if no paragraph has a positive score.
    """
    kb_path = Path(filepath) if filepath else _DEFAULT_KB_PATH
    paragraphs = _load_paragraphs(kb_path)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return "NO_RELEVANT_DATA_FOUND"

    # Score and rank
    scored: list[tuple[float, str]] = [
        (_score_paragraph(query_tokens, para), para) for para in paragraphs
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Filter out zero-score paragraphs and take top_k
    top_results = [para for score, para in scored[:top_k] if score > 0]

    if not top_results:
        return "NO_RELEVANT_DATA_FOUND"

    return "\n---\n".join(top_results)


# ---------------------------------------------------------------------------
# LangChain Tool
# ---------------------------------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """Search the corporate knowledge base for policy information.

    Use this tool when you need to look up company policies such as
    travel allowances, remote work rules, leave entitlements, welfare
    benefits, data security guidelines, or training budgets.

    Args:
        query: A natural-language question or keyword phrase describing
               the information you need (e.g. "international travel per diem").

    Returns:
        The most relevant policy paragraphs, or "NO_RELEVANT_DATA_FOUND"
        if nothing matches.
    """
    return search_knowledge_base_impl(query=query)
