"""
LangGraph node functions for the two-agent retrieval pipeline.

Defines the shared graph state and the two processing nodes:

  1. ``data_retriever_node`` – invokes the knowledge-base search tool and
     stores raw snippets.
  2. ``report_generator_node`` – feeds the snippets to an LLM with the
     Report Generator system prompt and produces the final answer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.prompts import (
    DATA_RETRIEVER_SYSTEM_PROMPT,
    REPORT_GENERATOR_SYSTEM_PROMPT,
)
from src.config import get_llm
from src.tools.retriever import search_knowledge_base_impl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph state schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Shared state flowing through every node in the LangGraph pipeline.

    Attributes:
        user_query: The original question submitted by the user.
        retrieved_snippets: Raw text snippets returned by the retriever.
        final_report: The polished, synthesised answer for the user.
        logs: An ordered list of human-readable log entries tracking
              each node's execution.
    """

    user_query: str
    retrieved_snippets: str
    final_report: str
    logs: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    """Return a compact UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Node 1 – Data Retriever
# ---------------------------------------------------------------------------


def data_retriever_node(state: AgentState) -> dict:
    """Search the knowledge base for snippets relevant to the user query.

    This node calls ``search_knowledge_base_impl`` directly (the pure-Python
    implementation) so retrieval works deterministically without an LLM call.
    The raw result is stored in ``retrieved_snippets``.

    Args:
        state: The current graph state containing ``user_query``.

    Returns:
        A partial state update with ``retrieved_snippets`` and a new log
        entry appended.
    """
    query: str = state["user_query"]
    logger.info("[DataRetriever] Searching for: %s", query)

    try:
        snippets = search_knowledge_base_impl(query=query)
    except FileNotFoundError as exc:
        logger.error("[DataRetriever] Knowledge base missing: %s", exc)
        snippets = "NO_RELEVANT_DATA_FOUND"
    except Exception as exc:  # noqa: BLE001
        logger.exception("[DataRetriever] Unexpected error during retrieval")
        snippets = "NO_RELEVANT_DATA_FOUND"

    snippet_preview = snippets[:120].replace("\n", " ")
    log_entry = (
        f"[{_timestamp()}] DataRetriever | query='{query}' | "
        f"result_length={len(snippets)} | preview='{snippet_preview}...'"
    )
    logger.info(log_entry)

    return {
        "retrieved_snippets": snippets,
        "logs": state.get("logs", []) + [log_entry],
    }


# ---------------------------------------------------------------------------
# Node 2 – Report Generator
# ---------------------------------------------------------------------------


def report_generator_node(state: AgentState) -> dict:
    """Synthesise retrieved snippets into a polished answer using the LLM.

    The Report Generator system prompt instructs the model to rely strictly
    on the provided context snippets and format the output in Markdown.

    Uses the centralized ``get_llm()`` factory from ``src.config`` to
    obtain the appropriate LLM backend (Google Gemini / Azure / OpenAI).

    Args:
        state: The current graph state containing ``user_query`` and
               ``retrieved_snippets``.

    Returns:
        A partial state update with ``final_report`` and a new log entry.
    """
    query: str = state["user_query"]
    snippets: str = state["retrieved_snippets"]
    logger.info("[ReportGenerator] Generating report for: %s", query)

    llm = get_llm()

    # Build the message sequence for the LLM
    messages = [
        SystemMessage(content=REPORT_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## User Question\n{query}\n\n"
                f"## Retrieved Context Snippets\n{snippets}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        final_report = response.content
    except Exception as exc:  # noqa: BLE001
        logger.exception("[ReportGenerator] LLM invocation failed")
        final_report = (
            "An error occurred while generating the report. "
            f"Please try again later. (Error: {exc})"
        )

    report_preview = final_report[:120].replace("\n", " ")
    log_entry = (
        f"[{_timestamp()}] ReportGenerator | query='{query}' | "
        f"report_length={len(final_report)} | preview='{report_preview}...'"
    )
    logger.info(log_entry)

    return {
        "final_report": final_report,
        "logs": state.get("logs", []) + [log_entry],
    }
