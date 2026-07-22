"""
LangGraph workflow for the two-agent retrieval pipeline.

Assembles the ``data_retriever_node`` and ``report_generator_node`` into a
sequential ``StateGraph`` and exposes a convenience ``run_pipeline()`` helper
for one-shot query execution.

Graph topology::

    START ──> retriever ──> generator ──> END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    AgentState,
    data_retriever_node,
    report_generator_node,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build the LangGraph workflow
# ---------------------------------------------------------------------------


def build_workflow() -> StateGraph:
    """Construct and return the (uncompiled) StateGraph.

    Returns:
        A ``StateGraph`` with ``retriever`` and ``generator`` nodes wired
        in sequence.
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("retriever", data_retriever_node)
    workflow.add_node("generator", report_generator_node)

    # Define sequential edges: START -> retriever -> generator -> END
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)

    return workflow


# ---------------------------------------------------------------------------
# Compiled application (module-level singleton)
# ---------------------------------------------------------------------------

_workflow = build_workflow()
app = _workflow.compile()
"""Compiled LangGraph application ready for ``.invoke()`` calls."""


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_pipeline(query: str) -> dict[str, Any]:
    """Execute the full retrieval-and-report pipeline for a single query.

    Args:
        query: The user's natural-language question.

    Returns:
        A dictionary containing the final ``AgentState`` with keys:
        ``user_query``, ``retrieved_snippets``, ``final_report``, and
        ``logs``.

    Example::

        >>> from src.workflow import run_pipeline
        >>> result = run_pipeline("What is the daily travel per diem?")
        >>> print(result["final_report"])
    """
    logger.info("=" * 60)
    logger.info("Pipeline started | query='%s'", query)
    logger.info("=" * 60)

    initial_state: AgentState = {
        "user_query": query,
        "retrieved_snippets": "",
        "final_report": "",
        "logs": [],
    }

    final_state = app.invoke(initial_state)

    logger.info("Pipeline completed | logs=%d entries", len(final_state.get("logs", [])))
    for entry in final_state.get("logs", []):
        logger.info("  %s", entry)

    return final_state
