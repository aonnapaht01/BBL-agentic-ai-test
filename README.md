# BBL Multi-Agent RAG Assessment

A production-grade Multi-Agent Retrieval-Augmented Generation (RAG) architecture built for the **Bangkok Bank (BBL) Generative AI Developer** assessment. This project demonstrates advanced tool orchestration, stateful workflows, and zero-hallucination guardrails using LangChain, LangGraph, and Pydantic.

## 🌟 Key Features

- **Stateful Workflow (LangGraph):** Implements a robust state machine (`StateGraph`) defining the flow between specialized agents.
- **Multi-Agent Collaboration:**
  - **Data Retriever Agent:** Highly constrained tool-caller that strictly fetches raw, relevant context snippets from the knowledge base using term-frequency (TF) scoring.
  - **Report Generator Agent:** Synthesizes the raw retrieved context into a polished Markdown response, strictly bounded by the retrieved context.
- **Zero-Hallucination Guardrails:** System prompts are strictly engineered to prevent the LLM from hallucinating policies not found in the knowledge base, gracefully responding with "NO_RELEVANT_DATA_FOUND" mechanisms.
- **Dynamic Multi-Provider LLM Factory:** Seamlessly switches between LLM providers via `.env` configuration (BBL Custom API, Groq, Google Gemini, Azure OpenAI, standard OpenAI) without modifying code.

## 📂 Project Structure

```text
BBL/
├── data/
│   └── knowledge_base.txt       # Corporate policy knowledge base
├── screenshots/                 # Output screenshots and evidence of execution
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── nodes.py             # Graph node functions & AgentState schema
│   │   └── prompts.py           # Tailored system prompts for both agents
│   ├── tools/
│   │   ├── __init__.py
│   │   └── retriever.py         # TF-based custom retrieval logic + LangChain @tool
│   ├── __init__.py
│   ├── config.py                # Dynamic LLM factory & Pydantic settings
│   └── workflow.py              # LangGraph pipeline definition (StateGraph)
├── .env.example                 # Environment variable template
├── .gitignore                   # Standard Python gitignore
├── main.py                      # Interactive CLI entrypoint using `rich`
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- At least one API key (BBL Assessment Key, Groq, Gemini, Azure OpenAI, or OpenAI).

### 1. Installation

Clone the repository and install the required dependencies:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the template environment file to create your own `.env`:

```bash
cp .env.example .env
```

Open `.env` and configure your preferred LLM provider. The system dynamically selects the LLM based on the following priority:

1. **BBL Custom API (`BBL_API_KEY`):** Priority 0. Assessment-specific endpoint (defaults to `gpt-5-mini`).
2. **Groq (`GROQ_API_KEY`):** Priority 1. Fast & free local testing (defaults to `llama-3.1-8b-instant`).
3. **Google Gemini (`GOOGLE_API_KEY`):** Priority 2. Free local testing (defaults to `gemini-2.0-flash`).
4. **Azure OpenAI (`AZURE_OPENAI_*`):** Priority 3. Enterprise deployment.
5. **Standard OpenAI (`OPENAI_API_KEY`):** Priority 4. Default fallback.

*Note: Simply provide the API key for the provider you wish to use, and comment out the others.*

### 3. Usage

Run the CLI entrypoint to interact with the multi-agent system. The CLI provides beautifully formatted output using `rich`.

```bash
# Interactive mode (chat loop)
python main.py

# Run predefined BBL assessment demo queries
python main.py --demo

# Run a single one-shot query
python main.py -q "What is the daily travel per diem?"
```

## 📸 Screenshots & Verification

Execution outputs, handling of edge cases (e.g., no data found), and architecture verification are documented visually inside the `screenshots/` directory.

---
*Developed for the BBL Generative AI Developer Assessment.*
