# BBL AI Engineer Programming Test

An AI-powered agent system built with LangChain and LangGraph, designed to demonstrate advanced capabilities in tool orchestration, multi-agent collaboration, and intelligent decision-making.

## Project Structure

```
BBL/
├── data/                   # Knowledge base and data files
├── src/
│   ├── __init__.py
│   ├── config.py           # Centralized configuration management
│   ├── tools/
│   │   └── __init__.py     # Custom tool definitions
│   └── agents/
│       └── __init__.py     # Agent definitions and orchestration
├── screenshots/            # Output screenshots and evidence
├── .env.example            # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## License

This project is part of the BBL AI Engineer assessment.
