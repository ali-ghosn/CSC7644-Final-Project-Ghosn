# AI Task Decomposition Copilot

**CSC 7644: Applied LLM Development — Final Project**  
Louisiana State University

A deterministic, schema-constrained tool-calling system that converts natural language technical tasks into structured execution plans and executes them through a validated controller-loop workflow.

---

## Overview

The **AI Task Decomposition Copilot** is a systems-focused LLM project built for **CSC 7644: Applied LLM Development** at Louisiana State University.

The project was designed to evaluate how structured planning, schema validation, and controller-loop execution affect the reliability and explainability of LLM-generated workflows.

Given a natural language task like:

> *"Add 3 and 5, then multiply the result by 2."*

the system:

1. Sends the task to an **LLM Planner** that generates a strict JSON execution plan.
2. Validates the generated plan before any execution occurs.
3. Passes the validated plan to a **Controller Loop** that resolves references, executes tools sequentially, and halts immediately on failure.
4. Returns both structured results and readable terminal logs.

This project focuses on validated tool execution rather than autonomous behavior. The goal was to build a reliable orchestration workflow with explicit validation and execution constraints.

---

## Key Features

- Strict JSON-only planner output
- JSON Schema validation before every tool execution
- Sequential synchronous execution flow
- Retry/recovery loop for malformed planner output
- Multi-step execution support
- Variable reference resolution using `$prev`
- Sandboxed file access restricted to `./workspace/`
- Structured terminal logging for debugging and demonstrations
- Modular tool registry for easier extension and testing
- Pytest-based validation and sandbox testing

---

## Architecture

```text
┌─────────────────────┐
│      User Input     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│     LLM Planner     │
│  (Strict JSON Out)  │
└─────────┬───────────┘
          │
          ▼
┌────────────────────────────────────────────┐
│               Controller Loop              │
│ Parse → Validate → Resolve → Execute       │
└──────────────────┬─────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌─────────────────┐   ┌──────────────────┐
│ JSON Schema     │   │  Tool Executor   │
│ Validation      │   │  via Registry    │
└─────────────────┘   └────────┬─────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │   Final Output   │
                      └──────────────────┘
```

---

## Component Summary

| Component | File | Responsibility |
|---|---|---|
| LLM Planner | `planner.py` | Generates structured execution plans |
| Controller Loop | `controller.py` | Validates and executes steps sequentially |
| Tool Registry | `registry.py` | Maps tool names to functions and schemas |
| Schema Definitions | `schemas/` | JSON Schema validation definitions |
| Tool Implementations | `tools/` | Individual tool logic |
| Validators | `utils/validators.py` | Plan and argument validation |
| Path Sandbox | `utils/path_utils.py` | Restricts filesystem access |
| Logging Utilities | `utils/logging_utils.py` | Structured terminal output helpers |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM API | OpenAI Chat Completions API |
| Validation | `jsonschema` |
| Environment Config | `python-dotenv` |
| Testing | `pytest` |
| Architecture Style | Pure Python controller-loop design |

The project intentionally avoids orchestration frameworks such as LangChain or CrewAI in order to keep the architecture explicit, modular, and easier to reason about.

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- pip
- Git
- OpenAI API key

---

### 1. Clone the Repository

```bash
git clone https://github.com/ali-ghosn/CSC7644-Final-Project-Ghosn.git
cd CSC7644-Final-Project-Ghosn
```

---

### 2. Create a Virtual Environment

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o
MAX_RETRIES=3
WORKSPACE_DIR=./workspace
LOG_LEVEL=INFO
```

> Do not commit `.env` to GitHub.

---

## Running the Application

### Phase 1 — Foundation Check

```bash
python main.py
```

This verifies:
- environment loading
- registry wiring
- schema loading
- tool registration
- logging setup

Expected output:

```text
AI Task Decomposition Copilot — Phase 1 Scaffold

✔ Environment loaded
✔ 5 tools registered

Planner and controller integration will be added in later phases.
```

---

### Full Multi-Step Execution (Phase 3)

```bash
python main.py --task "Add 3 and 5, then multiply the result by 2"
```

Expected execution flow:

```text
[Planner Output]
→ Generated structured execution plan

[Controller]
✔ Executing math.add
→ Result: 8

✔ Executing math.multiply
→ Result: 16

[Final Result]
16
```

---

## Multi-Step Execution

The controller supports passing outputs between execution steps.

Example:

```json
{
  "tool": "math.multiply",
  "arguments": {
    "a": "$prev",
    "b": 2
  }
}
```

In this example, `$prev` resolves to the output of the previous step.

---

## Running Tests

Run all tests:

```bash
pytest -v
```

Run with coverage:

```bash
pytest --cov=. --cov-report=term-missing
```

Current coverage includes:
- tool execution
- schema validation
- malformed input rejection
- file sandbox enforcement
- registry validation
- variable reference handling

---

## Supported Tools

| Tool | Description |
|---|---|
| `math.add` | Adds two numbers |
| `math.multiply` | Multiplies two numbers |
| `string.concat` | Concatenates strings |
| `datetime.now` | Returns the current timestamp |
| `file.read` | Reads sandboxed text files |

---

## Security Constraints

Implemented safeguards include:

- No shell execution
- No arbitrary Python execution
- No unrestricted filesystem access
- Sandboxed `file.read`
- Path traversal rejection
- Absolute path rejection
- Schema validation before execution
- Restricted tool allowlisting

`file.read` is restricted to:

```text
./workspace/
```

---

## Repository Organization

```text
project/
│
├── main.py              # Entry point
├── planner.py           # LLM planner and retry logic
├── controller.py        # Validation + execution controller loop
├── registry.py          # Tool registry and metadata
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── tools/
│   ├── math_tools.py
│   ├── string_tools.py
│   ├── datetime_tools.py
│   └── file_tools.py
│
├── schemas/
│   ├── __init__.py
│   ├── math_schemas.py
│   ├── string_schemas.py
│   ├── datetime_schemas.py
│   └── file_schemas.py
│
├── utils/
│   ├── validators.py
│   ├── logging_utils.py
│   └── path_utils.py
│
├── workspace/
│   └── sample.txt
│
└── tests/
    ├── test_tools.py
    └── test_validation.py
```

---

## Limitations

- Sequential execution only
- Limited MVP toolset
- No persistent memory
- No external API/tool integrations
- Text-based workflows only
- Requires OpenAI API access

These limitations were intentional in order to prioritize deterministic execution and controller-loop reliability.

---

## Future Work

Potential future improvements include:

- additional utility tools
- improved planner robustness
- lightweight retrieval for larger toolsets
- optional JSON export mode
- improved execution tracing
- limited parallel execution support
- lightweight CLI REPL mode

---

## Attributions

- OpenAI Python SDK Documentation
- jsonschema Documentation
- python-dotenv Documentation
- pytest Documentation
- PEP 8 Style Guide

All implementation code was written specifically for this project.

---

## Final Notes

This project was developed as the final project submission for:

**CSC 7644 — Applied LLM Development**  
Louisiana State University
