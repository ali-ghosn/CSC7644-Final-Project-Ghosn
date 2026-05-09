# AI Task Decomposition Copilot

**CSC 7644: Applied LLM Development — Final Project**
Louisiana State University

An LLM application that converts natural language technical tasks into validated, step-by-step JSON execution plans and executes them through a sequential controller loop.

---

## Overview

When given a natural language task like:

> *"Add 3 and 5, then multiply the result by 2"*

the system:

1. Calls an **LLM Planner** (OpenAI Chat Completions API) that outputs a strict, schema-validated JSON execution plan — and nothing else
2. Passes the plan to a **Controller Loop** that resolves variable references, validates arguments against JSON Schemas, and executes each tool
3. Returns a fully structured output with per-step results, resolved argument values, and a final result

This is **not** a general autonomous agent. It is a controlled, deterministic, debuggable orchestration system with explicit failure modes and security boundaries.

---

## Key Features

- **Strict JSON-only planner** — no prose, no markdown, no chain-of-thought in LLM output
- **Retry/recovery loop** — up to 3 attempts; correction messages injected into conversation history on JSON parse error, plan schema violation, argument schema violation, or unknown tool name
- **Two-gate validation** — planner validates raw argument values; controller re-validates fully-resolved arguments with concrete types
- **Variable reference resolution** — `$prev`, `$step1`, `$step2`, etc. resolved from execution state before each tool call
- **Sandboxed file access** — `file.read` hard-restricted to `./workspace/` via 4-layer path enforcement
- **Abort-on-failure** — any step failure halts execution immediately; partial results preserved in output
- **Structured outputs** — every run returns a consistent dict with status, per-step results, and error context
- **`--demo` mode** — runs a hardcoded 4-step plan with no API key required
- **210 pytest tests** — covering tools, schemas, validators, path sandbox, planner retry logic (fully mocked), and controller execution paths

---

## Architecture

```
┌──────────────────────┐
│      User Input      │  Natural language task string
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     LLM Planner      │  openai.chat.completions (temperature=0, seed=42)
│  (Strict JSON Only)  │  System prompt built from TOOL_METADATA at runtime
│                      │  Retry loop: up to MAX_RETRIES on malformed output
└──────────┬───────────┘
           │
           │  {"steps": [{"step_id": ..., "tool": ..., "arguments": {...}}, ...]}
           ▼
┌──────────────────────────────────────────────────────────────┐
│                      Controller Loop                         │
│  For each step (sequential):                                 │
│    1. Resolve $prev / $step1 / $step2 references             │
│    2. Validate resolved arguments (JSON Schema)              │
│    3. Look up tool in registry                               │
│    4. Execute tool                                           │
│    5. Record result in execution state                       │
│    → Abort immediately on any failure                        │
└────────────────┬─────────────────────────┬───────────────────┘
                 │                         │
                 ▼                         ▼
   ┌─────────────────────┐    ┌───────────────────────┐
   │   JSON Schema       │    │   Tool Registry       │
   │   Validator         │    │   math.add            │
   │   (reject invalid)  │    │   math.multiply       │
   └─────────────────────┘    │   string.concat       │
                              │   datetime.now        │
                              │   file.read (sandbox) │
                              └──────────┬────────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │  Structured Output   │
                               │  {status, results,   │
                               │   final_result,...}  │
                               └──────────────────────┘

User Input → LLM Planner → JSON Plan → Controller Loop → Tool Executor → Final Output
```

### Component Summary

| Component | File | Responsibility |
|---|---|---|
| LLM Planner | `planner.py` | OpenAI API, system prompt, retry loop |
| Controller Loop | `controller.py` | Resolve references, validate, execute, output |
| Tool Registry | `registry.py` | Maps tool names → callables + TOOL_METADATA |
| Schema Definitions | `schemas/` | JSON Schema per tool for argument validation |
| Tool Implementations | `tools/` | Pure Python functions, no side effects |
| Path Sandbox | `utils/path_utils.py` | Enforces `./workspace/` file access boundary |
| Validators | `utils/validators.py` | Plan-level and argument-level schema validation |
| Logging Utilities | `utils/logging_utils.py` | Structured terminal output helpers |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM API | OpenAI Chat Completions API (`gpt-4` recommended) |
| Schema Validation | `jsonschema` (draft-07, raw — no frameworks) |
| Environment Config | `python-dotenv` |
| Testing | `pytest` + `pytest-cov` |
| Language | Python 3.11+ |
| Frameworks | **None** — no LangChain, no CrewAI, pure Python |

---

## Setup Instructions

### Prerequisites

- Python **3.11 or higher**
- `pip`
- An OpenAI API key — [platform.openai.com](https://platform.openai.com/account/api-keys)
- **WSL users:** Ubuntu 22.04 or 24.04 recommended (Ubuntu 26.04 has known venv packaging issues)

---

### macOS

**1. Check Python version**
```bash
python3 --version
```
If below 3.11, install via Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```

**2. Clone the repository**
```bash
git clone https://github.com/ali-ghosn/CSC7644-Final-Project-Ghosn.git
cd CSC7644-Final-Project-Ghosn
```

**3. Create and activate virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
You should see `(.venv)` in your prompt.

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

If pip is outdated:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**5. Configure environment variables**
```bash
cp .env.example .env
nano .env
```
Set your API key:
```dotenv
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=1024
MAX_RETRIES=3
WORKSPACE_DIR=./workspace
LOG_LEVEL=INFO
```
Save: `Ctrl+O`, `Enter`, `Ctrl+X`

---

### Windows (WSL — Recommended)

WSL (Windows Subsystem for Linux) gives you a Linux environment on Windows and is the recommended way to run this project on Windows.

**1. Install WSL with Ubuntu 24.04**

Open PowerShell as Administrator:
```powershell
wsl --install -d Ubuntu-24.04
```
Restart your machine if prompted. Then open "Ubuntu 24.04" from the Start Menu.

> Do NOT use Ubuntu 26.04 — it has known Python venv packaging issues.

**2. Install Python venv support**
```bash
sudo apt update
sudo apt install python3-venv python3-pip -y
```

Common issue — if you see `"Package python3.12-venv is not available"`:
```bash
sudo apt install python3-venv python3-pip -y
```
Use `python3-venv` without the version number.

**3. Clone the repository**
```bash
cd ~
mkdir projects && cd projects
git clone https://github.com/ali-ghosn/CSC7644-Final-Project-Ghosn.git
cd CSC7644-Final-Project-Ghosn
```

If git is not installed:
```bash
sudo apt install git -y
```

**4. Create and activate virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
You should see `(.venv)` in your prompt.

If venv creation fails with "ensurepip is not available":
```bash
sudo apt install python3-venv -y
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

**5. Install dependencies**
```bash
pip install -r requirements.txt
```

**6. Configure environment variables**
```bash
cp .env.example .env
nano .env
```
Set your API key:
```dotenv
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=1024
MAX_RETRIES=3
WORKSPACE_DIR=./workspace
LOG_LEVEL=INFO
```
Save: `Ctrl+O`, `Enter`, `Ctrl+X`

> Note: When uploading `.env.example` to GitHub, the browser may warn "This file is hidden." Click Upload anyway — it will appear in the repo normally.

---

### Windows (Native — without WSL)

**1. Install Python 3.11+**

Download from [python.org/downloads](https://www.python.org/downloads/).
During installation, check **"Add Python to PATH"**.

Verify:
```cmd
python --version
```

**2. Clone the repository**
```cmd
git clone https://github.com/ali-ghosn/CSC7644-Final-Project-Ghosn.git
cd CSC7644-Final-Project-Ghosn
```

If git is not installed, download from [git-scm.com](https://git-scm.com/download/win).

**3. Create and activate virtual environment**
```cmd
python -m venv .venv
.venv\Scripts\activate
```
You should see `(.venv)` in your prompt.

If activation is blocked by execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then retry activation.

**4. Install dependencies**
```cmd
pip install -r requirements.txt
```

**5. Configure environment variables**

Copy `.env.example` to `.env` and open it in Notepad or any text editor:
```cmd
copy .env.example .env
notepad .env
```
Set your API key and save.

---

### Linux (Ubuntu/Debian)

**1. Install Python and venv**
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
```

**2. Clone the repository**
```bash
git clone https://github.com/ali-ghosn/CSC7644-Final-Project-Ghosn.git
cd CSC7644-Final-Project-Ghosn
```

**3. Create and activate virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Configure environment variables**
```bash
cp .env.example .env
nano .env
```
Set your API key, save with `Ctrl+O`, `Enter`, `Ctrl+X`.

---

## Running the Application

### Reactivating your environment (every new terminal session)

**macOS / Linux / WSL:**
```bash
source .venv/bin/activate
```
**Windows:**
```cmd
.venv\Scripts\activate
```

---

### Demo mode — no API key required

Runs a hardcoded 4-step plan through the full controller pipeline. Use this to verify installation and for screen recordings.

```bash
python main.py --demo
```

Expected output:
```
────────────────────────────────────────────────────────────
  AI Task Decomposition Copilot  |  DEMO MODE
────────────────────────────────────────────────────────────

  Task: Add 3 and 5, multiply by 2, get today's date, build a report filename.

────────────────────────────────────────────────────────────
  Planner Output — Execution Plan
────────────────────────────────────────────────────────────
{
  "steps": [
    {"step_id": "step1", "tool": "math.add",      "arguments": {"a": 3, "b": 5}},
    {"step_id": "step2", "tool": "math.multiply",  "arguments": {"a": "$prev", "b": 2}},
    {"step_id": "step3", "tool": "datetime.now",   "arguments": {"fmt": "%Y-%m-%d"}},
    {"step_id": "step4", "tool": "string.concat",  "arguments": {"a": "Report_", "b": "$step3"}}
  ]
}

────────────────────────────────────────────────────────────
  Controller Loop
────────────────────────────────────────────────────────────
  ✔  Step 1: executing math.add
  ✔  Schema validation passed for math.add
     → Result: 8

  ✔  Step 2: executing math.multiply
  ✔  Schema validation passed for math.multiply
     → Result: 16

  ✔  Step 3: executing datetime.now
  ✔  Schema validation passed for datetime.now
     → Result: '2026-05-08'

  ✔  Step 4: executing string.concat
  ✔  Schema validation passed for string.concat
     → Result: 'Report_2026-05-08'

────────────────────────────────────────────────────────────
  Execution Complete
────────────────────────────────────────────────────────────

  ✔  All 4 step(s) completed successfully.
  ✔  Final result: 'Report_2026-05-08'
```

---

### Live mode — requires `OPENAI_API_KEY`

```bash
python main.py "Add 3 and 5, then multiply the result by 2"
python main.py "Get the current date and prepend it to the word Report"
python main.py "Read the file sample.txt from the workspace"
python main.py "Concatenate Hello and World with a space between them"
```

### Other flags

```bash
python main.py --list-tools     # Print all registered tools
python main.py --test-prompt    # Print the planner system prompt
python main.py --json "Add 10 and 5"   # Emit only structured JSON output
```

---

## Running Tests

```bash
# All tests, verbose
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. -q

# Individual test modules
pytest tests/test_tools.py       -v   # 38 tests — tool implementations
pytest tests/test_validation.py  -v   # 46 tests — schemas, validators, sandbox, registry
pytest tests/test_planner.py     -v   # 57 tests — planner retry logic (fully mocked)
pytest tests/test_controller.py  -v   # 69 tests — controller execution paths
```

**Current status: 210 tests, 210 passing. Zero real API calls made during testing.**

If `pytest` is not found:
```bash
pip install pytest pytest-cov
```

---

## Supported Tools

| Tool | Arguments | Description |
|---|---|---|
| `math.add` | `a: number, b: number` | Returns `a + b` |
| `math.multiply` | `a: number, b: number` | Returns `a * b` |
| `string.concat` | `a: string, b: string, separator?: string` | Returns `a + separator + b` |
| `datetime.now` | `fmt?: string` | Current timestamp (default: `%Y-%m-%d %H:%M:%S`) |
| `file.read` | `path: string` | Reads a UTF-8 text file from `./workspace/` only |

### Variable references in arguments

| Syntax | Resolves to |
|---|---|
| `"$prev"` | Output of the immediately preceding step |
| `"$step1"` | Output of the step with `step_id: "step1"` |
| `"$step2"` | Output of the step with `step_id: "step2"` |

---

## Security

| Constraint | Enforcement layer |
|---|---|
| No shell execution | Not implemented; not in registry |
| No arbitrary code execution | Not implemented; not in registry |
| `file.read` confined to `./workspace/` | `utils/path_utils.resolve_safe_path()` |
| Absolute paths rejected | Schema pattern check + `path_utils` guard |
| Path traversal (`../`) rejected | String check + canonical realpath boundary check |
| Symlink escapes rejected | `os.path.realpath()` resolves all links before boundary check |

---

## Repository Organization

```
CSC7644-Final-Project-Ghosn/
│
├── main.py               Entry point — CLI, planner + controller pipeline
├── planner.py            LLM Planner — OpenAI API, system prompt, retry loop
├── controller.py         Controller Loop — resolve, validate, execute, output
├── registry.py           Tool registry and TOOL_METADATA for prompt generation
├── conftest.py           Pytest root configuration
├── requirements.txt      Python dependencies
├── .env.example          Environment variable template (copy to .env)
├── .gitignore
├── README.md
│
├── tools/
│   ├── __init__.py
│   ├── math_tools.py     math.add, math.multiply
│   ├── string_tools.py   string.concat
│   ├── datetime_tools.py datetime.now
│   └── file_tools.py     file.read (sandboxed, delegates to path_utils)
│
├── schemas/
│   ├── __init__.py       TOOL_SCHEMAS dict — tool_name → JSON Schema
│   ├── math_schemas.py   Schemas for math.add, math.multiply
│   ├── string_schemas.py Schema for string.concat
│   ├── datetime_schemas.py Schema for datetime.now
│   └── file_schemas.py   Schema for file.read
│
├── utils/
│   ├── __init__.py
│   ├── validators.py     validate_plan(), validate_step_arguments(), parse_plan_json()
│   ├── logging_utils.py  ANSI terminal output helpers (TTY-aware)
│   └── path_utils.py     4-layer workspace sandbox enforcement
│
├── workspace/
│   ├── sample.txt        Demo file for file.read tool
│   └── notes.txt         Architecture notes
│
└── tests/
    ├── __init__.py
    ├── test_tools.py         Tool implementation tests (38)
    ├── test_validation.py    Schema + path sandbox + registry tests (46)
    ├── test_planner.py       Planner retry logic tests — fully mocked (57)
    └── test_controller.py    Controller execution tests (69)
```

---

## Limitations

- **Sequential execution only** — steps cannot run in parallel (by design)
- **5 MVP tools only** — no HTTP, database, or computation-heavy tools
- **Text files only** — `file.read` reads UTF-8; binary files not supported
- **Single-machine local time** — `datetime.now` uses system local time
- **Stateless** — each `main.py` invocation starts fresh; no memory between runs
- **Type bridging** — numeric tool outputs cannot be directly passed to string tools; the schema validator correctly rejects this
- **OpenAI dependency** — live mode requires a valid API key and network access
- **Model sensitivity** — `gpt-4` is recommended; `gpt-4o` may produce inconsistent JSON formatting

---

## Future Work

- Add `math.subtract`, `math.divide`, `string.to_upper`, `string.to_int` tools
- Add a `web.fetch` tool with an explicit domain allowlist
- Implement conditional branching based on prior step results
- Add cost tracking (token count + estimated cost per plan)
- Add a simple REPL mode for interactive multi-turn task input
- Build a minimal Streamlit UI for richer demo presentations
- Support parallel execution of independent step branches

---

## Attributions

- [OpenAI Python SDK](https://platform.openai.com/docs/api-reference) — API integration patterns
- [jsonschema](https://python-jsonschema.readthedocs.io/) — JSON Schema validation
- [python-dotenv](https://github.com/theskumar/python-dotenv) — environment variable management
- [PEP 8](https://peps.python.org/pep-0008/) — code style standard followed throughout

All implementations are original. No external code was adapted or copied.

---

*CSC 7644: Applied LLM Development — Louisiana State University*
