"""
main.py
--------
Entry point for the AI Task Decomposition Copilot.

CSC 7644: Applied LLM Development — Final Project

Full end-to-end pipeline:

    User Input (CLI)
    → LLM Planner         planner.plan()     — OpenAI → strict JSON plan
    → Controller Loop     controller.run()   — validate → execute → output
    → Final Output        stdout (human or JSON)

Usage
-----
    python main.py "Add 3 and 5, then multiply the result by 2"
    python main.py "Get the current date and prepend it to the word Report"
    python main.py "Read the file sample.txt from the workspace"
    python main.py "Concatenate Hello and World with a space separator"

Flags
-----
    --json          Emit only the structured JSON result (machine-readable)
    --list-tools    Print all registered tools and exit
    --test-prompt   Print the constructed planner system prompt and exit
    --demo          Run a hardcoded demo plan without calling OpenAI (no API key needed)

Environment
-----------
    OPENAI_API_KEY   Required for live planner calls
    OPENAI_MODEL     Default: gpt-4o
    MAX_RETRIES      Default: 3
    LOG_LEVEL        Default: INFO
"""

import json
import os
import sys

from dotenv import load_dotenv

from controller import run
from planner import PlannerError, _build_system_prompt, plan
from registry import TOOL_METADATA
from utils.logging_utils import (
    get_logger,
    log_error,
    log_planner_output,
    log_section,
)

log = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_tool_summary() -> None:
    """Print a formatted table of all registered tools to stdout."""
    log_section("Registered Tools")
    for meta in TOOL_METADATA:
        args = ", ".join(
            f"{k}: {v['type']}" for k, v in meta["arguments"].items()
        )
        print(f"  {meta['name']:20s}  ({args})")
        print(f"    └─ {meta['description']}")
    print()


def _print_usage() -> None:
    """Print CLI usage help."""
    log_section("AI Task Decomposition Copilot  |  CSC 7644")
    _print_tool_summary()
    print('  Usage:   python main.py "<task description>"')
    print()
    print("  Examples:")
    print('    python main.py "Add 3 and 5, then multiply the result by 2"')
    print('    python main.py "Get the current date and prepend it to Report"')
    print('    python main.py "Read the file sample.txt from the workspace"')
    print()
    print("  Flags:")
    print("    --list-tools    Print all registered tools and exit")
    print("    --test-prompt   Print the planner system prompt and exit")
    print("    --demo          Run a hardcoded demo without an API key")
    print("    --json          Emit only the final structured JSON result")
    print()
    print("  Setup:   Copy .env.example → .env and set OPENAI_API_KEY")


def _run_demo() -> int:
    """Run a hardcoded multi-step demo plan without calling the OpenAI API.

    Useful for screen recordings and testing the controller pipeline when
    an API key is unavailable.  Demonstrates $prev reference resolution
    across a three-step plan.

    Returns:
        0 on success, 1 on failure.
    """
    log_section("AI Task Decomposition Copilot  |  DEMO MODE")
    print(
        "\n  This demo runs a hardcoded plan — no OpenAI API call is made.\n"
        "  Task: Add 3 and 5, multiply by 2, get today's date, build a report filename.\n"
    )

    demo_plan = {
        "steps": [
            {
                "step_id":   "step1",
                "tool":      "math.add",
                "arguments": {"a": 3, "b": 5},
            },
            {
                "step_id":   "step2",
                "tool":      "math.multiply",
                "arguments": {"a": "$prev", "b": 2},
            },
            {
                "step_id":   "step3",
                "tool":      "datetime.now",
                "arguments": {"fmt": "%Y-%m-%d"},
            },
            {
                "step_id":   "step4",
                "tool":      "string.concat",
                "arguments": {"a": "Report_", "b": "$step3"},
            },
        ]
    }

    log_planner_output(json.dumps(demo_plan, indent=2))

    output = run(demo_plan)

    log_section("Final Output")
    print(json.dumps(output, indent=2, default=str))
    return 0 if output["status"] == "success" else 1


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    """Run the AI Task Decomposition Copilot end-to-end.

    Returns:
        Exit code — 0 on success, 1 on any error.
    """
    load_dotenv()

    # ── Flags that bypass the pipeline ────────────────────────────────────────
    if "--list-tools" in sys.argv:
        _print_tool_summary()
        return 0

    if "--test-prompt" in sys.argv:
        log_section("Planner System Prompt")
        print(_build_system_prompt())
        return 0

    if "--demo" in sys.argv:
        return _run_demo()

    # ── Parse remaining args ───────────────────────────────────────────────────
    json_mode = "--json" in sys.argv
    task_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not task_args:
        _print_usage()
        return 0

    task = " ".join(task_args)

    # ── Guard: API key required for live planner ───────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        log_error(
            "OPENAI_API_KEY is not set.\n"
            "  Copy .env.example → .env and add your OpenAI API key,\n"
            "  or run --demo to test the controller without an API key."
        )
        return 1

    if not json_mode:
        log_section("AI Task Decomposition Copilot  |  CSC 7644")
        print(f"\n  Task: {task!r}\n")
        print("  Calling LLM Planner ...\n")

    # ── Phase 1: LLM Planner ───────────────────────────────────────────────────
    try:
        execution_plan = plan(task)
    except PlannerError as exc:
        log_error(
            f"Planner failed after {exc.attempts} attempt(s).\n"
            f"  Last error: {exc.last_error}"
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log_error(f"Unexpected planner error: {type(exc).__name__}: {exc}")
        return 1

    if not json_mode:
        log_planner_output(json.dumps(execution_plan, indent=2))

    # ── Phase 2: Controller Loop ───────────────────────────────────────────────
    output = run(execution_plan)

    # ── Output ─────────────────────────────────────────────────────────────────
    if json_mode:
        print(json.dumps(output, indent=2, default=str))
    else:
        log_section("Final Output")
        print(json.dumps(output, indent=2, default=str))

    return 0 if output["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
