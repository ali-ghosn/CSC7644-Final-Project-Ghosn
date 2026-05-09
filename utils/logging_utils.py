"""
utils/logging_utils.py
-----------------------
Structured terminal logging helpers for the AI Task Decomposition Copilot.

CSC 7644: Applied LLM Development — Final Project

Design goals
------------
- Human-readable, visually consistent output for demos and screen recordings.
- ANSI colour support on TTY terminals; gracefully degrades to plain text
  when piped or redirected (e.g. CI, log files).
- Zero external dependencies — stdlib ``logging`` + ``sys`` only.
- Log level controlled by the LOG_LEVEL environment variable (.env).

Two output layers
-----------------
1. Python logging (get_logger)
   Used for DEBUG/INFO/WARNING/ERROR messages inside modules.
   Respects LOG_LEVEL; suppressed when LOG_LEVEL=WARNING or higher.

2. Controller print helpers (log_section, log_step, log_result, etc.)
   Write directly to stdout/stderr so they always appear in screen
   recordings, regardless of LOG_LEVEL.

Usage:
    from utils.logging_utils import get_logger, log_section, log_step

    log = get_logger(__name__)
    log.info("Planner attempt 1/3")

    log_section("Controller Loop")
    log_step(1, "math.add")
    log_result(1, 8)
"""

import logging
import os
import sys
from typing import Any

# ── ANSI colour codes ──────────────────────────────────────────────────────────
# Disabled automatically when stdout is not a TTY (piped output, CI, log files).
_IS_TTY = sys.stdout.isatty()

_GREEN  = "\033[92m" if _IS_TTY else ""
_YELLOW = "\033[93m" if _IS_TTY else ""
_RED    = "\033[91m" if _IS_TTY else ""
_CYAN   = "\033[96m" if _IS_TTY else ""
_BOLD   = "\033[1m"  if _IS_TTY else ""
_RESET  = "\033[0m"  if _IS_TTY else ""

# ── Python logging configuration ───────────────────────────────────────────────

_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG":   logging.DEBUG,
    "INFO":    logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR":   logging.ERROR,
}

_DEFAULT_LOG_LEVEL = "INFO"


def _resolve_log_level() -> int:
    """Read LOG_LEVEL from the environment and return the stdlib int constant."""
    raw = os.environ.get("LOG_LEVEL", _DEFAULT_LOG_LEVEL).upper()
    return _LOG_LEVEL_MAP.get(raw, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a configured Logger instance for the given module name.

    Attaches a stdout StreamHandler with a readable format if the logger
    has no handlers yet.  Safe to call multiple times (idempotent).

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(_resolve_log_level())
        fmt = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.setLevel(_resolve_log_level())
    return logger


# ── Controller-level print helpers ─────────────────────────────────────────────
# These write directly to stdout so they always appear in demos,
# regardless of the LOG_LEVEL environment variable.

def log_section(title: str) -> None:
    """Print a bold section header divider.

    Used for major pipeline phases: [Planner Output], [Controller Loop], etc.

    Args:
        title: Section title string.
    """
    print(f"\n{_BOLD}{_CYAN}{'─' * 60}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * 60}{_RESET}")


def log_step(step_index: int, tool_name: str) -> None:
    """Print a step execution header with a green checkmark.

    Args:
        step_index: 1-based position of the current step.
        tool_name:  The fully-qualified tool name, e.g. ``"math.add"``.
    """
    print(f"  {_GREEN}✔{_RESET}  Step {step_index}: executing {_BOLD}{tool_name}{_RESET}")


def log_result(step_index: int, result: Any) -> None:
    """Print the result returned by a tool invocation.

    Args:
        step_index: 1-based position of the completed step.
        result:     The value returned by the tool function.
    """
    print(f"     {_YELLOW}→ Result:{_RESET} {result!r}")


def log_error(message: str) -> None:
    """Print a bold red error message to stderr.

    Args:
        message: Human-readable description of the failure.
    """
    print(f"\n  {_RED}{_BOLD}✘ ERROR:{_RESET} {_RED}{message}{_RESET}", file=sys.stderr)


def log_validation_pass(tool_name: str) -> None:
    """Print a schema validation success message.

    Args:
        tool_name: The tool whose argument schema just passed.
    """
    print(
        f"  {_GREEN}✔{_RESET}  Schema validation passed for {_BOLD}{tool_name}{_RESET}"
    )


def log_validation_fail(tool_name: str, error_detail: str) -> None:
    """Print a schema validation failure message to stderr.

    Args:
        tool_name:    The tool whose argument schema failed.
        error_detail: The specific jsonschema error message.
    """
    print(
        f"  {_RED}✘{_RESET}  Schema validation FAILED for {_BOLD}{tool_name}{_RESET}: "
        f"{error_detail}",
        file=sys.stderr,
    )


def log_retry(attempt: int, max_retries: int, reason: str) -> None:
    """Print a planner retry warning.

    Args:
        attempt:     Current attempt number (1-based).
        max_retries: Maximum allowed attempts.
        reason:      Short description of why the retry was triggered.
    """
    print(f"\n  {_YELLOW}↺  Retry {attempt}/{max_retries}:{_RESET} {reason}")


def log_planner_output(raw_json: str) -> None:
    """Print the raw planner JSON output in a clearly labelled block.

    Used by main.py to display the execution plan before the controller runs.

    Args:
        raw_json: Pretty-printed JSON string of the execution plan.
    """
    log_section("Planner Output — Execution Plan")
    print(raw_json)
