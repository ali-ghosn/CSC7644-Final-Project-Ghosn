"""
utils/validators.py
--------------------
JSON Schema validation helpers for the AI Task Decomposition Copilot.

This module sits between the planner output and the tool executor.
It provides two levels of validation:

    1. Plan-level validation  — checks the overall structure of the JSON
       execution plan produced by the LLM Planner (list of steps, required
       keys, correct types).

    2. Argument-level validation — validates the ``arguments`` block of a
       single step against the tool-specific JSON Schema defined in schemas/.

Both validators raise ``jsonschema.ValidationError`` on failure so that the
controller can catch a single exception type and decide whether to retry or
abort.

Dependencies:
    jsonschema — installed via requirements.txt
    schemas    — TOOL_SCHEMAS mapping (tool_name → JSON Schema dict)
"""

import json
from typing import Any

import jsonschema
from jsonschema import ValidationError, validate

from schemas import TOOL_SCHEMAS
from utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Plan-level schema ──────────────────────────────────────────────────────
# The planner is expected to return a JSON object with a single key "steps"
# whose value is a non-empty array of step objects.
# Each step must have at minimum: "tool" (string) and "arguments" (object).
# An optional "step_id" string field is allowed for reference resolution.

_PLAN_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Execution Plan",
    "description": "Top-level plan object produced by the LLM Planner.",
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "description": "Ordered list of tool execution steps.",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {
                        "type": "string",
                        "description": (
                            "Optional human-readable identifier for this step, "
                            "e.g. 'step1'.  Used for $step1 reference resolution."
                        ),
                    },
                    "tool": {
                        "type": "string",
                        "description": "Fully-qualified tool name, e.g. 'math.add'.",
                        "minLength": 1,
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Key-value argument map passed to the tool.",
                    },
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["steps"],
    "additionalProperties": False,
}


# ── Public API ─────────────────────────────────────────────────────────────

def validate_plan(plan: Any) -> None:
    """Validate the top-level structure of a planner-generated execution plan.

    This is the first validation gate in the controller loop.  It checks that
    the plan object is a dict with a ``steps`` array, and that every step
    contains at minimum a ``tool`` string and an ``arguments`` object.

    Does NOT validate individual argument values — that is handled by
    validate_step_arguments().

    Args:
        plan: The parsed JSON object (Python dict) from the planner.

    Raises:
        jsonschema.ValidationError: If the plan structure is invalid.

    Example:
        >>> validate_plan({"steps": [{"tool": "math.add", "arguments": {"a": 1, "b": 2}}]})
        # passes silently

        >>> validate_plan({"steps": []})
        ValidationError: [] is too short
    """
    try:
        validate(instance=plan, schema=_PLAN_SCHEMA)
        log.debug("Plan-level validation passed (%d steps).", len(plan.get("steps", [])))
    except ValidationError as exc:
        log.debug("Plan-level validation failed: %s", exc.message)
        raise


def validate_step_arguments(tool_name: str, arguments: dict) -> None:
    """Validate the argument block of a single step against its tool schema.

    Looks up the JSON Schema for ``tool_name`` in TOOL_SCHEMAS and validates
    the provided ``arguments`` dict.  Raises ValidationError immediately on
    the first violation.

    Args:
        tool_name:  The fully-qualified tool name, e.g. ``"math.add"``.
        arguments:  The ``arguments`` dict from the planner step.

    Raises:
        jsonschema.ValidationError: If arguments fail schema validation.
        KeyError: If tool_name is not registered in TOOL_SCHEMAS.

    Example:
        >>> validate_step_arguments("math.add", {"a": 3, "b": 5})
        # passes silently

        >>> validate_step_arguments("math.add", {"a": "three", "b": 5})
        ValidationError: 'three' is not of type 'number'
    """
    if tool_name not in TOOL_SCHEMAS:
        raise KeyError(
            f"validate_step_arguments: unknown tool '{tool_name}'. "
            f"Registered tools: {list(TOOL_SCHEMAS.keys())}"
        )

    schema = TOOL_SCHEMAS[tool_name]

    try:
        validate(instance=arguments, schema=schema)
        log.debug("Argument validation passed for tool '%s'.", tool_name)
    except ValidationError as exc:
        log.debug("Argument validation failed for tool '%s': %s", tool_name, exc.message)
        raise


def parse_plan_json(raw: str) -> dict:
    """Parse a raw JSON string into a Python dict.

    Strips common LLM formatting artefacts (markdown fences, leading/trailing
    whitespace) before parsing.  This is a best-effort helper used by the
    planner recovery loop; it does NOT perform schema validation.

    Args:
        raw: The raw string output from the LLM.

    Returns:
        A parsed Python dict.

    Raises:
        json.JSONDecodeError: If the string cannot be parsed as JSON after
                              stripping markdown artefacts.

    Example:
        >>> parse_plan_json('{"steps": [...]}')
        {'steps': [...]}

        >>> parse_plan_json('```json\n{"steps": [...]}\n```')
        {'steps': [...]}
    """
    cleaned = raw.strip()

    # Strip markdown code fences if the LLM wrapped the JSON (it shouldn't,
    # but this provides a soft recovery layer before a full retry is triggered).
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove the opening fence (```json or ```) and the closing fence.
        inner = [
            line for line in lines
            if not line.strip().startswith("```")
        ]
        cleaned = "\n".join(inner).strip()

    return json.loads(cleaned)
