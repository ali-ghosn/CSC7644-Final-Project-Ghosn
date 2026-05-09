"""
planner.py
-----------
LLM Planner for the AI Task Decomposition Copilot.

Responsibility
--------------
Convert a natural language task string into a validated JSON execution plan
by calling the OpenAI Chat Completions API under strict constraints:

    - The model MUST output only raw JSON — no prose, no markdown, no reasoning.
    - The system prompt is built at runtime from TOOL_METADATA so it stays
      in sync with the registry automatically.
    - Raw JSON Schemas are NEVER sent to the model — only human-readable
      tool descriptions, argument types, and examples.

Retry / Recovery Loop
---------------------
The planner implements a 3-attempt recovery loop (configurable via MAX_RETRIES):

    Attempt 1 → send system prompt + user task
    Attempt N → send the prior conversation history + a correction message
                 that explains exactly what went wrong (parse error or schema
                 violation) and asks the model to fix only that issue.

Failure modes handled per attempt:
    - json.JSONDecodeError       → prompt the model to return valid JSON only
    - jsonschema.ValidationError → prompt the model to fix the specific field
    - KeyError (unknown tool)    → prompt the model to use only listed tool names
    - openai.APIError            → re-raised immediately (not retried)

If all attempts are exhausted a PlannerError is raised with a full
diagnostic message.

Configuration (via .env)
------------------------
    OPENAI_API_KEY       — required
    OPENAI_MODEL         — default: gpt-4o
    OPENAI_MAX_TOKENS    — default: 1024
    MAX_RETRIES          — default: 3
"""

import json
import os
from typing import Any

import openai
from jsonschema import ValidationError

from registry import TOOL_METADATA, list_tools
from utils.logging_utils import get_logger, log_retry
from utils.validators import parse_plan_json, validate_plan, validate_step_arguments

log = get_logger(__name__)


# ── Custom exception ───────────────────────────────────────────────────────

class PlannerError(RuntimeError):
    """Raised when the planner fails to produce a valid plan after all retries.

    Attributes:
        attempts:   Number of attempts made before giving up.
        last_raw:   The last raw string returned by the model (may be None).
        last_error: The last exception that caused the failure.
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_raw: str | None = None,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_raw = last_raw
        self.last_error = last_error


# ── Configuration helpers ──────────────────────────────────────────────────

def _get_model() -> str:
    """Return the OpenAI model name from the environment."""
    return os.environ.get("OPENAI_MODEL", "gpt-4o")


def _get_max_tokens() -> int:
    """Return the max_tokens setting from the environment."""
    try:
        return int(os.environ.get("OPENAI_MAX_TOKENS", "1024"))
    except ValueError:
        return 1024


def _get_max_retries() -> int:
    """Return the maximum retry count from the environment."""
    try:
        return max(1, int(os.environ.get("MAX_RETRIES", "3")))
    except ValueError:
        return 3


# ── Prompt builders ────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """Build the planner system prompt from the live tool registry.

    The prompt is reconstructed on every call so it automatically reflects
    any tools added to or removed from the registry.  Raw JSON Schemas are
    never exposed — only human-readable descriptions and examples.

    Returns:
        A complete system prompt string.
    """
    tool_lines: list[str] = []

    for meta in TOOL_METADATA:
        # Format each argument: name, type, required/optional, description
        arg_parts: list[str] = []
        for arg_name, arg_info in meta["arguments"].items():
            req = "required" if arg_info.get("required", True) else "optional"
            desc = arg_info.get("description", "")
            arg_parts.append(
                f'    "{arg_name}": {arg_info["type"]} ({req}) — {desc}'
            )

        args_block = "\n".join(arg_parts) if arg_parts else "    (no arguments)"
        example_json = json.dumps(meta["example"])
        example_out = repr(meta["example_output"])

        tool_lines.append(
            f'TOOL: {meta["name"]}\n'
            f'  Description: {meta["description"]}\n'
            f'  Arguments:\n{args_block}\n'
            f'  Example arguments: {example_json}\n'
            f'  Example output: {example_out}'
        )

    tools_block = "\n\n".join(tool_lines)
    tool_names = ", ".join(list_tools())

    return (
        "You are a task decomposition engine. Your ONLY job is to convert a natural "
        "language technical task into a JSON execution plan.\n\n"

        "ABSOLUTE OUTPUT RULES — VIOLATIONS WILL CAUSE SYSTEM FAILURE:\n"
        "1. Output ONLY raw JSON. No prose. No markdown. No code fences. No explanation.\n"
        "2. Your entire response must be a single JSON object parseable with json.loads().\n"
        "3. Do NOT include reasoning, chain-of-thought, or commentary of any kind.\n"
        "4. Do NOT wrap the JSON in ```json or ``` blocks.\n\n"

        "OUTPUT FORMAT — always return exactly this structure:\n"
        '{\n'
        '  "steps": [\n'
        '    {\n'
        '      "step_id": "step1",\n'
        '      "tool": "<tool_name>",\n'
        '      "arguments": { "<arg>": <value>, ... }\n'
        '    }\n'
        '  ]\n'
        '}\n\n'

        "RULES FOR STEPS:\n"
        '- "step_id" must be a string: "step1", "step2", etc. (required on every step)\n'
        '- "tool" must be one of the registered tool names listed below\n'
        '- "arguments" must contain only the argument keys defined for that tool\n'
        '- Use "$prev" to reference the output of the immediately preceding step\n'
        '- Use "$step1", "$step2", etc. to reference a specific named step output\n'
        "- Argument values must match the expected types unless using a reference\n\n"

        f"REGISTERED TOOLS:\n{tools_block}\n\n"
        f"VALID TOOL NAMES: {tool_names}\n"
    )


def _build_correction_message(
    error_type: str, detail: str, raw_response: str
) -> str:
    """Build a recovery prompt explaining exactly what went wrong.

    Appended to the conversation history so the model has full context
    when generating its corrected response.

    Args:
        error_type:   Short label, e.g. "INVALID JSON" or "ARGUMENT SCHEMA ERROR".
        detail:       The specific error text from the parser or validator.
        raw_response: The model's last raw output (reflected back for correction).

    Returns:
        A user-turn correction message string.
    """
    return (
        f"Your previous response caused a {error_type}.\n\n"
        f"Error detail: {detail}\n\n"
        f"Your previous response was:\n{raw_response}\n\n"
        "Fix ONLY the reported problem and return the corrected JSON object. "
        "No explanation. No markdown. No code fences. Raw JSON only."
    )


# ── OpenAI API call ────────────────────────────────────────────────────────

def _call_openai(messages: list[dict[str, str]]) -> str:
    """Make a single OpenAI Chat Completions API call and return the raw text.

    Uses ``response_format={"type": "json_object"}`` to instruct the API to
    return valid JSON.  This is a soft guarantee from OpenAI — the planner
    still validates the content structure independently.

    Sets temperature=0 and seed=42 for maximum determinism.

    Args:
        messages: Full conversation history in OpenAI message format.

    Returns:
        The raw content string from the model's first completion choice.

    Raises:
        openai.AuthenticationError: For invalid or missing API keys.
        openai.APIError:            For other API-level failures.
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=_get_model(),
        max_tokens=_get_max_tokens(),
        # json_object mode enforces syntactic JSON validity at the API level.
        # Our validators still check structural and semantic correctness.
        response_format={"type": "json_object"},
        messages=messages,      # type: ignore[arg-type]
        temperature=0,          # Deterministic: same prompt → same plan
        seed=42,                # Additional reproducibility hint (best-effort)
    )

    raw = response.choices[0].message.content
    log.debug("Raw planner response:\n%s", raw)
    return raw or ""


# ── Internal step validation helper ───────────────────────────────────────

def _is_reference(value: object) -> bool:
    """Return True if value is a variable reference like '$prev' or '$step1'.

    References are resolved by the controller at execution time, so their
    concrete type is unknown at planning time — they must bypass type checks.
    """
    return isinstance(value, str) and value.startswith("$")


def _strip_references(arguments: dict) -> dict:
    """Return a copy of arguments with all $-reference values removed.

    Only reference-valued keys are omitted.  Concrete values are preserved
    for full schema validation.
    """
    return {k: v for k, v in arguments.items() if not _is_reference(v)}


def _validate_all_steps(plan: dict) -> None:
    """Run argument-level validation on every step in an already plan-validated dict.

    Reference values ($prev, $step1, etc.) are excluded from schema validation
    because their concrete type is only resolved by the controller at runtime.
    The corresponding keys are also removed from the schema's 'required' list so
    that jsonschema does not flag them as missing.

    Args:
        plan: A dict that has already passed validate_plan().

    Raises:
        KeyError:                   If a step references an unregistered tool.
        jsonschema.ValidationError: If any step's concrete arguments fail their schema.
    """
    from schemas import TOOL_SCHEMAS
    import copy

    for i, step in enumerate(plan["steps"], start=1):
        tool_name = step["tool"]
        arguments = step["arguments"]
        log.debug("Validating step %d arguments for tool '%s'.", i, tool_name)

        # Identify which argument keys hold reference strings.
        ref_keys = {k for k, v in arguments.items() if _is_reference(v)}

        if not ref_keys:
            # No references — validate the full argument dict as-is.
            validate_step_arguments(tool_name, arguments)
        else:
            # Validate only the concrete (non-reference) arguments.
            concrete_args = _strip_references(arguments)

            if tool_name not in TOOL_SCHEMAS:
                raise KeyError(
                    f"validate_step_arguments: unknown tool '{tool_name}'."
                )

            # Deep-copy the schema so we can safely remove ref keys from 'required'.
            schema = copy.deepcopy(TOOL_SCHEMAS[tool_name])
            if "required" in schema:
                schema["required"] = [
                    k for k in schema["required"] if k not in ref_keys
                ]

            from jsonschema import validate as jvalidate
            jvalidate(instance=concrete_args, schema=schema)


# ── Public API ─────────────────────────────────────────────────────────────

def plan(task: str) -> dict[str, Any]:
    """Convert a natural language task string into a validated JSON execution plan.

    This is the primary public interface of the planner module.

    Calls the OpenAI Chat Completions API with a strict JSON-only system prompt,
    then validates the response at two levels:

        1. Plan structure   — via validate_plan() (steps array, required keys)
        2. Step arguments   — via validate_step_arguments() (per-tool schemas)

    On any validation failure the planner appends a correction message to the
    conversation history and retries, up to MAX_RETRIES attempts total.

    Args:
        task: Natural language description of the task to decompose, e.g.:
              "Add 3 and 5, then multiply the result by 2."

    Returns:
        A validated execution plan dict:
        {
            "steps": [
                {
                    "step_id": "step1",
                    "tool": "math.add",
                    "arguments": {"a": 3, "b": 5}
                },
                {
                    "step_id": "step2",
                    "tool": "math.multiply",
                    "arguments": {"a": "$prev", "b": 2}
                }
            ]
        }

    Raises:
        ValueError:                   If task is not a non-empty string.
        PlannerError:                 If a valid plan cannot be produced after
                                      MAX_RETRIES attempts.
        openai.AuthenticationError:   If the API key is invalid/missing.
        openai.APIError:              For other unrecoverable API failures.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("plan(): 'task' must be a non-empty string.")

    max_retries = _get_max_retries()

    # Initial conversation: system prompt + first user task message.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user",   "content": task.strip()},
    ]

    last_raw: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        log.debug("Planner attempt %d/%d.", attempt, max_retries)

        # ── 1. Call OpenAI ───────────────────────────────────────────────────
        try:
            raw = _call_openai(messages)
        except (openai.AuthenticationError, openai.PermissionDeniedError):
            # Configuration errors — retrying will not fix these.
            # Re-raise the original exception directly; constructing a new
            # AuthenticationError requires response/body kwargs in openai>=1.x.
            raise
        except openai.APIError:
            # Surface other API failures directly.
            raise

        last_raw = raw

        # ── 2. Parse JSON ────────────────────────────────────────────────────
        try:
            parsed = parse_plan_json(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            detail = f"JSONDecodeError at position {exc.pos}: {exc.msg}"
            log.warning("Attempt %d/%d — JSON parse failed: %s", attempt, max_retries, detail)

            if attempt < max_retries:
                log_retry(attempt, max_retries, f"malformed JSON — {exc.msg}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": _build_correction_message("INVALID JSON", detail, raw),
                })
            continue

        # ── 3. Plan-level structure validation ───────────────────────────────
        try:
            validate_plan(parsed)
        except ValidationError as exc:
            last_error = exc
            detail = exc.message
            log.warning(
                "Attempt %d/%d — plan schema violation: %s", attempt, max_retries, detail
            )

            if attempt < max_retries:
                log_retry(attempt, max_retries, f"plan schema — {detail}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": _build_correction_message(
                        "PLAN SCHEMA ERROR", detail, raw
                    ),
                })
            continue

        # ── 4. Per-step argument validation ──────────────────────────────────
        try:
            _validate_all_steps(parsed)
        except KeyError as exc:
            last_error = exc
            detail = str(exc)
            log.warning(
                "Attempt %d/%d — unknown tool name: %s", attempt, max_retries, detail
            )

            if attempt < max_retries:
                log_retry(attempt, max_retries, f"unknown tool — {detail}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": _build_correction_message(
                        "UNKNOWN TOOL NAME", detail, raw
                    ),
                })
            continue

        except ValidationError as exc:
            last_error = exc
            detail = exc.message
            log.warning(
                "Attempt %d/%d — argument schema violation: %s",
                attempt, max_retries, detail,
            )

            if attempt < max_retries:
                log_retry(attempt, max_retries, f"argument schema — {detail}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": _build_correction_message(
                        "ARGUMENT SCHEMA ERROR", detail, raw
                    ),
                })
            continue

        # ── All validation passed ─────────────────────────────────────────────
        log.info(
            "Planner succeeded on attempt %d/%d — %d step(s) generated.",
            attempt, max_retries, len(parsed["steps"]),
        )
        return parsed

    # ── All attempts exhausted ─────────────────────────────────────────────
    raise PlannerError(
        f"Planner failed to produce a valid plan after {max_retries} attempt(s).\n"
        f"Last error type : {type(last_error).__name__}\n"
        f"Last error      : {last_error}\n"
        f"Last raw output :\n{last_raw}",
        attempts=max_retries,
        last_raw=last_raw,
        last_error=last_error,
    )
