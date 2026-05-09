"""
controller.py
--------------
Controller Loop for the AI Task Decomposition Copilot.

CSC 7644: Applied LLM Development — Final Project

Responsibility
--------------
Accept a validated JSON execution plan from the planner and execute it
step-by-step under strict sequential, synchronous constraints:

    1. Resolve variable references ($prev, $step1, $step2, ...) in arguments
       by substituting the recorded output of prior steps.
    2. Re-validate the fully-resolved arguments against the tool's JSON Schema.
       (Second validation gate — planner validated with raw values; controller
       validates again post-resolution with concrete types.)
    3. Execute the tool callable via the registry using keyword arguments.
    4. Record the result in execution state.
    5. On any failure: stop immediately and return a structured failure output.
       Nothing is swallowed silently.

Execution model
---------------
- Sequential only: steps execute in plan order, one at a time.
- Synchronous only: no threads, no async, no parallelism.
- Abort-on-failure: any exception in any step halts the entire run.
- No retries at execution time: tool failures surface directly to the caller.
  (Retry logic lives in the planner for malformed plan outputs only.)

Reference resolution
--------------------
Supported syntax for $-reference values in step arguments:

    "$prev"    → output of the immediately preceding step
    "$step1"   → output of the step whose step_id is "step1"
    "$step2"   → output of the step whose step_id is "step2"
    (etc.)

References are resolved before schema validation, so the validator always
sees concrete typed values, not placeholder strings.

Output format
-------------
run() always returns a dict, never raises on tool failure:

    Success:
    {
        "status":          "success",
        "steps_completed": 2,
        "steps_total":     2,
        "results": [
            {
                "step_index": 1,
                "step_id":    "step1",
                "tool":       "math.add",
                "arguments":  {"a": 3, "b": 5},
                "result":     8
            },
            {
                "step_index": 2,
                "step_id":    "step2",
                "tool":       "math.multiply",
                "arguments":  {"a": 8, "b": 2},
                "result":     16
            }
        ],
        "final_result": 16,
        "error":        null
    }

    Failure (hard stop at failing step):
    {
        "status":          "failure",
        "steps_completed": 1,
        "steps_total":     2,
        "results":         [ ... completed steps ... ],
        "final_result":    8,
        "error": {
            "step_index": 2,
            "step_id":    "step2",
            "tool":       "math.multiply",
            "type":       "ValidationError",
            "message":    "..."
        }
    }
"""

from typing import Any

from jsonschema import ValidationError

from registry import get_tool
from utils.logging_utils import (
    get_logger,
    log_error,
    log_result,
    log_section,
    log_step,
    log_validation_fail,
    log_validation_pass,
)
from utils.validators import validate_step_arguments

log = get_logger(__name__)


# ── Custom exception ───────────────────────────────────────────────────────────

class ControllerError(RuntimeError):
    """Raised when the controller cannot complete execution of a step.

    Wraps any error that occurs during reference resolution, schema validation,
    registry lookup, or tool execution into a single consistent type so the
    run() loop can handle all failures through one catch clause.

    Attributes:
        step_index: 1-based index of the failing step in the plan.
        step_id:    The step_id string from the plan (None if omitted).
        tool_name:  The tool name of the failing step.
        cause:      The underlying exception (may be None).
    """

    def __init__(
        self,
        message: str,
        step_index: int,
        step_id: str | None,
        tool_name: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.step_index = step_index
        self.step_id    = step_id
        self.tool_name  = tool_name
        self.cause      = cause


# ── Execution state ────────────────────────────────────────────────────────────

class ExecutionState:
    """Tracks per-step outputs for $-reference resolution.

    Maintains two parallel indexes:
        _by_step_id — maps "step1", "step2", etc. → result (for $step1/$step2)
        _ordered    — ordered list of all results   (for $prev)

    Usage:
        state = ExecutionState()
        state.record("step1", 8)
        state.resolve("$prev")    # → 8
        state.resolve("$step1")   # → 8
    """

    def __init__(self) -> None:
        self._by_step_id: dict[str, Any] = {}
        self._ordered: list[Any] = []

    def record(self, step_id: str | None, result: Any) -> None:
        """Record the output of a completed step.

        Args:
            step_id: The step's step_id string, or None if the step has no ID.
            result:  The value returned by the tool callable.
        """
        self._ordered.append(result)
        if step_id:
            self._by_step_id[step_id] = result

    def resolve(self, reference: str) -> Any:
        """Resolve a $-reference string to its recorded concrete value.

        Args:
            reference: A reference string, e.g. "$prev" or "$step1".

        Returns:
            The recorded result value for that reference.

        Raises:
            KeyError:   If $prev is used before any step has run, or if the
                        named step ID has not been recorded yet.
            ValueError: If the string is not a valid $-reference.
        """
        if reference == "$prev":
            if not self._ordered:
                raise KeyError(
                    "'$prev' was used in the first step — no previous result exists."
                )
            return self._ordered[-1]

        if reference.startswith("$"):
            step_id = reference[1:]   # strip leading "$"
            if step_id not in self._by_step_id:
                recorded = list(self._by_step_id.keys())
                raise KeyError(
                    f"Reference '{reference}' cannot be resolved. "
                    f"Recorded step IDs so far: {recorded}"
                )
            return self._by_step_id[step_id]

        raise ValueError(f"Not a valid $-reference: '{reference}'")

    @property
    def last_result(self) -> Any:
        """Return the most recently recorded result, or None if no steps have run."""
        return self._ordered[-1] if self._ordered else None

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the step_id → result mapping (for structured output)."""
        return dict(self._by_step_id)


# ── Reference helpers ──────────────────────────────────────────────────────────

def _is_reference(value: object) -> bool:
    """Return True if value is a $-reference placeholder string."""
    return isinstance(value, str) and value.startswith("$")


def resolve_arguments(arguments: dict, state: ExecutionState) -> dict:
    """Replace all $-reference values in an arguments dict with concrete values.

    Only top-level argument values are examined.  Nested structures are not
    recursively resolved — the MVP tool surface does not require this.

    Args:
        arguments: Raw arguments dict from a plan step (may contain $-refs).
        state:     Current execution state with recorded step results.

    Returns:
        A new dict with every $-reference replaced by its resolved value.
        Concrete (non-reference) values are passed through unchanged.

    Raises:
        KeyError: If any $-reference cannot be resolved from the current state.

    Example:
        >>> state = ExecutionState()
        >>> state.record("step1", 8)
        >>> resolve_arguments({"a": "$prev", "b": 2}, state)
        {"a": 8, "b": 2}
    """
    resolved: dict[str, Any] = {}
    for key, value in arguments.items():
        resolved[key] = state.resolve(value) if _is_reference(value) else value
    return resolved


# ── Per-step execution ─────────────────────────────────────────────────────────

def _execute_step(
    step: dict,
    step_index: int,
    state: ExecutionState,
) -> tuple[Any, dict]:
    """Execute one plan step and return (result, resolved_arguments).

    Returns both the tool's output and the fully-resolved argument dict.
    The resolved_arguments are captured *before* state.record() is called,
    ensuring that $prev in the output record reflects the *prior* step's
    value, not the current step's.

    Pipeline per step:
        1. Resolve $-references against current state
        2. Validate resolved arguments against the tool's JSON Schema
        3. Look up the tool callable in the registry
        4. Invoke the tool with keyword arguments
        5. Record the result in state

    Args:
        step:       Step dict with keys: tool, arguments, (optional) step_id.
        step_index: 1-based position of this step in the plan.
        state:      Mutable execution state; updated by this function.

    Returns:
        Tuple of (result_value, resolved_arguments_dict).

    Raises:
        ControllerError: Wrapping any failure in the above pipeline.
    """
    tool_name = step["tool"]
    step_id   = step.get("step_id")
    raw_args  = step["arguments"]

    log_step(step_index, tool_name)

    # ── 1. Resolve $-references ──────────────────────────────────────────────
    try:
        resolved_args = resolve_arguments(raw_args, state)
    except KeyError as exc:
        msg = (
            f"Reference resolution failed at step {step_index} ({tool_name}): {exc}"
        )
        log.error(msg)
        raise ControllerError(msg, step_index, step_id, tool_name, cause=exc) from exc

    log.debug("Step %d resolved args: %s", step_index, resolved_args)

    # ── 2. Validate resolved arguments against tool schema ───────────────────
    try:
        validate_step_arguments(tool_name, resolved_args)
        log_validation_pass(tool_name)
    except KeyError as exc:
        msg = f"Unregistered tool '{tool_name}' at step {step_index}."
        log_validation_fail(tool_name, str(exc))
        raise ControllerError(msg, step_index, step_id, tool_name, cause=exc) from exc
    except ValidationError as exc:
        msg = (
            f"Argument validation failed for '{tool_name}' "
            f"at step {step_index}: {exc.message}"
        )
        log_validation_fail(tool_name, exc.message)
        raise ControllerError(msg, step_index, step_id, tool_name, cause=exc) from exc

    # ── 3. Look up tool callable in registry ────────────────────────────────
    try:
        tool_fn = get_tool(tool_name)
    except KeyError as exc:
        msg = f"Tool '{tool_name}' not found in registry at step {step_index}."
        raise ControllerError(msg, step_index, step_id, tool_name, cause=exc) from exc

    # ── 4. Execute the tool ──────────────────────────────────────────────────
    try:
        result = tool_fn(**resolved_args)
    except Exception as exc:  # noqa: BLE001 — intentional broad catch, re-wrapped below
        msg = (
            f"Tool '{tool_name}' raised {type(exc).__name__} "
            f"at step {step_index}: {exc}"
        )
        log.error(msg)
        raise ControllerError(msg, step_index, step_id, tool_name, cause=exc) from exc

    # ── 5. Record result in execution state ──────────────────────────────────
    # resolved_args is captured here — before state.record() — so $prev in
    # the output record correctly reflects the value at the time of execution.
    state.record(step_id, result)
    log_result(step_index, result)

    return result, resolved_args


# ── Public API ─────────────────────────────────────────────────────────────────

def run(plan: dict) -> dict[str, Any]:
    """Execute a validated JSON execution plan step by step.

    Iterates every step in plan["steps"] sequentially.  Any ControllerError
    halts execution immediately — the function never raises, always returning
    a structured dict that callers can inspect for status and error details.

    Args:
        plan: A validated execution plan dict, e.g. as returned by
              planner.plan(). Must have a "steps" key containing a
              non-empty list of step dicts.

    Returns:
        A structured dict containing execution status, per-step results,
        the final output value, and an error block (null on success).
        See the module docstring for the exact schema.

    Raises:
        ValueError: If plan is not a dict with a non-empty "steps" list.
                    (Guards against callers bypassing the planner.)
    """
    # ── Input validation ─────────────────────────────────────────────────────
    if not isinstance(plan, dict) or "steps" not in plan:
        raise ValueError("run(): 'plan' must be a dict with a 'steps' key.")
    if not isinstance(plan["steps"], list) or not plan["steps"]:
        raise ValueError("run(): plan['steps'] must be a non-empty list.")

    steps       = plan["steps"]
    steps_total = len(steps)
    state       = ExecutionState()

    # Accumulates one record per successfully completed step.
    step_records: list[dict[str, Any]] = []

    log_section("Controller Loop")

    # ── Sequential execution loop ─────────────────────────────────────────────
    for step_index, step in enumerate(steps, start=1):
        tool_name = step.get("tool", "<unknown>")
        step_id   = step.get("step_id")

        try:
            result, resolved_args = _execute_step(step, step_index, state)
        except ControllerError as exc:
            # Hard stop — record the failure and return immediately.
            log_error(
                f"Step {step_index}/{steps_total} failed "
                f"({exc.tool_name}): {exc}"
            )
            return {
                "status":          "failure",
                "steps_completed": step_index - 1,
                "steps_total":     steps_total,
                "results":         step_records,
                "final_result":    state.last_result,
                "error": {
                    "step_index": exc.step_index,
                    "step_id":    exc.step_id,
                    "tool":       exc.tool_name,
                    "type":       type(exc.cause).__name__ if exc.cause else "ControllerError",
                    "message":    str(exc),
                },
            }

        # resolved_args contains concrete values captured before state.record(),
        # so $prev in the output reflects the prior step's value — not ours.
        step_records.append({
            "step_index": step_index,
            "step_id":    step_id,
            "tool":       tool_name,
            "arguments":  resolved_args,
            "result":     result,
        })

    # ── All steps completed ───────────────────────────────────────────────────
    final = state.last_result
    log_section("Execution Complete")
    print(f"\n  ✔  All {steps_total} step(s) completed successfully.")
    print(f"  ✔  Final result: {final!r}\n")

    return {
        "status":          "success",
        "steps_completed": steps_total,
        "steps_total":     steps_total,
        "results":         step_records,
        "final_result":    final,
        "error":           None,
    }
