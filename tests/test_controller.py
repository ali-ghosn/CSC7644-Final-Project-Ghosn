"""
tests/test_controller.py
-------------------------
Pytest test suite for the Phase 3 Controller Loop.

No OpenAI API calls are made.  Plans are constructed as plain dicts and
passed directly to controller.run() and its sub-components.

Covers:
    ExecutionState
        - record() and resolve() for $prev
        - resolve() for $step1, $step2
        - last_result property
        - snapshot()
        - error on $prev with no prior steps
        - error on unknown $stepN reference

    resolve_arguments()
        - concrete values pass through unchanged
        - $prev is substituted
        - $step1 / $step2 are substituted
        - mixed concrete + reference dicts
        - error on unresolvable reference

    controller.run() — happy paths
        - single-step math.add
        - single-step math.multiply
        - single-step string.concat
        - single-step datetime.now
        - single-step file.read (sandboxed)
        - multi-step with $prev reference
        - multi-step with explicit $step1 reference
        - multi-step 3-step chain
        - output structure (status, steps_completed, results, final_result, error)
        - resolved argument values appear in output (not raw $refs)

    controller.run() — failure paths
        - unregistered tool name → failure output, no exception raised
        - schema validation failure on resolved args → failure output
        - tool execution error (TypeError in tool) → failure output
        - $prev used in first step → failure output
        - unresolvable $stepN reference → failure output
        - failure stops execution (subsequent steps not run)
        - error dict fields are populated correctly

    controller.run() — input validation
        - non-dict input raises ValueError
        - missing 'steps' key raises ValueError
        - empty steps list raises ValueError

    ControllerError
        - attributes: step_index, step_id, tool_name, cause

Run:
    pytest tests/test_controller.py -v
"""

import os
from unittest.mock import patch

import pytest

from controller import (
    ControllerError,
    ExecutionState,
    resolve_arguments,
    run,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _step(tool: str, arguments: dict, step_id: str = "step1") -> dict:
    """Build a minimal plan step dict."""
    return {"step_id": step_id, "tool": tool, "arguments": arguments}


def _plan(*steps: dict) -> dict:
    """Build a plan dict from a sequence of step dicts."""
    return {"steps": list(steps)}


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionState
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionState:
    """Tests for controller.ExecutionState"""

    def test_record_and_resolve_prev(self):
        """$prev resolves to the most recently recorded result."""
        state = ExecutionState()
        state.record("step1", 42)
        assert state.resolve("$prev") == 42

    def test_prev_returns_most_recent(self):
        """$prev always returns the last recorded result."""
        state = ExecutionState()
        state.record("step1", 10)
        state.record("step2", 99)
        assert state.resolve("$prev") == 99

    def test_resolve_by_step_id(self):
        """$step1 resolves to the result recorded under 'step1'."""
        state = ExecutionState()
        state.record("step1", 8)
        assert state.resolve("$step1") == 8

    def test_resolve_multiple_step_ids(self):
        """Multiple named step references resolve independently."""
        state = ExecutionState()
        state.record("step1", 3)
        state.record("step2", 7)
        assert state.resolve("$step1") == 3
        assert state.resolve("$step2") == 7

    def test_prev_with_no_prior_steps_raises_key_error(self):
        """$prev on an empty state raises KeyError."""
        state = ExecutionState()
        with pytest.raises(KeyError, match="no previous result"):
            state.resolve("$prev")

    def test_unknown_step_id_raises_key_error(self):
        """Resolving an unknown $stepN raises KeyError."""
        state = ExecutionState()
        state.record("step1", 1)
        with pytest.raises(KeyError, match="step99"):
            state.resolve("$step99")

    def test_last_result_none_when_empty(self):
        """last_result is None before any steps are recorded."""
        state = ExecutionState()
        assert state.last_result is None

    def test_last_result_after_recording(self):
        """last_result equals the most recently recorded value."""
        state = ExecutionState()
        state.record("step1", 5)
        state.record("step2", 10)
        assert state.last_result == 10

    def test_snapshot_returns_copy(self):
        """snapshot() returns the step_id → result mapping."""
        state = ExecutionState()
        state.record("step1", 1)
        state.record("step2", 2)
        snap = state.snapshot()
        assert snap == {"step1": 1, "step2": 2}

    def test_snapshot_is_independent_copy(self):
        """Mutating the snapshot does not affect the state."""
        state = ExecutionState()
        state.record("step1", 1)
        snap = state.snapshot()
        snap["step1"] = 999
        assert state.resolve("$step1") == 1

    def test_record_without_step_id(self):
        """Recording with step_id=None only adds to ordered list."""
        state = ExecutionState()
        state.record(None, 42)
        assert state.resolve("$prev") == 42

    def test_non_reference_string_raises_value_error(self):
        """Passing a non-reference string raises ValueError."""
        state = ExecutionState()
        with pytest.raises(ValueError, match=r"Not a valid"):
            state.resolve("plain_string")


# ─────────────────────────────────────────────────────────────────────────────
# resolve_arguments
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveArguments:
    """Tests for controller.resolve_arguments()"""

    def test_concrete_values_pass_through(self):
        """Arguments with no references are returned unchanged."""
        state = ExecutionState()
        result = resolve_arguments({"a": 3, "b": 5}, state)
        assert result == {"a": 3, "b": 5}

    def test_prev_reference_is_substituted(self):
        """$prev is replaced with the last recorded result."""
        state = ExecutionState()
        state.record("step1", 8)
        result = resolve_arguments({"a": "$prev", "b": 2}, state)
        assert result == {"a": 8, "b": 2}

    def test_step_id_reference_is_substituted(self):
        """$step1 is replaced with the result of step1."""
        state = ExecutionState()
        state.record("step1", 7)
        result = resolve_arguments({"a": "$step1", "b": 3}, state)
        assert result == {"a": 7, "b": 3}

    def test_multiple_references_resolved(self):
        """Multiple references in the same arguments dict are all resolved."""
        state = ExecutionState()
        state.record("step1", 4)
        state.record("step2", 5)
        result = resolve_arguments({"a": "$step1", "b": "$step2"}, state)
        assert result == {"a": 4, "b": 5}

    def test_mixed_concrete_and_reference(self):
        """Mix of concrete values and references works correctly."""
        state = ExecutionState()
        state.record("step1", 10)
        result = resolve_arguments({"a": "$step1", "b": 99, "separator": "-"}, state)
        assert result == {"a": 10, "b": 99, "separator": "-"}

    def test_empty_arguments_returns_empty(self):
        """Empty arguments dict returns empty dict."""
        state = ExecutionState()
        result = resolve_arguments({}, state)
        assert result == {}

    def test_unresolvable_reference_raises_key_error(self):
        """Reference to an unrecorded step raises KeyError."""
        state = ExecutionState()
        with pytest.raises(KeyError):
            resolve_arguments({"a": "$step99"}, state)

    def test_resolved_dict_is_a_new_object(self):
        """resolve_arguments returns a new dict, not a mutated original."""
        state = ExecutionState()
        state.record("step1", 1)
        original = {"a": "$prev"}
        resolved = resolve_arguments(original, state)
        assert original == {"a": "$prev"}    # original unchanged
        assert resolved == {"a": 1}


# ─────────────────────────────────────────────────────────────────────────────
# run() — input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestRunInputValidation:
    """Tests for run() input guards."""

    def test_raises_on_non_dict(self):
        """Non-dict input raises ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            run("not a dict")

    def test_raises_on_missing_steps_key(self):
        """Dict without 'steps' key raises ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            run({"tool": "math.add"})

    def test_raises_on_empty_steps(self):
        """Empty steps list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            run({"steps": []})


# ─────────────────────────────────────────────────────────────────────────────
# run() — single-step happy paths
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSingleStepHappyPath:
    """Tests for run() with valid single-step plans."""

    def test_math_add(self):
        """math.add returns correct sum."""
        result = run(_plan(_step("math.add", {"a": 3, "b": 5})))
        assert result["status"] == "success"
        assert result["final_result"] == 8

    def test_math_multiply(self):
        """math.multiply returns correct product."""
        result = run(_plan(_step("math.multiply", {"a": 4, "b": 3})))
        assert result["status"] == "success"
        assert result["final_result"] == 12

    def test_math_add_floats(self):
        """math.add works with float inputs."""
        result = run(_plan(_step("math.add", {"a": 1.5, "b": 2.5})))
        assert result["status"] == "success"
        assert result["final_result"] == 4.0

    def test_string_concat_no_separator(self):
        """string.concat without separator returns concatenated string."""
        result = run(_plan(_step("string.concat", {"a": "foo", "b": "bar"})))
        assert result["status"] == "success"
        assert result["final_result"] == "foobar"

    def test_string_concat_with_separator(self):
        """string.concat with separator returns joined string."""
        result = run(_plan(
            _step("string.concat", {"a": "Hello", "b": "World", "separator": " "})
        ))
        assert result["status"] == "success"
        assert result["final_result"] == "Hello World"

    def test_datetime_now_default_format(self):
        """datetime.now returns a non-empty timestamp string."""
        import re
        result = run(_plan(_step("datetime.now", {})))
        assert result["status"] == "success"
        ts = result["final_result"]
        assert isinstance(ts, str)
        assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", ts)

    def test_datetime_now_custom_format(self):
        """datetime.now with custom fmt respects the format."""
        import re
        result = run(_plan(_step("datetime.now", {"fmt": "%Y-%m-%d"})))
        assert result["status"] == "success"
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", result["final_result"])

    def test_file_read(self, tmp_path):
        """file.read returns the content of a workspace file."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "hello.txt").write_text("hello world", encoding="utf-8")

        with patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}):
            result = run(_plan(_step("file.read", {"path": "hello.txt"})))

        assert result["status"] == "success"
        assert result["final_result"] == "hello world"


# ─────────────────────────────────────────────────────────────────────────────
# run() — multi-step happy paths
# ─────────────────────────────────────────────────────────────────────────────

class TestRunMultiStepHappyPath:
    """Tests for run() with multi-step plans using $prev and $stepN references."""

    def test_two_steps_with_prev(self):
        """Two-step plan: add then multiply using $prev."""
        plan = _plan(
            _step("math.add",      {"a": 3, "b": 5},         step_id="step1"),
            _step("math.multiply", {"a": "$prev", "b": 2},   step_id="step2"),
        )
        result = run(plan)
        assert result["status"] == "success"
        assert result["final_result"] == 16      # (3+5) * 2
        assert result["steps_completed"] == 2

    def test_two_steps_with_named_reference(self):
        """Two-step plan: second step references step1 by name."""
        plan = _plan(
            _step("math.add",      {"a": 4, "b": 6},          step_id="step1"),
            _step("math.multiply", {"a": "$step1", "b": 3},   step_id="step2"),
        )
        result = run(plan)
        assert result["status"] == "success"
        assert result["final_result"] == 30     # (4+6) * 3

    def test_three_step_chain(self):
        """Three-step chain where int result passed to string tool — correct failure.

        math.multiply produces int 20; string.concat requires str for 'b'.
        The controller's post-resolution schema validation catches the type
        mismatch and returns a failure output.  This is the correct behavior.
        """
        plan = _plan(
            _step("math.add",      {"a": 2, "b": 3},              step_id="step1"),
            _step("math.multiply", {"a": "$prev", "b": 4},        step_id="step2"),
            _step("string.concat", {"a": "Result: ", "b": "$prev"}, step_id="step3"),
        )
        result = run(plan)
        # string.concat expects 'b' to be a string; receives int 20 → validation fails.
        assert result["status"] == "failure"
        assert result["steps_completed"] == 2   # step1 and step2 succeeded
        assert result["error"]["step_index"] == 3

    def test_three_step_string_chain(self):
        """Three-step chain using string tools."""
        plan = _plan(
            _step("string.concat", {"a": "Hello",  "b": " "},   step_id="step1"),
            _step("string.concat", {"a": "$prev",  "b": "World"}, step_id="step2"),
            _step("string.concat", {"a": "$step2", "b": "!"},   step_id="step3"),
        )
        result = run(plan)
        assert result["status"] == "success"
        assert result["final_result"] == "Hello World!"

    def test_add_then_add_using_prev(self):
        """Add twice: second add uses $prev."""
        plan = _plan(
            _step("math.add", {"a": 10, "b": 5},        step_id="step1"),
            _step("math.add", {"a": "$prev", "b": 3},   step_id="step2"),
        )
        result = run(plan)
        assert result["status"] == "success"
        assert result["final_result"] == 18     # 15 + 3

    def test_results_list_length_matches_steps(self):
        """The results list contains one entry per executed step."""
        plan = _plan(
            _step("math.add",      {"a": 1, "b": 2}, step_id="step1"),
            _step("math.multiply", {"a": "$prev", "b": 4}, step_id="step2"),
        )
        result = run(plan)
        assert len(result["results"]) == 2

    def test_results_contain_resolved_arguments(self):
        """Output results show resolved values, not raw $-references."""
        plan = _plan(
            _step("math.add",      {"a": 3, "b": 5},        step_id="step1"),
            _step("math.multiply", {"a": "$prev", "b": 2},  step_id="step2"),
        )
        result = run(plan)
        # The second step's recorded arguments should show the resolved value (8),
        # not the raw reference string ("$prev").
        step2_record = result["results"][1]
        assert step2_record["arguments"]["a"] == 8   # resolved from $prev

    def test_results_contain_correct_step_ids(self):
        """Each result record contains the correct step_id."""
        plan = _plan(
            _step("math.add",      {"a": 1, "b": 2}, step_id="alpha"),
            _step("math.multiply", {"a": "$prev", "b": 3}, step_id="beta"),
        )
        result = run(plan)
        assert result["results"][0]["step_id"] == "alpha"
        assert result["results"][1]["step_id"] == "beta"

    def test_steps_completed_equals_steps_total_on_success(self):
        """steps_completed equals steps_total on a fully successful run."""
        plan = _plan(
            _step("math.add", {"a": 1, "b": 2}, step_id="step1"),
        )
        result = run(plan)
        assert result["steps_completed"] == result["steps_total"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# run() — output structure
# ─────────────────────────────────────────────────────────────────────────────

class TestRunOutputStructure:
    """Tests verifying the output dict structure of run()."""

    def _success_output(self):
        return run(_plan(_step("math.add", {"a": 1, "b": 2})))

    def test_success_status_is_success(self):
        assert self._success_output()["status"] == "success"

    def test_success_error_is_none(self):
        assert self._success_output()["error"] is None

    def test_success_has_steps_completed(self):
        out = self._success_output()
        assert "steps_completed" in out
        assert isinstance(out["steps_completed"], int)

    def test_success_has_steps_total(self):
        out = self._success_output()
        assert "steps_total" in out
        assert isinstance(out["steps_total"], int)

    def test_success_has_results_list(self):
        out = self._success_output()
        assert "results" in out
        assert isinstance(out["results"], list)

    def test_success_has_final_result(self):
        out = self._success_output()
        assert "final_result" in out

    def test_result_record_has_required_keys(self):
        """Each entry in results has step_index, step_id, tool, arguments, result."""
        out = self._success_output()
        record = out["results"][0]
        for key in ("step_index", "step_id", "tool", "arguments", "result"):
            assert key in record, f"Missing key '{key}' in result record"

    def test_result_record_step_index_is_one_based(self):
        """step_index in the first result record is 1."""
        out = self._success_output()
        assert out["results"][0]["step_index"] == 1

    def test_result_record_result_matches_final_result(self):
        """For a single-step plan, result record matches final_result."""
        out = self._success_output()
        assert out["results"][0]["result"] == out["final_result"]


# ─────────────────────────────────────────────────────────────────────────────
# run() — failure paths
# ─────────────────────────────────────────────────────────────────────────────

class TestRunFailurePaths:
    """Tests for run() failure handling."""

    def test_unregistered_tool_returns_failure(self):
        """An unregistered tool name produces a failure output."""
        plan = _plan(_step("shell.execute", {"cmd": "ls"}))
        result = run(plan)
        assert result["status"] == "failure"

    def test_unregistered_tool_stops_execution(self):
        """Failure on step 1 means step 2 is never executed."""
        plan = _plan(
            _step("shell.execute", {"cmd": "ls"},       step_id="step1"),
            _step("math.add",      {"a": 1, "b": 2},   step_id="step2"),
        )
        result = run(plan)
        assert result["status"] == "failure"
        assert result["steps_completed"] == 0
        # Only 0 steps completed — step 2 was never run.
        assert len(result["results"]) == 0

    def test_schema_validation_failure_returns_failure(self):
        """Argument type mismatch on resolved args produces failure output."""
        # math.add expects numbers; pass strings directly (no references)
        plan = _plan(_step("math.add", {"a": "not-a-number", "b": 5}))
        result = run(plan)
        assert result["status"] == "failure"

    def test_tool_execution_error_returns_failure(self):
        """A TypeError raised by the tool itself is caught and returns failure."""
        # file.read on a non-existent file raises FileNotFoundError
        plan = _plan(_step("file.read", {"path": "nonexistent_xyz.txt"}))
        result = run(plan)
        assert result["status"] == "failure"

    def test_prev_in_first_step_returns_failure(self):
        """$prev in the first step (no prior results) returns failure."""
        plan = _plan(_step("math.add", {"a": "$prev", "b": 5}))
        result = run(plan)
        assert result["status"] == "failure"

    def test_unresolvable_step_reference_returns_failure(self):
        """$stepXXX that was never recorded returns failure."""
        plan = _plan(
            _step("math.add",      {"a": "$step99", "b": 1}, step_id="step1"),
        )
        result = run(plan)
        assert result["status"] == "failure"

    def test_failure_after_success_reports_correct_steps_completed(self):
        """If step 1 succeeds and step 2 fails, steps_completed == 1."""
        plan = _plan(
            _step("math.add",      {"a": 3, "b": 5},      step_id="step1"),
            _step("shell.execute", {"cmd": "bad"},         step_id="step2"),
        )
        result = run(plan)
        assert result["status"] == "failure"
        assert result["steps_completed"] == 1

    def test_failure_output_has_error_dict(self):
        """Failure output includes a non-null 'error' dict."""
        plan = _plan(_step("shell.execute", {"cmd": "ls"}))
        result = run(plan)
        assert result["error"] is not None
        assert isinstance(result["error"], dict)

    def test_failure_error_dict_has_required_keys(self):
        """error dict contains step_index, step_id, tool, type, message."""
        plan = _plan(_step("shell.execute", {"cmd": "ls"}))
        result = run(plan)
        error = result["error"]
        for key in ("step_index", "step_id", "tool", "type", "message"):
            assert key in error, f"Missing key '{key}' in error dict"

    def test_failure_error_step_index_is_correct(self):
        """error.step_index matches the index of the failing step."""
        plan = _plan(
            _step("math.add",      {"a": 1, "b": 2},   step_id="step1"),
            _step("shell.execute", {"cmd": "ls"},       step_id="step2"),
        )
        result = run(plan)
        assert result["error"]["step_index"] == 2

    def test_failure_preserves_prior_results(self):
        """Results from successfully completed steps are preserved on failure."""
        plan = _plan(
            _step("math.add",      {"a": 3, "b": 5},   step_id="step1"),
            _step("shell.execute", {"cmd": "ls"},       step_id="step2"),
        )
        result = run(plan)
        assert result["steps_completed"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["result"] == 8

    def test_failure_final_result_is_last_successful(self):
        """final_result on failure is the last successfully computed value."""
        plan = _plan(
            _step("math.add",      {"a": 3, "b": 5},   step_id="step1"),
            _step("shell.execute", {"cmd": "ls"},       step_id="step2"),
        )
        result = run(plan)
        # Step 1 succeeded with result 8; that is the last successful result.
        assert result["final_result"] == 8

    def test_failure_no_exception_raised(self):
        """run() does NOT raise an exception on tool failure — it returns a dict."""
        plan = _plan(_step("shell.execute", {"cmd": "ls"}))
        # Should not raise:
        result = run(plan)
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# ControllerError
# ─────────────────────────────────────────────────────────────────────────────

class TestControllerError:
    """Tests for the ControllerError exception class."""

    def test_is_runtime_error(self):
        """ControllerError subclasses RuntimeError."""
        err = ControllerError("msg", step_index=1, step_id="step1", tool_name="math.add")
        assert isinstance(err, RuntimeError)

    def test_stores_step_index(self):
        err = ControllerError("msg", step_index=2, step_id="s", tool_name="t")
        assert err.step_index == 2

    def test_stores_step_id(self):
        err = ControllerError("msg", step_index=1, step_id="step1", tool_name="t")
        assert err.step_id == "step1"

    def test_stores_tool_name(self):
        err = ControllerError("msg", step_index=1, step_id="s", tool_name="math.add")
        assert err.tool_name == "math.add"

    def test_stores_cause(self):
        cause = ValueError("root")
        err = ControllerError("msg", step_index=1, step_id="s", tool_name="t", cause=cause)
        assert err.cause is cause

    def test_cause_defaults_to_none(self):
        err = ControllerError("msg", step_index=1, step_id="s", tool_name="t")
        assert err.cause is None

    def test_step_id_can_be_none(self):
        """step_id is optional — steps without IDs set it to None."""
        err = ControllerError("msg", step_index=1, step_id=None, tool_name="t")
        assert err.step_id is None
