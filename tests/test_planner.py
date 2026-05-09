"""
tests/test_planner.py
----------------------
Pytest test suite for the Phase 2 LLM Planner.

All OpenAI API calls are mocked — no real network requests are made.
Tests verify:

    - System prompt construction (tool names, descriptions, examples present)
    - Successful single-step and multi-step plan generation
    - JSON parse error → retry with correction message → success
    - Plan schema violation → retry → success
    - Argument schema violation → retry → success
    - Unknown tool name → retry → success
    - Exhaustion of all retries → PlannerError raised
    - Authentication errors are re-raised immediately (not retried)
    - Empty / non-string task input raises ValueError
    - Correction message content (error type, detail, raw response included)
    - MAX_RETRIES environment variable is respected

Run:
    pytest tests/test_planner.py -v
"""

import json
import os
from unittest.mock import MagicMock, call, patch

import pytest
from jsonschema import ValidationError

import planner as planner_module
from planner import (
    PlannerError,
    _build_correction_message,
    _build_system_prompt,
    _get_max_retries,
    _get_model,
    plan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_openai_response(content: str) -> MagicMock:
    """Return a minimal mock that mimics openai.ChatCompletion response structure."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# Valid single-step plan JSON strings used across multiple tests.
_VALID_ADD_PLAN = json.dumps({
    "steps": [
        {"step_id": "step1", "tool": "math.add", "arguments": {"a": 3, "b": 5}}
    ]
})

_VALID_MULTI_PLAN = json.dumps({
    "steps": [
        {"step_id": "step1", "tool": "math.add",      "arguments": {"a": 3,      "b": 5}},
        {"step_id": "step2", "tool": "math.multiply",  "arguments": {"a": "$prev", "b": 2}},
    ]
})

_VALID_CONCAT_PLAN = json.dumps({
    "steps": [
        {
            "step_id": "step1",
            "tool": "string.concat",
            "arguments": {"a": "Hello", "b": "World", "separator": " "},
        }
    ]
})

_VALID_DATETIME_PLAN = json.dumps({
    "steps": [
        {"step_id": "step1", "tool": "datetime.now", "arguments": {}}
    ]
})

_VALID_FILE_PLAN = json.dumps({
    "steps": [
        {"step_id": "step1", "tool": "file.read", "arguments": {"path": "sample.txt"}}
    ]
})


# ─────────────────────────────────────────────────────────────────────────────
# _build_system_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    """Tests for planner._build_system_prompt()"""

    def test_prompt_contains_all_tool_names(self):
        """All five tool names appear in the system prompt."""
        prompt = _build_system_prompt()
        for name in ["math.add", "math.multiply", "string.concat", "datetime.now", "file.read"]:
            assert name in prompt, f"Tool '{name}' missing from system prompt"

    def test_prompt_contains_valid_tool_names_list(self):
        """The VALID TOOL NAMES line lists all tools."""
        prompt = _build_system_prompt()
        assert "VALID TOOL NAMES" in prompt

    def test_prompt_contains_output_format_instructions(self):
        """The prompt specifies the required JSON output structure."""
        prompt = _build_system_prompt()
        assert '"steps"' in prompt
        assert '"step_id"' in prompt
        assert '"tool"' in prompt
        assert '"arguments"' in prompt

    def test_prompt_forbids_prose(self):
        """The prompt explicitly instructs the model not to output prose."""
        prompt = _build_system_prompt()
        assert "No prose" in prompt or "no prose" in prompt.lower()

    def test_prompt_forbids_markdown(self):
        """The prompt prohibits markdown output."""
        prompt = _build_system_prompt()
        assert "markdown" in prompt.lower()

    def test_prompt_contains_prev_reference_instruction(self):
        """The prompt documents the $prev reference syntax."""
        prompt = _build_system_prompt()
        assert "$prev" in prompt

    def test_prompt_contains_step_reference_instruction(self):
        """The prompt documents $step1/$step2 reference syntax."""
        prompt = _build_system_prompt()
        assert "$step1" in prompt

    def test_prompt_contains_example_arguments(self):
        """Tool examples appear in the prompt."""
        prompt = _build_system_prompt()
        # math.add example is {"a": 3, "b": 5}
        assert '"a"' in prompt and '"b"' in prompt

    def test_prompt_is_string(self):
        """The returned prompt is a non-empty string."""
        prompt = _build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100


# ─────────────────────────────────────────────────────────────────────────────
# _build_correction_message
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCorrectionMessage:
    """Tests for planner._build_correction_message()"""

    def test_contains_error_type(self):
        """The error type label appears in the correction message."""
        msg = _build_correction_message("INVALID JSON", "detail here", "bad raw")
        assert "INVALID JSON" in msg

    def test_contains_detail(self):
        """The specific error detail appears in the correction message."""
        msg = _build_correction_message("PLAN SCHEMA ERROR", "steps is required", "raw")
        assert "steps is required" in msg

    def test_contains_raw_response(self):
        """The model's previous raw output is echoed back."""
        raw = '{"broken": json'
        msg = _build_correction_message("INVALID JSON", "parse error", raw)
        assert raw in msg

    def test_instructs_json_only(self):
        """The correction message tells the model to return raw JSON."""
        msg = _build_correction_message("SCHEMA ERROR", "detail", "raw")
        assert "JSON" in msg

    def test_instructs_no_explanation(self):
        """The correction message forbids prose/explanation."""
        msg = _build_correction_message("SCHEMA ERROR", "detail", "raw")
        lower = msg.lower()
        assert "no explanation" in lower or "raw json only" in lower


# ─────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestConfiguration:
    """Tests for environment-driven configuration helpers."""

    def test_get_model_default(self):
        """Default model is gpt-4o."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_MODEL", None)
            assert _get_model() == "gpt-4o"

    def test_get_model_custom(self):
        """Custom model name is read from environment."""
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4-turbo"}):
            assert _get_model() == "gpt-4-turbo"

    def test_get_max_retries_default(self):
        """Default max retries is 3."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_RETRIES", None)
            assert _get_max_retries() == 3

    def test_get_max_retries_custom(self):
        """Custom MAX_RETRIES is read from environment."""
        with patch.dict(os.environ, {"MAX_RETRIES": "5"}):
            assert _get_max_retries() == 5

    def test_get_max_retries_invalid_falls_back_to_default(self):
        """Non-integer MAX_RETRIES falls back to 3."""
        with patch.dict(os.environ, {"MAX_RETRIES": "banana"}):
            assert _get_max_retries() == 3

    def test_get_max_retries_minimum_is_one(self):
        """MAX_RETRIES=0 is clamped to 1."""
        with patch.dict(os.environ, {"MAX_RETRIES": "0"}):
            assert _get_max_retries() == 1


# ─────────────────────────────────────────────────────────────────────────────
# plan() — input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanInputValidation:
    """Tests for plan() argument validation (no API calls needed)."""

    def test_raises_on_empty_string(self):
        """Empty task string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            plan("")

    def test_raises_on_whitespace_only(self):
        """Whitespace-only task string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            plan("   ")

    def test_raises_on_non_string(self):
        """Non-string task raises ValueError."""
        with pytest.raises(ValueError):
            plan(42)

    def test_raises_on_none(self):
        """None task raises ValueError."""
        with pytest.raises(ValueError):
            plan(None)


# ─────────────────────────────────────────────────────────────────────────────
# plan() — happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanHappyPath:
    """Tests for plan() when the model returns a valid plan on the first attempt."""

    @patch("planner._call_openai")
    def test_single_step_math_add(self, mock_call):
        """A valid math.add plan is returned on the first attempt."""
        mock_call.return_value = _VALID_ADD_PLAN
        result = plan("Add 3 and 5")

        assert result["steps"][0]["tool"] == "math.add"
        assert result["steps"][0]["arguments"] == {"a": 3, "b": 5}
        mock_call.assert_called_once()

    @patch("planner._call_openai")
    def test_multi_step_plan(self, mock_call):
        """A valid multi-step plan with $prev reference is returned."""
        mock_call.return_value = _VALID_MULTI_PLAN
        result = plan("Add 3 and 5, then multiply the result by 2")

        assert len(result["steps"]) == 2
        assert result["steps"][0]["tool"] == "math.add"
        assert result["steps"][1]["tool"] == "math.multiply"
        assert result["steps"][1]["arguments"]["a"] == "$prev"

    @patch("planner._call_openai")
    def test_string_concat_plan(self, mock_call):
        """A valid string.concat plan is returned."""
        mock_call.return_value = _VALID_CONCAT_PLAN
        result = plan("Concatenate Hello and World with a space")

        assert result["steps"][0]["tool"] == "string.concat"
        assert result["steps"][0]["arguments"]["separator"] == " "

    @patch("planner._call_openai")
    def test_datetime_now_plan_empty_args(self, mock_call):
        """datetime.now with empty arguments dict is valid."""
        mock_call.return_value = _VALID_DATETIME_PLAN
        result = plan("What is the current time?")

        assert result["steps"][0]["tool"] == "datetime.now"
        assert result["steps"][0]["arguments"] == {}

    @patch("planner._call_openai")
    def test_file_read_plan(self, mock_call):
        """A valid file.read plan is returned."""
        mock_call.return_value = _VALID_FILE_PLAN
        result = plan("Read the sample.txt file from the workspace")

        assert result["steps"][0]["tool"] == "file.read"
        assert result["steps"][0]["arguments"]["path"] == "sample.txt"

    @patch("planner._call_openai")
    def test_returns_dict(self, mock_call):
        """plan() always returns a Python dict, never a raw string."""
        mock_call.return_value = _VALID_ADD_PLAN
        result = plan("Add 1 and 2")
        assert isinstance(result, dict)

    @patch("planner._call_openai")
    def test_result_contains_steps_key(self, mock_call):
        """Returned dict always has a 'steps' key."""
        mock_call.return_value = _VALID_ADD_PLAN
        result = plan("Add 1 and 2")
        assert "steps" in result
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) >= 1

    @patch("planner._call_openai")
    def test_strips_task_whitespace(self, mock_call):
        """Leading/trailing whitespace in the task is stripped before sending."""
        mock_call.return_value = _VALID_ADD_PLAN
        plan("   Add 3 and 5   ")
        # Verify the message sent to the API has stripped content.
        sent_messages = mock_call.call_args[0][0]
        user_msg = next(m for m in sent_messages if m["role"] == "user")
        assert user_msg["content"] == "Add 3 and 5"


# ─────────────────────────────────────────────────────────────────────────────
# plan() — retry on JSON parse failure
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanRetryOnJsonError:
    """Tests for the retry loop triggered by malformed JSON responses."""

    @patch("planner._call_openai")
    def test_retries_once_on_bad_json_then_succeeds(self, mock_call):
        """Bad JSON on attempt 1, valid JSON on attempt 2 → success."""
        mock_call.side_effect = [
            "This is not JSON at all.",   # attempt 1 — parse fails
            _VALID_ADD_PLAN,              # attempt 2 — success
        ]
        result = plan("Add 3 and 5")

        assert result["steps"][0]["tool"] == "math.add"
        assert mock_call.call_count == 2

    @patch("planner._call_openai")
    def test_retry_includes_correction_message(self, mock_call):
        """The second API call includes a correction user message."""
        mock_call.side_effect = [
            "not json",
            _VALID_ADD_PLAN,
        ]
        plan("Add 3 and 5")

        # Second call should have 4 messages: system, user task, assistant bad, correction
        second_call_messages = mock_call.call_args_list[1][0][0]
        roles = [m["role"] for m in second_call_messages]
        assert roles == ["system", "user", "assistant", "user"]

    @patch("planner._call_openai")
    def test_correction_message_mentions_invalid_json(self, mock_call):
        """The correction message labels the error as INVALID JSON."""
        mock_call.side_effect = ["not json", _VALID_ADD_PLAN]
        plan("Add 1 and 1")

        last_messages = mock_call.call_args_list[1][0][0]
        correction = last_messages[-1]["content"]
        assert "INVALID JSON" in correction

    @patch("planner._call_openai")
    def test_bad_json_three_times_raises_planner_error(self, mock_call):
        """Three consecutive JSON parse failures raise PlannerError."""
        mock_call.return_value = "definitely not json {{}}"

        with patch.dict(os.environ, {"MAX_RETRIES": "3"}):
            with pytest.raises(PlannerError) as exc_info:
                plan("Compute something")

        assert exc_info.value.attempts == 3
        assert mock_call.call_count == 3

    @patch("planner._call_openai")
    def test_planner_error_contains_last_raw(self, mock_call):
        """PlannerError.last_raw holds the last model response string."""
        bad_output = "i am not json"
        mock_call.return_value = bad_output

        with patch.dict(os.environ, {"MAX_RETRIES": "1"}):
            with pytest.raises(PlannerError) as exc_info:
                plan("do something")

        assert exc_info.value.last_raw == bad_output

    @patch("planner._call_openai")
    def test_planner_error_contains_last_error(self, mock_call):
        """PlannerError.last_error holds the last causing exception."""
        mock_call.return_value = "not json"

        with patch.dict(os.environ, {"MAX_RETRIES": "1"}):
            with pytest.raises(PlannerError) as exc_info:
                plan("do something")

        import json as json_mod
        assert isinstance(exc_info.value.last_error, json_mod.JSONDecodeError)


# ─────────────────────────────────────────────────────────────────────────────
# plan() — retry on plan schema validation failure
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanRetryOnPlanSchemaError:
    """Tests for the retry loop triggered by plan-level schema violations."""

    @patch("planner._call_openai")
    def test_retries_on_empty_steps_then_succeeds(self, mock_call):
        """Empty steps array → retry → valid plan succeeds."""
        bad_plan = json.dumps({"steps": []})  # fails minItems
        mock_call.side_effect = [bad_plan, _VALID_ADD_PLAN]

        result = plan("Add 3 and 5")
        assert result["steps"][0]["tool"] == "math.add"
        assert mock_call.call_count == 2

    @patch("planner._call_openai")
    def test_retries_on_missing_steps_key(self, mock_call):
        """Plan missing 'steps' key → retry → success."""
        bad_plan = json.dumps({"tool": "math.add", "arguments": {"a": 1, "b": 2}})
        mock_call.side_effect = [bad_plan, _VALID_ADD_PLAN]

        result = plan("Add 1 and 2")
        assert "steps" in result
        assert mock_call.call_count == 2

    @patch("planner._call_openai")
    def test_correction_message_mentions_plan_schema_error(self, mock_call):
        """Correction message for a plan-level error says PLAN SCHEMA ERROR."""
        bad_plan = json.dumps({"steps": []})
        mock_call.side_effect = [bad_plan, _VALID_ADD_PLAN]

        plan("Add 3 and 5")

        last_messages = mock_call.call_args_list[1][0][0]
        correction = last_messages[-1]["content"]
        assert "PLAN SCHEMA ERROR" in correction

    @patch("planner._call_openai")
    def test_three_bad_plan_structures_raise_planner_error(self, mock_call):
        """Three consecutive plan-level schema failures raise PlannerError."""
        bad_plan = json.dumps({"steps": []})
        mock_call.return_value = bad_plan

        with patch.dict(os.environ, {"MAX_RETRIES": "3"}):
            with pytest.raises(PlannerError) as exc_info:
                plan("Do something")

        assert exc_info.value.attempts == 3


# ─────────────────────────────────────────────────────────────────────────────
# plan() — retry on argument schema validation failure
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanRetryOnArgumentSchemaError:
    """Tests for the retry loop triggered by per-step argument violations."""

    @patch("planner._call_openai")
    def test_retries_on_wrong_arg_type_then_succeeds(self, mock_call):
        """String instead of number for math.add → retry → success."""
        bad_plan = json.dumps({
            "steps": [
                {"step_id": "step1", "tool": "math.add", "arguments": {"a": "three", "b": 5}}
            ]
        })
        mock_call.side_effect = [bad_plan, _VALID_ADD_PLAN]

        result = plan("Add three and 5")
        assert result["steps"][0]["arguments"]["a"] == 3
        assert mock_call.call_count == 2

    @patch("planner._call_openai")
    def test_correction_message_mentions_argument_schema_error(self, mock_call):
        """Correction message for argument errors says ARGUMENT SCHEMA ERROR."""
        bad_plan = json.dumps({
            "steps": [
                {"step_id": "step1", "tool": "math.add", "arguments": {"a": "bad", "b": 5}}
            ]
        })
        mock_call.side_effect = [bad_plan, _VALID_ADD_PLAN]

        plan("Do math")

        last_messages = mock_call.call_args_list[1][0][0]
        correction = last_messages[-1]["content"]
        assert "ARGUMENT SCHEMA ERROR" in correction

    @patch("planner._call_openai")
    def test_retries_on_extra_argument_key(self, mock_call):
        """Extra argument key in math.add fails additionalProperties → retry."""
        bad_plan = json.dumps({
            "steps": [
                {
                    "step_id": "step1",
                    "tool": "math.add",
                    "arguments": {"a": 3, "b": 5, "c": 99},  # c is not allowed
                }
            ]
        })
        mock_call.side_effect = [bad_plan, _VALID_ADD_PLAN]

        result = plan("Add 3 and 5")
        assert mock_call.call_count == 2
        assert result["steps"][0]["tool"] == "math.add"


# ─────────────────────────────────────────────────────────────────────────────
# plan() — retry on unknown tool name
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanRetryOnUnknownTool:
    """Tests for the retry loop triggered by unregistered tool names."""

    @patch("planner._call_openai")
    def test_retries_on_unknown_tool_then_succeeds(self, mock_call):
        """Unknown tool name → retry → success."""
        bad_plan = json.dumps({
            "steps": [
                {"step_id": "step1", "tool": "shell.execute", "arguments": {"cmd": "ls"}}
            ]
        })
        mock_call.side_effect = [bad_plan, _VALID_ADD_PLAN]

        result = plan("Run ls")
        assert result["steps"][0]["tool"] == "math.add"
        assert mock_call.call_count == 2

    @patch("planner._call_openai")
    def test_correction_message_mentions_unknown_tool_name(self, mock_call):
        """Correction message for unknown tool says UNKNOWN TOOL NAME."""
        bad_plan = json.dumps({
            "steps": [
                {"step_id": "step1", "tool": "nonexistent.tool", "arguments": {}}
            ]
        })
        mock_call.side_effect = [bad_plan, _VALID_ADD_PLAN]

        plan("Do something")

        last_messages = mock_call.call_args_list[1][0][0]
        correction = last_messages[-1]["content"]
        assert "UNKNOWN TOOL NAME" in correction


# ─────────────────────────────────────────────────────────────────────────────
# plan() — retry count respects MAX_RETRIES environment variable
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanRetriesRespectMaxRetries:
    """Tests that the retry loop uses MAX_RETRIES from the environment."""

    @patch("planner._call_openai")
    def test_max_retries_one_gives_exactly_one_call(self, mock_call):
        """With MAX_RETRIES=1, the API is called exactly once even on failure."""
        mock_call.return_value = "bad json"

        with patch.dict(os.environ, {"MAX_RETRIES": "1"}):
            with pytest.raises(PlannerError):
                plan("Do something")

        assert mock_call.call_count == 1

    @patch("planner._call_openai")
    def test_max_retries_two_gives_at_most_two_calls(self, mock_call):
        """With MAX_RETRIES=2, the API is called at most twice on failure."""
        mock_call.return_value = "bad json"

        with patch.dict(os.environ, {"MAX_RETRIES": "2"}):
            with pytest.raises(PlannerError):
                plan("Do something")

        assert mock_call.call_count == 2

    @patch("planner._call_openai")
    def test_success_on_last_attempt_is_not_an_error(self, mock_call):
        """Succeeding on the last allowed attempt returns the plan, not an error."""
        mock_call.side_effect = [
            "bad json",         # attempt 1
            "bad json",         # attempt 2
            _VALID_ADD_PLAN,    # attempt 3 — success
        ]
        with patch.dict(os.environ, {"MAX_RETRIES": "3"}):
            result = plan("Add 3 and 5")

        assert result["steps"][0]["tool"] == "math.add"
        assert mock_call.call_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# plan() — authentication errors are not retried
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanAuthenticationError:
    """Tests that authentication errors are re-raised immediately."""

    @patch("planner._call_openai")
    def test_auth_error_is_not_retried(self, mock_call):
        """openai.AuthenticationError raised by _call_openai propagates immediately."""
        import openai as openai_mod

        mock_call.side_effect = openai_mod.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body=None,
        )

        with pytest.raises(openai_mod.AuthenticationError):
            plan("Do something")

        # Only one call should have been attempted.
        assert mock_call.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# PlannerError attributes
# ─────────────────────────────────────────────────────────────────────────────

class TestPlannerError:
    """Tests for the PlannerError exception class."""

    def test_planner_error_is_runtime_error(self):
        """PlannerError subclasses RuntimeError."""
        err = PlannerError("test", attempts=3)
        assert isinstance(err, RuntimeError)

    def test_planner_error_stores_attempts(self):
        """PlannerError.attempts stores the number of attempts made."""
        err = PlannerError("test", attempts=3)
        assert err.attempts == 3

    def test_planner_error_stores_last_raw(self):
        """PlannerError.last_raw stores the last raw model output."""
        err = PlannerError("test", attempts=2, last_raw="bad response")
        assert err.last_raw == "bad response"

    def test_planner_error_stores_last_error(self):
        """PlannerError.last_error stores the causing exception."""
        cause = ValueError("root cause")
        err = PlannerError("test", attempts=1, last_error=cause)
        assert err.last_error is cause

    def test_planner_error_defaults_to_none(self):
        """last_raw and last_error default to None."""
        err = PlannerError("test", attempts=1)
        assert err.last_raw is None
        assert err.last_error is None

    @patch("planner._call_openai")
    def test_planner_error_message_includes_attempt_count(self, mock_call):
        """PlannerError message mentions the number of attempts."""
        mock_call.return_value = "bad json"

        with patch.dict(os.environ, {"MAX_RETRIES": "2"}):
            with pytest.raises(PlannerError) as exc_info:
                plan("Do something")

        assert "2" in str(exc_info.value)
