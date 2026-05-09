"""
tests/test_validation.py
-------------------------
Pytest test suite for schema validation and path sandboxing utilities.

Covers:
    - validators.validate_plan()         (valid/invalid plan structures)
    - validators.validate_step_arguments() (per-tool schema validation)
    - validators.parse_plan_json()        (JSON parsing + fence stripping)
    - path_utils.resolve_safe_path()      (sandbox boundary enforcement)
    - registry                            (get_tool, list_tools, is_registered)

Run:
    pytest tests/test_validation.py -v
"""

import json
import os
import pytest
from unittest.mock import patch

import jsonschema
from jsonschema import ValidationError

from utils.validators import validate_plan, validate_step_arguments, parse_plan_json
from utils.path_utils import resolve_safe_path
from registry import get_tool, list_tools, is_registered


# =============================================================================
# validators.validate_plan
# =============================================================================

class TestValidatePlan:
    """Tests for utils.validators.validate_plan"""

    def test_valid_single_step_plan(self):
        """A minimal valid plan passes without error."""
        plan = {
            "steps": [
                {"tool": "math.add", "arguments": {"a": 1, "b": 2}}
            ]
        }
        validate_plan(plan)  # should not raise

    def test_valid_multi_step_plan(self):
        """A multi-step plan with step_id fields passes."""
        plan = {
            "steps": [
                {"step_id": "step1", "tool": "math.add",      "arguments": {"a": 1, "b": 2}},
                {"step_id": "step2", "tool": "math.multiply",  "arguments": {"a": "$prev", "b": 3}},
            ]
        }
        validate_plan(plan)  # should not raise

    def test_rejects_empty_steps_array(self):
        """Empty steps array fails minItems constraint."""
        with pytest.raises(ValidationError):
            validate_plan({"steps": []})

    def test_rejects_missing_steps_key(self):
        """Plan with no 'steps' key fails required constraint."""
        with pytest.raises(ValidationError):
            validate_plan({"tool": "math.add"})

    def test_rejects_missing_tool_in_step(self):
        """A step without 'tool' fails required constraint."""
        with pytest.raises(ValidationError):
            validate_plan({"steps": [{"arguments": {"a": 1, "b": 2}}]})

    def test_rejects_missing_arguments_in_step(self):
        """A step without 'arguments' fails required constraint."""
        with pytest.raises(ValidationError):
            validate_plan({"steps": [{"tool": "math.add"}]})

    def test_rejects_extra_top_level_keys(self):
        """Additional top-level keys fail additionalProperties constraint."""
        with pytest.raises(ValidationError):
            validate_plan({
                "steps": [{"tool": "math.add", "arguments": {"a": 1, "b": 2}}],
                "extra_key": "forbidden",
            })

    def test_rejects_non_string_tool_name(self):
        """Tool name must be a string."""
        with pytest.raises(ValidationError):
            validate_plan({"steps": [{"tool": 42, "arguments": {"a": 1, "b": 2}}]})

    def test_rejects_empty_tool_name(self):
        """Empty tool name fails minLength constraint."""
        with pytest.raises(ValidationError):
            validate_plan({"steps": [{"tool": "", "arguments": {"a": 1, "b": 2}}]})


# =============================================================================
# validators.validate_step_arguments
# =============================================================================

class TestValidateStepArguments:
    """Tests for utils.validators.validate_step_arguments"""

    # ── math.add ──────────────────────────────────────────────────────────

    def test_math_add_valid(self):
        """Valid math.add arguments pass."""
        validate_step_arguments("math.add", {"a": 3, "b": 5})

    def test_math_add_rejects_string_a(self):
        """String 'a' fails type constraint."""
        with pytest.raises(ValidationError, match="is not of type"):
            validate_step_arguments("math.add", {"a": "three", "b": 5})

    def test_math_add_rejects_missing_b(self):
        """Missing 'b' fails required constraint."""
        with pytest.raises(ValidationError):
            validate_step_arguments("math.add", {"a": 3})

    def test_math_add_rejects_extra_key(self):
        """Extra keys fail additionalProperties constraint."""
        with pytest.raises(ValidationError):
            validate_step_arguments("math.add", {"a": 3, "b": 5, "c": 99})

    # ── math.multiply ─────────────────────────────────────────────────────

    def test_math_multiply_valid(self):
        """Valid math.multiply arguments pass."""
        validate_step_arguments("math.multiply", {"a": 4, "b": 3})

    def test_math_multiply_float_valid(self):
        """Float arguments are valid for math.multiply."""
        validate_step_arguments("math.multiply", {"a": 2.5, "b": 4.0})

    # ── string.concat ─────────────────────────────────────────────────────

    def test_string_concat_valid_no_separator(self):
        """Valid string.concat without optional separator."""
        validate_step_arguments("string.concat", {"a": "foo", "b": "bar"})

    def test_string_concat_valid_with_separator(self):
        """Valid string.concat with separator."""
        validate_step_arguments("string.concat", {"a": "foo", "b": "bar", "separator": "-"})

    def test_string_concat_rejects_int_a(self):
        """Integer 'a' fails type constraint."""
        with pytest.raises(ValidationError):
            validate_step_arguments("string.concat", {"a": 123, "b": "bar"})

    # ── datetime.now ──────────────────────────────────────────────────────

    def test_datetime_now_valid_empty(self):
        """datetime.now accepts empty arguments (all optional)."""
        validate_step_arguments("datetime.now", {})

    def test_datetime_now_valid_with_fmt(self):
        """datetime.now accepts a fmt string."""
        validate_step_arguments("datetime.now", {"fmt": "%Y-%m-%d"})

    def test_datetime_now_rejects_int_fmt(self):
        """Non-string fmt fails type constraint."""
        with pytest.raises(ValidationError):
            validate_step_arguments("datetime.now", {"fmt": 20240101})

    # ── file.read ─────────────────────────────────────────────────────────

    def test_file_read_valid(self):
        """Valid relative path passes."""
        validate_step_arguments("file.read", {"path": "sample.txt"})

    def test_file_read_rejects_absolute_path(self):
        """Absolute path fails schema pattern constraint."""
        with pytest.raises(ValidationError):
            validate_step_arguments("file.read", {"path": "/etc/passwd"})

    def test_file_read_rejects_empty_path(self):
        """Empty path string fails minLength constraint."""
        with pytest.raises(ValidationError):
            validate_step_arguments("file.read", {"path": ""})

    # ── Unknown tool ──────────────────────────────────────────────────────

    def test_raises_key_error_for_unknown_tool(self):
        """Unknown tool name raises KeyError."""
        with pytest.raises(KeyError, match="unknown tool"):
            validate_step_arguments("unknown.tool", {"x": 1})


# =============================================================================
# validators.parse_plan_json
# =============================================================================

class TestParsePlanJson:
    """Tests for utils.validators.parse_plan_json"""

    def test_parses_clean_json(self):
        """Clean JSON string is parsed correctly."""
        raw = '{"steps": [{"tool": "math.add", "arguments": {"a": 1, "b": 2}}]}'
        result = parse_plan_json(raw)
        assert result["steps"][0]["tool"] == "math.add"

    def test_strips_markdown_fences(self):
        """JSON wrapped in ```json fences is parsed correctly."""
        raw = '```json\n{"steps": []}\n```'
        # Note: empty steps would fail validate_plan, but parse_plan_json
        # only parses — it does not validate.
        result = parse_plan_json(raw)
        assert "steps" in result

    def test_strips_plain_fences(self):
        """JSON wrapped in plain ``` fences is parsed correctly."""
        raw = '```\n{"key": "value"}\n```'
        result = parse_plan_json(raw)
        assert result["key"] == "value"

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is handled."""
        raw = '   \n{"steps": []}\n   '
        result = parse_plan_json(raw)
        assert "steps" in result

    def test_raises_on_invalid_json(self):
        """Non-JSON string raises json.JSONDecodeError."""
        import json
        with pytest.raises(json.JSONDecodeError):
            parse_plan_json("this is not json")

    def test_raises_on_empty_string(self):
        """Empty string raises json.JSONDecodeError."""
        import json
        with pytest.raises(json.JSONDecodeError):
            parse_plan_json("")


# =============================================================================
# path_utils.resolve_safe_path
# =============================================================================

class TestResolveSafePath:
    """Tests for utils.path_utils.resolve_safe_path"""

    def test_valid_relative_path(self, tmp_path):
        """A relative path inside workspace resolves correctly."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "hello.txt"
        target.write_text("hi", encoding="utf-8")

        result = resolve_safe_path("hello.txt", workspace_root=str(workspace))
        assert result == str(target.resolve())

    def test_valid_nested_path(self, tmp_path):
        """A valid nested relative path resolves correctly."""
        workspace = tmp_path / "workspace"
        subdir = workspace / "subdir"
        subdir.mkdir(parents=True)
        target = subdir / "data.txt"
        target.write_text("data", encoding="utf-8")

        result = resolve_safe_path("subdir/data.txt", workspace_root=str(workspace))
        assert result == str(target.resolve())

    def test_rejects_absolute_path(self, tmp_path):
        """Absolute paths raise PermissionError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with pytest.raises(PermissionError, match="absolute"):
            resolve_safe_path("/etc/passwd", workspace_root=str(workspace))

    def test_rejects_traversal_dotdot_slash(self, tmp_path):
        """'../' traversal raises PermissionError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with pytest.raises(PermissionError, match="traversal"):
            resolve_safe_path("../secret.txt", workspace_root=str(workspace))

    def test_rejects_traversal_embedded(self, tmp_path):
        """Embedded traversal raises PermissionError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with pytest.raises(PermissionError):
            resolve_safe_path("subdir/../../escape.txt", workspace_root=str(workspace))

    def test_raises_file_not_found(self, tmp_path):
        """Non-existent file raises FileNotFoundError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with pytest.raises(FileNotFoundError):
            resolve_safe_path("missing.txt", workspace_root=str(workspace))

    def test_rejects_non_string_input(self, tmp_path):
        """Non-string input raises TypeError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with pytest.raises(TypeError):
            resolve_safe_path(42, workspace_root=str(workspace))


# =============================================================================
# registry
# =============================================================================

class TestRegistry:
    """Tests for registry module."""

    def test_get_tool_returns_callable(self):
        """get_tool returns a callable for each registered tool."""
        for name in list_tools():
            fn = get_tool(name)
            assert callable(fn), f"{name} should be callable"

    def test_get_tool_raises_for_unknown(self):
        """get_tool raises KeyError for unregistered tool names."""
        with pytest.raises(KeyError, match="not registered"):
            get_tool("nonexistent.tool")

    def test_list_tools_returns_all_expected(self):
        """list_tools returns all five MVP tools."""
        tools = list_tools()
        expected = {"math.add", "math.multiply", "string.concat", "datetime.now", "file.read"}
        assert expected.issubset(set(tools)), (
            f"Missing tools: {expected - set(tools)}"
        )

    def test_list_tools_sorted(self):
        """list_tools returns a sorted list."""
        tools = list_tools()
        assert tools == sorted(tools)

    def test_is_registered_true(self):
        """is_registered returns True for known tools."""
        assert is_registered("math.add") is True
        assert is_registered("file.read") is True

    def test_is_registered_false(self):
        """is_registered returns False for unknown tools."""
        assert is_registered("shell.execute") is False
        assert is_registered("") is False

    def test_math_add_via_registry(self):
        """math.add works correctly when invoked through the registry."""
        fn = get_tool("math.add")
        assert fn(a=3, b=5) == 8

    def test_math_multiply_via_registry(self):
        """math.multiply works correctly when invoked through the registry."""
        fn = get_tool("math.multiply")
        assert fn(a=4, b=3) == 12

    def test_string_concat_via_registry(self):
        """string.concat works correctly when invoked through the registry."""
        fn = get_tool("string.concat")
        assert fn(a="Hello", b="World", separator=" ") == "Hello World"
