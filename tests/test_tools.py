"""
tests/test_tools.py
--------------------
Pytest test suite for all Phase 1 tool implementations.

Covers:
    - math.add         (valid inputs, type errors)
    - math.multiply    (valid inputs, type errors)
    - string.concat    (valid inputs, separator variants, type errors)
    - datetime.now     (default format, custom format, bad format)
    - file.read        (happy path, missing file, sandbox violations)

Each test group is clearly separated for readability in CI output and
during screen recordings.

Run:
    pytest tests/test_tools.py -v
"""

import os
import re
import pytest
from unittest.mock import patch

# ── Tool imports ──────────────────────────────────────────────────────────
from tools.math_tools import add, multiply
from tools.string_tools import concat
from tools.datetime_tools import now
from tools.file_tools import read


# =============================================================================
# math.add
# =============================================================================

class TestMathAdd:
    """Tests for tools.math_tools.add"""

    def test_add_two_integers(self):
        """Standard integer addition."""
        assert add(3, 5) == 8

    def test_add_two_floats(self):
        """Float addition with tolerance."""
        assert abs(add(1.1, 2.2) - 3.3) < 1e-9

    def test_add_int_and_float(self):
        """Mixed int/float addition."""
        assert add(3, 0.5) == 3.5

    def test_add_negative_numbers(self):
        """Negative operands."""
        assert add(-10, 4) == -6

    def test_add_zeroes(self):
        """Identity element: a + 0 = a."""
        assert add(7, 0) == 7
        assert add(0, 0) == 0

    def test_add_large_numbers(self):
        """Large integer addition."""
        assert add(10**9, 10**9) == 2 * 10**9

    def test_add_rejects_string_a(self):
        """Non-numeric 'a' raises TypeError."""
        with pytest.raises(TypeError, match="math.add"):
            add("three", 5)

    def test_add_rejects_string_b(self):
        """Non-numeric 'b' raises TypeError."""
        with pytest.raises(TypeError, match="math.add"):
            add(3, "five")

    def test_add_rejects_none(self):
        """None input raises TypeError."""
        with pytest.raises(TypeError):
            add(None, 5)


# =============================================================================
# math.multiply
# =============================================================================

class TestMathMultiply:
    """Tests for tools.math_tools.multiply"""

    def test_multiply_two_integers(self):
        """Standard integer multiplication."""
        assert multiply(4, 3) == 12

    def test_multiply_by_zero(self):
        """Anything × 0 = 0."""
        assert multiply(100, 0) == 0

    def test_multiply_floats(self):
        """Float multiplication."""
        assert abs(multiply(2.5, 4.0) - 10.0) < 1e-9

    def test_multiply_negative(self):
        """Negative × positive = negative."""
        assert multiply(-3, 4) == -12

    def test_multiply_two_negatives(self):
        """Negative × negative = positive."""
        assert multiply(-3, -4) == 12

    def test_multiply_rejects_list(self):
        """List input raises TypeError."""
        with pytest.raises(TypeError, match="math.multiply"):
            multiply([1, 2], 3)


# =============================================================================
# string.concat
# =============================================================================

class TestStringConcat:
    """Tests for tools.string_tools.concat"""

    def test_concat_no_separator(self):
        """Default separator is empty string."""
        assert concat("foo", "bar") == "foobar"

    def test_concat_with_space(self):
        """Space separator."""
        assert concat("Hello", "World", separator=" ") == "Hello World"

    def test_concat_with_dash(self):
        """Dash separator."""
        assert concat("2024", "01", separator="-") == "2024-01"

    def test_concat_empty_strings(self):
        """Both operands are empty strings."""
        assert concat("", "") == ""

    def test_concat_empty_a(self):
        """Empty first operand."""
        assert concat("", "world") == "world"

    def test_concat_preserves_whitespace(self):
        """Whitespace inside strings is preserved."""
        assert concat("  a  ", "  b  ") == "  a    b  "

    def test_concat_rejects_int_a(self):
        """Non-string 'a' raises TypeError."""
        with pytest.raises(TypeError, match="string.concat"):
            concat(123, "b")

    def test_concat_rejects_int_b(self):
        """Non-string 'b' raises TypeError."""
        with pytest.raises(TypeError, match="string.concat"):
            concat("a", 456)

    def test_concat_rejects_int_separator(self):
        """Non-string separator raises TypeError."""
        with pytest.raises(TypeError, match="string.concat"):
            concat("a", "b", separator=99)


# =============================================================================
# datetime.now
# =============================================================================

class TestDatetimeNow:
    """Tests for tools.datetime_tools.now"""

    def test_now_default_format(self):
        """Default output matches YYYY-MM-DD HH:MM:SS."""
        result = now()
        assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", result), (
            f"Unexpected format: {result!r}"
        )

    def test_now_date_only_format(self):
        """Custom date-only format."""
        result = now(fmt="%Y-%m-%d")
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", result)

    def test_now_iso_compact(self):
        """ISO compact format."""
        result = now(fmt="%Y%m%dT%H%M%S")
        assert re.match(r"^\d{8}T\d{6}$", result)

    def test_now_rejects_non_string_fmt(self):
        """Non-string fmt raises TypeError."""
        with pytest.raises(TypeError, match="datetime.now"):
            now(fmt=123)

    def test_now_rejects_invalid_format(self):
        """Truly invalid strftime directive raises ValueError.

        Note: Python's strftime silently ignores most unknown directives,
        so we use a format that is known to raise on the current platform.
        We mock strftime to guarantee the ValueError path is exercised.
        """
        from unittest.mock import patch
        from datetime import datetime as real_dt

        class _MockDT:
            def strftime(self, fmt):
                raise ValueError("forced test error")

        with patch("tools.datetime_tools.datetime") as mock_dt:
            mock_dt.now.return_value = _MockDT()
            with pytest.raises(ValueError, match="invalid format string"):
                now(fmt="%invalid%")


# =============================================================================
# file.read
# =============================================================================

class TestFileRead:
    """Tests for tools.file_tools.read

    These tests manipulate WORKSPACE_DIR via environment variable patching
    so they work correctly regardless of the current working directory.
    """

    def test_read_sample_file(self, tmp_path):
        """Happy path: read an existing file from the workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "hello.txt").write_text("hello world", encoding="utf-8")

        with patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}):
            content = read("hello.txt")
        assert content == "hello world"

    def test_read_preserves_multiline_content(self, tmp_path):
        """File contents including newlines are returned verbatim."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "multi.txt").write_text("line1\nline2\nline3", encoding="utf-8")

        with patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}):
            content = read("multi.txt")
        assert content == "line1\nline2\nline3"

    def test_read_raises_file_not_found(self, tmp_path):
        """Non-existent file raises FileNotFoundError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}):
            with pytest.raises(FileNotFoundError):
                read("nonexistent.txt")

    def test_read_rejects_absolute_path(self, tmp_path):
        """Absolute path raises PermissionError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}):
            with pytest.raises(PermissionError):
                read("/etc/passwd")

    def test_read_rejects_path_traversal(self, tmp_path):
        """Path traversal sequence raises PermissionError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}):
            with pytest.raises(PermissionError):
                read("../sensitive.txt")

    def test_read_rejects_double_dot_in_subpath(self, tmp_path):
        """Traversal embedded in subpath raises PermissionError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}):
            with pytest.raises(PermissionError):
                read("subdir/../../etc/passwd")

    def test_read_rejects_directory(self, tmp_path):
        """Reading a directory raises IsADirectoryError."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "subdir").mkdir()

        with patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}):
            with pytest.raises(IsADirectoryError):
                read("subdir")

    def test_read_rejects_non_string_path(self, tmp_path):
        """Non-string path raises TypeError."""
        with pytest.raises(TypeError, match="file.read"):
            read(123)
