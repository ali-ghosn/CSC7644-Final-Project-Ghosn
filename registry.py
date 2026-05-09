"""
registry.py
------------
Central tool registry for the AI Task Decomposition Copilot.

The registry maps fully-qualified tool names (e.g. ``"math.add"``) to the
Python callables that implement them.  It is the single source of truth for
which tools are available and how they are invoked.

Design principles:
    - The controller calls ONLY through the registry — it never imports
      tool modules directly.
    - The planner prompt is generated from TOOL_METADATA (below), ensuring
      that the LLM always has an accurate, up-to-date view of available tools.
    - Adding a new tool requires only: (1) implementing the function in
      tools/, (2) adding one entry to _REGISTRY, and (3) adding one entry
      to TOOL_METADATA.

Usage:
    from registry import get_tool, list_tools, TOOL_METADATA

    fn = get_tool("math.add")
    result = fn(a=3, b=5)          # → 8

    names = list_tools()           # → ["math.add", "math.multiply", ...]
    meta  = TOOL_METADATA          # → list of tool description dicts
"""

from typing import Any, Callable

from tools.math_tools import add, multiply
from tools.string_tools import concat
from tools.datetime_tools import now
from tools.file_tools import read

# ── Internal callable registry ────────────────────────────────────────────
# Maps tool_name → callable.  All callables accept only keyword arguments.

_REGISTRY: dict[str, Callable[..., Any]] = {
    "math.add":       add,
    "math.multiply":  multiply,
    "string.concat":  concat,
    "datetime.now":   now,
    "file.read":      read,
}

# ── Tool metadata (used to build the planner system prompt) ───────────────
# Each entry describes a tool to the LLM without exposing raw JSON Schemas.
# The controller builds the planner prompt from this list at runtime.

TOOL_METADATA: list[dict[str, Any]] = [
    {
        "name": "math.add",
        "description": "Add two numbers together and return their sum.",
        "arguments": {
            "a": {"type": "number", "required": True,  "description": "First addend."},
            "b": {"type": "number", "required": True,  "description": "Second addend."},
        },
        "example": {"a": 3, "b": 5},
        "example_output": 8,
    },
    {
        "name": "math.multiply",
        "description": "Multiply two numbers and return their product.",
        "arguments": {
            "a": {"type": "number", "required": True,  "description": "First factor."},
            "b": {"type": "number", "required": True,  "description": "Second factor."},
        },
        "example": {"a": 4, "b": 3},
        "example_output": 12,
    },
    {
        "name": "string.concat",
        "description": "Concatenate two strings, optionally separated by a delimiter.",
        "arguments": {
            "a":         {"type": "string", "required": True,  "description": "First string."},
            "b":         {"type": "string", "required": True,  "description": "Second string."},
            "separator": {"type": "string", "required": False, "description": "Delimiter string (default: '')."},
        },
        "example": {"a": "Hello", "b": "World", "separator": " "},
        "example_output": "Hello World",
    },
    {
        "name": "datetime.now",
        "description": "Return the current local date and time as a formatted string.",
        "arguments": {
            "fmt": {
                "type": "string",
                "required": False,
                "description": "strftime format string (default: '%Y-%m-%d %H:%M:%S').",
            },
        },
        "example": {"fmt": "%Y-%m-%d"},
        "example_output": "2024-06-15",
    },
    {
        "name": "file.read",
        "description": (
            "Read and return the text content of a file from the sandboxed workspace. "
            "Only relative paths inside ./workspace/ are allowed."
        ),
        "arguments": {
            "path": {
                "type": "string",
                "required": True,
                "description": "Relative path to a file inside ./workspace/ (e.g. 'sample.txt').",
            },
        },
        "example": {"path": "sample.txt"},
        "example_output": "<file contents as string>",
    },
]


# ── Public API ─────────────────────────────────────────────────────────────

def get_tool(tool_name: str) -> Callable[..., Any]:
    """Look up and return the callable for the given tool name.

    Args:
        tool_name: Fully-qualified tool identifier, e.g. ``"math.add"``.

    Returns:
        The Python callable that implements the tool.

    Raises:
        KeyError: If the tool name is not registered.

    Example:
        >>> fn = get_tool("math.add")
        >>> fn(a=3, b=5)
        8
    """
    if tool_name not in _REGISTRY:
        registered = list(_REGISTRY.keys())
        raise KeyError(
            f"Tool '{tool_name}' is not registered. "
            f"Available tools: {registered}"
        )
    return _REGISTRY[tool_name]


def list_tools() -> list[str]:
    """Return a sorted list of all registered tool names.

    Returns:
        Sorted list of tool name strings.

    Example:
        >>> list_tools()
        ['datetime.now', 'file.read', 'math.add', 'math.multiply', 'string.concat']
    """
    return sorted(_REGISTRY.keys())


def is_registered(tool_name: str) -> bool:
    """Check whether a tool name is registered without raising an exception.

    Args:
        tool_name: The tool name to check.

    Returns:
        True if the tool is registered, False otherwise.
    """
    return tool_name in _REGISTRY
