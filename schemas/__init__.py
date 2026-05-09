"""
schemas/__init__.py
--------------------
Marks the schemas/ directory as a Python package and exposes a
convenience mapping from tool name → JSON Schema dict.

This registry-like mapping is used by validators.py to look up the
correct schema for a given tool call without hard-coding tool names
in validation logic.
"""

from schemas.math_schemas import ADD_SCHEMA, MULTIPLY_SCHEMA
from schemas.string_schemas import CONCAT_SCHEMA
from schemas.datetime_schemas import NOW_SCHEMA
from schemas.file_schemas import READ_SCHEMA

# Maps every registered tool name to its argument JSON Schema.
# The controller uses this to validate planner-generated argument blocks.
TOOL_SCHEMAS: dict[str, dict] = {
    "math.add": ADD_SCHEMA,
    "math.multiply": MULTIPLY_SCHEMA,
    "string.concat": CONCAT_SCHEMA,
    "datetime.now": NOW_SCHEMA,
    "file.read": READ_SCHEMA,
}

__all__ = ["TOOL_SCHEMAS"]
