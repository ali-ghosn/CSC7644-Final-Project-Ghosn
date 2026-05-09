"""
schemas/string_schemas.py
--------------------------
JSON Schema definitions for string tool argument blocks.

Schemas defined:
    CONCAT_SCHEMA — arguments for string.concat
"""

# ---------------------------------------------------------------------------
# string.concat
# ---------------------------------------------------------------------------
CONCAT_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "string.concat arguments",
    "description": "Validates arguments passed to the string.concat tool.",
    "type": "object",
    "properties": {
        "a": {
            "description": "The first string operand.",
            "type": "string",
        },
        "b": {
            "description": "The second string operand.",
            "type": "string",
        },
        "separator": {
            "description": (
                "Optional string inserted between 'a' and 'b'. "
                "Defaults to empty string when omitted."
            ),
            "type": "string",
            # Default is not enforced by the schema; the tool handles it.
        },
    },
    # 'a' and 'b' are required; 'separator' is optional.
    "required": ["a", "b"],
    "additionalProperties": False,
}
