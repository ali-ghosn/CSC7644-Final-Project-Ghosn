"""
schemas/datetime_schemas.py
----------------------------
JSON Schema definitions for datetime tool argument blocks.

Schemas defined:
    NOW_SCHEMA — arguments for datetime.now
"""

# ---------------------------------------------------------------------------
# datetime.now
# ---------------------------------------------------------------------------
NOW_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "datetime.now arguments",
    "description": "Validates arguments passed to the datetime.now tool.",
    "type": "object",
    "properties": {
        "fmt": {
            "description": (
                "A strftime-compatible format string. "
                "Defaults to '%Y-%m-%d %H:%M:%S' when omitted."
            ),
            "type": "string",
        },
    },
    # fmt is entirely optional — the tool has a sensible default.
    "required": [],
    "additionalProperties": False,
}
