"""
schemas/file_schemas.py
------------------------
JSON Schema definitions for file tool argument blocks.

The schema deliberately restricts ``path`` to relative strings — absolute
paths are rejected at the schema level AND again in path_utils.py as a
defense-in-depth measure.

Schemas defined:
    READ_SCHEMA — arguments for file.read
"""

# ---------------------------------------------------------------------------
# file.read
# ---------------------------------------------------------------------------
READ_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "file.read arguments",
    "description": "Validates arguments passed to the file.read tool.",
    "type": "object",
    "properties": {
        "path": {
            "description": (
                "Relative path to a file inside the workspace directory. "
                "Absolute paths and path traversal sequences are forbidden."
            ),
            "type": "string",
            # Reject absolute paths at schema validation time.
            # Forward-slash absolute paths (/etc/passwd) and Windows-style
            # paths (C:\\...) are blocked by the pattern below.
            # Path traversal ("../") is caught by path_utils at runtime.
            "pattern": r"^(?!\/)(?!.*:[\\/])[\w\-. /\\]+$",
            "minLength": 1,
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}
