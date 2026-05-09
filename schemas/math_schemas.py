"""
schemas/math_schemas.py
------------------------
JSON Schema definitions for math tool argument blocks.

Each schema validates the ``arguments`` object in a planner-generated
step — NOT the full step object.  The controller is responsible for
validating the outer step structure; these schemas focus purely on
argument correctness.

Schemas defined:
    ADD_SCHEMA      — arguments for math.add
    MULTIPLY_SCHEMA — arguments for math.multiply
"""

# ---------------------------------------------------------------------------
# math.add
# ---------------------------------------------------------------------------
ADD_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "math.add arguments",
    "description": "Validates arguments passed to the math.add tool.",
    "type": "object",
    "properties": {
        "a": {
            "description": "The first addend.",
            "type": "number",
        },
        "b": {
            "description": "The second addend.",
            "type": "number",
        },
    },
    # Both a and b are mandatory — no defaults are applied by the tool.
    "required": ["a", "b"],
    # Reject any extra keys the planner might hallucinate.
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# math.multiply
# ---------------------------------------------------------------------------
MULTIPLY_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "math.multiply arguments",
    "description": "Validates arguments passed to the math.multiply tool.",
    "type": "object",
    "properties": {
        "a": {
            "description": "The first factor.",
            "type": "number",
        },
        "b": {
            "description": "The second factor.",
            "type": "number",
        },
    },
    "required": ["a", "b"],
    "additionalProperties": False,
}
