"""
tools/math_tools.py
--------------------
Deterministic math tool implementations for the AI Task Decomposition Copilot.

Each function accepts typed arguments, performs a single well-defined
operation, and returns a plain Python value.  No side effects, no I/O.

Tools exposed:
    math.add      — adds two numbers
    math.multiply — multiplies two numbers
"""

from typing import Union

Number = Union[int, float]


def add(a: Number, b: Number) -> Number:
    """Add two numbers and return their sum.

    Args:
        a: The first operand (int or float).
        b: The second operand (int or float).

    Returns:
        The arithmetic sum of a and b.

    Raises:
        TypeError: If either argument is not a numeric type.

    Example:
        >>> add(3, 5)
        8
        >>> add(2.5, 1.5)
        4.0
    """
    if not isinstance(a, (int, float)):
        raise TypeError(f"math.add: 'a' must be a number, got {type(a).__name__}")
    if not isinstance(b, (int, float)):
        raise TypeError(f"math.add: 'b' must be a number, got {type(b).__name__}")

    return a + b


def multiply(a: Number, b: Number) -> Number:
    """Multiply two numbers and return their product.

    Args:
        a: The first factor (int or float).
        b: The second factor (int or float).

    Returns:
        The arithmetic product of a and b.

    Raises:
        TypeError: If either argument is not a numeric type.

    Example:
        >>> multiply(4, 3)
        12
        >>> multiply(2.5, 4)
        10.0
    """
    if not isinstance(a, (int, float)):
        raise TypeError(f"math.multiply: 'a' must be a number, got {type(a).__name__}")
    if not isinstance(b, (int, float)):
        raise TypeError(f"math.multiply: 'b' must be a number, got {type(b).__name__}")

    return a * b
