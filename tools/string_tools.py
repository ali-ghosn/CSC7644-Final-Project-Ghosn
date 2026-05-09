"""
tools/string_tools.py
----------------------
String manipulation tool implementations for the AI Task Decomposition Copilot.

Each function is pure (no side effects) and operates only on the
arguments passed to it.

Tools exposed:
    string.concat — concatenates two strings with an optional separator
"""


def concat(a: str, b: str, separator: str = "") -> str:
    """Concatenate two strings, optionally joined by a separator.

    Args:
        a:         The first string.
        b:         The second string.
        separator: An optional string inserted between a and b.
                   Defaults to "" (no separator).

    Returns:
        A single string composed of a + separator + b.

    Raises:
        TypeError: If a or b is not a string.

    Example:
        >>> concat("Hello", "World", separator=" ")
        'Hello World'
        >>> concat("foo", "bar")
        'foobar'
        >>> concat("2024", "01", separator="-")
        '2024-01'
    """
    if not isinstance(a, str):
        raise TypeError(f"string.concat: 'a' must be a string, got {type(a).__name__}")
    if not isinstance(b, str):
        raise TypeError(f"string.concat: 'b' must be a string, got {type(b).__name__}")
    if not isinstance(separator, str):
        raise TypeError(
            f"string.concat: 'separator' must be a string, got {type(separator).__name__}"
        )

    return f"{a}{separator}{b}"
