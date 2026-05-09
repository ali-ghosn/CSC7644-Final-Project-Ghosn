"""
tools/datetime_tools.py
------------------------
Date/time tool implementations for the AI Task Decomposition Copilot.

Returns human-readable timestamps in a predictable, configurable format.
This module intentionally avoids timezone complexity — all times are
reported in local system time (suitable for demos and single-machine use).

Tools exposed:
    datetime.now — returns the current date/time as a formatted string
"""

from datetime import datetime


# Default ISO-8601-like format used when no format is specified.
_DEFAULT_FORMAT = "%Y-%m-%d %H:%M:%S"


def now(fmt: str = _DEFAULT_FORMAT) -> str:
    """Return the current local date and time as a formatted string.

    Args:
        fmt: A strftime-compatible format string.
             Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        A string representing the current date/time in the requested format.

    Raises:
        ValueError: If the provided format string is invalid.

    Example:
        >>> now()
        '2024-06-15 14:32:07'
        >>> now(fmt="%B %d, %Y")
        'June 15, 2024'
        >>> now(fmt="%Y%m%dT%H%M%S")
        '20240615T143207'
    """
    if not isinstance(fmt, str):
        raise TypeError(
            f"datetime.now: 'fmt' must be a string, got {type(fmt).__name__}"
        )

    try:
        return datetime.now().strftime(fmt)
    except ValueError as exc:
        raise ValueError(
            f"datetime.now: invalid format string '{fmt}' — {exc}"
        ) from exc
