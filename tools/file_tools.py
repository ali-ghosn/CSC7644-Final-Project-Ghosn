"""
tools/file_tools.py
--------------------
Sandboxed file I/O tools for the AI Task Decomposition Copilot.

SECURITY MODEL
--------------
file.read is the ONLY permitted file operation.  Writes, deletes, and
directory listings are intentionally excluded from the MVP tool surface.

All read operations are subject to strict path validation enforced by
utils/path_utils.py BEFORE any filesystem interaction occurs:

    - Absolute paths are rejected.
    - Path traversal sequences ("../", "..\\") are rejected.
    - Symlinks that escape the sandbox are rejected.
    - Only paths that resolve inside WORKSPACE_DIR are allowed.

The validation layer in path_utils.resolve_safe_path() raises a
PermissionError for any violation.  This module trusts that layer
completely — no secondary checks are performed here.

Tools exposed:
    file.read — read a plain-text file from the sandboxed workspace
"""

import os

from utils.path_utils import resolve_safe_path


def read(path: str) -> str:
    """Read and return the contents of a sandboxed workspace file.

    The ``path`` argument must be a *relative* path that resolves inside
    the configured workspace directory (default: ``./workspace/``).
    Any attempt to escape the sandbox will raise a ``PermissionError``
    before the filesystem is touched.

    Args:
        path: Relative path to the target file, e.g. ``"sample.txt"``
              or ``"subdir/notes.txt"``.

    Returns:
        The full UTF-8 text content of the file as a single string.

    Raises:
        PermissionError: If the resolved path escapes the workspace sandbox.
        FileNotFoundError: If the file does not exist within the workspace.
        IsADirectoryError: If the resolved path points to a directory.
        UnicodeDecodeError: If the file is not valid UTF-8 text.

    Example:
        >>> read("sample.txt")
        'This is a sample workspace file.\n...'
    """
    if not isinstance(path, str):
        raise TypeError(
            f"file.read: 'path' must be a string, got {type(path).__name__}"
        )

    # Resolve and validate the path — raises PermissionError on any violation.
    safe_path = resolve_safe_path(path)

    # Guard against directory reads (would succeed on some systems).
    if os.path.isdir(safe_path):
        raise IsADirectoryError(
            f"file.read: '{path}' is a directory, not a file."
        )

    # Perform the actual read only after all checks pass.
    with open(safe_path, "r", encoding="utf-8") as fh:
        return fh.read()
