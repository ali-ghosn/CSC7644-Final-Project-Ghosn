"""
utils/path_utils.py
--------------------
Workspace sandbox enforcement for the AI Task Decomposition Copilot.

SECURITY MODEL
--------------
The file.read tool must ONLY access files that reside under the configured
workspace directory (WORKSPACE_DIR, default: ``./workspace``).

resolve_safe_path() is the single authoritative gate for all file
access decisions.  It:

    1. Rejects non-string inputs immediately.
    2. Rejects absolute paths (starts with "/" or contains drive letters).
    3. Rejects any path containing "../" or "..\\" traversal sequences.
    4. Resolves the path to its canonical absolute form via os.path.realpath().
    5. Confirms the canonical path is a descendant of the canonical workspace root.
    6. Raises PermissionError for every violation — callers must NOT
       catch this exception silently.

The controller loop treats a PermissionError from this module as a
hard failure and stops execution immediately (no retries).
"""

import os
from typing import Optional


def _get_workspace_root() -> str:
    """Return the canonical absolute path of the workspace directory.

    Reads WORKSPACE_DIR from the environment (set via .env / dotenv).
    Falls back to ``./workspace`` relative to the current working directory
    if the variable is not set.

    Returns:
        A canonical absolute path string for the workspace root.
    """
    workspace_env = os.environ.get("WORKSPACE_DIR", "./workspace")
    # realpath resolves symlinks and relative components.
    return os.path.realpath(workspace_env)


def resolve_safe_path(user_path: str, workspace_root: Optional[str] = None) -> str:
    """Validate and resolve a user-supplied path against the workspace sandbox.

    This is the single point of truth for all path security decisions.
    Always call this before opening any file.

    Args:
        user_path:       The raw path string supplied by the planner or user.
        workspace_root:  Optional override for the workspace root directory.
                         Defaults to the value from _get_workspace_root().

    Returns:
        The canonical absolute path of the file, confirmed to reside
        inside the workspace sandbox.

    Raises:
        TypeError:       If user_path is not a string.
        PermissionError: If the path is absolute, contains traversal sequences,
                         or resolves outside the workspace boundary.
        FileNotFoundError: If the resolved path does not exist on disk.

    Example:
        >>> resolve_safe_path("sample.txt")
        '/home/user/project/workspace/sample.txt'

        >>> resolve_safe_path("../etc/passwd")
        PermissionError: ...

        >>> resolve_safe_path("/etc/passwd")
        PermissionError: ...
    """
    if not isinstance(user_path, str):
        raise TypeError(
            f"resolve_safe_path: expected str, got {type(user_path).__name__}"
        )

    root = workspace_root or _get_workspace_root()

    # ── Guard 1: Reject absolute paths early ──────────────────────────────────
    # os.path.isabs catches "/" prefixes on Unix and drive-letter paths on Windows.
    if os.path.isabs(user_path):
        raise PermissionError(
            f"file.read: absolute paths are not permitted. Got: '{user_path}'"
        )

    # ── Guard 2: Reject explicit traversal sequences ──────────────────────────
    # Check the raw string before any normalisation so encoded variants
    # ("%2e%2e") are not silently bypassed here (they'd be blocked by Guard 3,
    # but an early explicit check is clearer).
    normalised = user_path.replace("\\", "/")
    if "../" in normalised or normalised.startswith(".."):
        raise PermissionError(
            f"file.read: path traversal sequences are not permitted. Got: '{user_path}'"
        )

    # ── Guard 3: Canonical boundary check ─────────────────────────────────────
    # Join the workspace root and the user path, then resolve to a canonical
    # absolute path (resolves symlinks, ".", "..", etc.).
    candidate = os.path.realpath(os.path.join(root, user_path))

    # The canonical path must start with the canonical workspace root.
    # We append os.sep to the root so that a directory named "workspace2"
    # is not accidentally accepted as a prefix match for "workspace".
    if not candidate.startswith(root + os.sep) and candidate != root:
        raise PermissionError(
            f"file.read: resolved path escapes the workspace sandbox.\n"
            f"  Requested : '{user_path}'\n"
            f"  Resolved  : '{candidate}'\n"
            f"  Workspace : '{root}'"
        )

    # ── Guard 4: Existence check ───────────────────────────────────────────────
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"file.read: file not found in workspace: '{user_path}'"
        )

    return candidate
