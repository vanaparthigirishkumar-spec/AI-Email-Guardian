"""Entry point for AI Email Guardian.

Usage:
    python main.py doctor   # verify the local setup
    python main.py run      # start the agent (available in later phases)
"""

from __future__ import annotations

from email_guardian.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
