"""Logging configuration: console output plus a rotating log file."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

_managed_handlers: list[logging.Handler] = []


def shutdown_logging() -> None:
    """Flush, close and detach every handler installed by setup_logging.

    Closing is required on Windows: an open log file keeps a lock on the
    path, which makes temp-directory cleanup fail with WinError 32.

    Every managed handler is attempted even if one of them fails, so a
    single broken handler cannot keep the others (or their files) open.
    Errors are not suppressed: the first failure is re-raised afterwards.
    """

    root = logging.getLogger()
    errors: list[Exception] = []

    while _managed_handlers:
        handler = _managed_handlers.pop()
        root.removeHandler(handler)
        try:
            handler.flush()
        except Exception as exc:
            errors.append(exc)
        try:
            handler.close()
        except Exception as exc:
            errors.append(exc)

    if errors:
        raise errors[0]


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """Configure root logging and return the application logger.

    Any handlers installed by a previous call are closed before the new
    ones are attached, so repeated calls never leak file handles.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    shutdown_logging()

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        RotatingFileHandler(
            log_dir / "email_guardian.log",
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        ),
    ]
    _managed_handlers.extend(handlers)

    try:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        for handler in handlers:
            handler.setFormatter(formatter)

        root = logging.getLogger()
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        for handler in handlers:
            root.addHandler(handler)
    except Exception:
        shutdown_logging()
        raise

    return logging.getLogger("email_guardian")
