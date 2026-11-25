"""Tests for logging setup and handler cleanup."""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from email_guardian.logging_setup import setup_logging, shutdown_logging


class LoggingLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name) / "logs"
        self.log_file = self.log_dir / "email_guardian.log"
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        """Close log handlers before deleting the temporary directory.

        unittest runs cleanups last-in-first-out, so a separate
        addCleanup(self._tmp.cleanup) would delete the directory while
        the log file is still open and fail with WinError 32 on Windows.
        """

        shutdown_logging()
        self._tmp.cleanup()

    def _file_handlers(self) -> list[logging.FileHandler]:
        expected = os.path.abspath(str(self.log_file))
        return [
            handler
            for handler in logging.getLogger().handlers
            if isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == expected
        ]

    def test_setup_writes_to_file_and_shutdown_flushes(self) -> None:
        logger = setup_logging(self.log_dir, "INFO")
        logger.info("hello from test")

        self.assertTrue(self.log_file.is_file())
        shutdown_logging()

        content = self.log_file.read_text(encoding="utf-8")
        self.assertIn("hello from test", content)

    def test_shutdown_closes_and_detaches_handlers(self) -> None:
        setup_logging(self.log_dir)
        handlers = self._file_handlers()
        self.assertEqual(len(handlers), 1)

        shutdown_logging()

        self.assertNotIn(handlers[0], logging.getLogger().handlers)
        self.assertIsNone(handlers[0].stream)
        self.assertEqual(self._file_handlers(), [])

        with self.log_file.open("a", encoding="utf-8") as stream:
            stream.write("after-close\n")
        self.assertIn("after-close", self.log_file.read_text(encoding="utf-8"))

    def test_repeated_setup_does_not_leak_handlers(self) -> None:
        setup_logging(self.log_dir)
        first = list(logging.getLogger().handlers)
        first_files = [h for h in first if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(first), 2)

        setup_logging(self.log_dir)
        second = list(logging.getLogger().handlers)

        self.assertEqual(len(second), 2)
        for handler in first:
            self.assertNotIn(handler, second)
        for handler in first_files:
            self.assertIsNone(handler.stream)

    def test_shutdown_is_idempotent(self) -> None:
        shutdown_logging()
        setup_logging(self.log_dir)
        shutdown_logging()
        shutdown_logging()

        self.assertEqual(self._file_handlers(), [])

    def test_shutdown_closes_every_handler_when_one_fails(self) -> None:
        setup_logging(self.log_dir)
        root_handlers = list(logging.getLogger().handlers)
        file_handler = next(h for h in root_handlers if isinstance(h, logging.FileHandler))
        console_handler = next(h for h in root_handlers if not isinstance(h, logging.FileHandler))
        console_handler.flush = MagicMock(side_effect=RuntimeError("flush failed"))

        with self.assertRaises(RuntimeError):
            shutdown_logging()

        self.assertIsNone(file_handler.stream)
        self.assertEqual(self._file_handlers(), [])

    def test_temp_directory_cleanup_succeeds_after_shutdown(self) -> None:
        """Deleting the log directory must work immediately after shutdown.

        This mirrors the Windows failure mode: if the RotatingFileHandler
        still held the file, TemporaryDirectory cleanup would raise
        PermissionError [WinError 32] here.
        """

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            setup_logging(log_dir)
            self.assertTrue((log_dir / "email_guardian.log").is_file())

            shutdown_logging()
            self.assertTrue(log_dir.is_dir())

        self.assertFalse(log_dir.exists())


if __name__ == "__main__":
    unittest.main()
